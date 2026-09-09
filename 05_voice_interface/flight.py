"""Stage E real-time flight controller (docs/ENGINEERING_PLAN.md Stage E
tasks 3-4). Wraps the ground-truth MuJoCo plant driven THROUGH the Stage D
cascade (outer position loop -> inner quaternion attitude loop), never
bypassing it (docs/CLAUDE.md rules). One ``step()`` advances the plant by
one control period (dt = 0.002 s) exactly as closed_loop_sim.rollout_*
does, so the identified-model-derived gains stay valid.

The plant is MuJoCo but it is accessed only via the cascade controllers,
satisfying the "never wire Stage E directly to the ground-truth simulator,
bypassing the Stage D cascade" rule.

The controller LOADS the frozen model artifact written by
03_validation.run_validation.freeze_selected_model() rather than
re-fitting SINDy from raw excitation parquet files on every launch. This
matters: the whole point of the staged-gate discipline in docs/CLAUDE.md
is that a specific, validated model is what gets to drive a controller or
a demo. Re-fitting cold at Stage E runtime would mean the flight
controller is silently running "whatever STLSQ produces this time" rather
than the exact model that passed Stage B/C - refitting is deterministic
given fixed data (docs/CLAUDE.md "Seeds are deterministic"), so in
practice it would agree, but that is an accident of reproducibility, not
a guarantee, and it costs several seconds of avoidable refitting on every
single launch. If the frozen artifact is missing, this fails loudly
rather than silently falling back to a fresh fit.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in ("01_simulation", "01_simulation/models", "04_control",
           "02_identification", "03_validation"):
    sys.path.insert(0, str(ROOT / _p))

from sim_driver import C_ROT, C_TRANS, SCENE_PATH  # noqa: E402
from controller import QuaternionController, gains_from_identified_model  # noqa: E402
from position_controller import PositionController  # noqa: E402
from trajectory import PointToPointTrajectory  # noqa: E402
from commands import target_offset  # noqa: E402
from sindy_fit import SINDyModel  # noqa: E402
from city import clamp_target_to_safe_zone  # noqa: E402
from closed_loop_sim import quat_rotate_vector  # noqa: E402

DT = 0.002
IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0])
FROZEN_MODEL_PATH = ROOT / "data" / "processed" / "sindy_fitted_model.npz"


def load_frozen_sindy() -> tuple[SINDyModel, np.ndarray]:
    """Load the Stage C-frozen, gate-passed SINDy model + its hover
    Jacobian. Raises rather than silently refitting: a missing artifact
    means Stage C hasn't been (re-)run since the last identification
    change, which is a configuration error the user must fix, not paper
    over with a surprise multi-second cold refit."""
    if not FROZEN_MODEL_PATH.exists():
        raise RuntimeError(
            f"Frozen model artifact not found at {FROZEN_MODEL_PATH}. "
            "Run `03_validation/run_validation.py` first — it fits, "
            "validates against the Stage B/C gates, and freezes the "
            "selected model for Stage E to load. Stage E refuses to "
            "silently refit an unvalidated model at runtime."
        )
    # allow_pickle=True: this file is written only by our own
    # freeze_selected_model() (object-dtype feature_names array needs it
    # to round-trip) and lives under this project's own data/processed/,
    # never fetched from an untrusted or external source.
    data = np.load(FROZEN_MODEL_PATH, allow_pickle=True)
    sindy = SINDyModel(threshold=float(data["threshold"]))
    sindy.coef_ = data["coef"]
    return sindy, data["A_sindy"]


def _sensor_slice(model: "mujoco.MjModel", name: str) -> slice:
    """(start, start+dim) into data.sensordata for a named sensor, looked
    up by id rather than a hardcoded offset - stays correct even if
    quad.xml's <sensor> block is ever reordered."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise RuntimeError(f"quad.xml has no sensor named {name!r} - "
                            "the IMU <sensor> block must define it.")
    adr = model.sensor_adr[sid]
    dim = model.sensor_dim[sid]
    return slice(adr, adr + dim)


class FlightController:
    def __init__(self):
        _sindy, A_sindy = load_frozen_sindy()
        Kq, Komega = gains_from_identified_model(A_sindy)
        self.inner = QuaternionController(Kq, Komega)
        self.outer = PositionController()
        # SCENE_PATH (not MODEL_PATH/quad.xml): adds a ground plane, sky,
        # and lights for actual visibility in the live viewer / offscreen
        # renderer, with byte-identical physics (see sim_driver.py).
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        # quad.xml declares a real IMU (<sensor> block on the "imu" site:
        # framepos/framequat/velocimeter/gyro) - the controller reads
        # state through THESE, not through data.qpos/qvel directly, so
        # "the controller uses the IMU" is literally true, not just
        # modeled-but-unused. Slices are looked up once here, not every
        # step. position/attitude/angular_velocity sensors are direct,
        # noiseless measurements of the exact same quantities qpos/qvel
        # already held, so this changes WHERE the numbers come from, not
        # their value. linear_velocity (the "velocimeter") is the one
        # real exception - see velocity() below.
        self._sens_pos = _sensor_slice(self.model, "position")
        self._sens_att = _sensor_slice(self.model, "attitude")
        self._sens_vel = _sensor_slice(self.model, "linear_velocity")
        self._sens_gyro = _sensor_slice(self.model, "angular_velocity")
        self.traj = None
        self.t_traj = 0.0
        self.command = "hover"
        self.reset()

    def reset(self, p0: np.ndarray | None = None):
        mujoco.mj_resetData(self.model, self.data)
        if p0 is not None:
            self.data.qpos[0:3] = p0
        mujoco.mj_forward(self.model, self.data)
        self.traj = PointToPointTrajectory(self.position(), self.position())
        self.t_traj = 0.0
        self.command = "hover"

    def position(self) -> np.ndarray:
        """World-frame position, read from the IMU's `framepos` sensor
        (not qpos directly) - numerically identical here since the site
        sits at the body origin with no rotation offset, but this is now
        genuinely the sensor reading, not a bypass of it."""
        return self.data.sensordata[self._sens_pos].copy()

    def velocity(self) -> np.ndarray:
        """World-frame velocity. The IMU's `velocimeter` sensor - like a
        real one - reports velocity in the BODY frame, not world frame,
        so it's rotated into world frame here using the (also
        sensor-sourced) attitude - exactly what a real flight computer
        does with an IMU-derived body-frame velocity estimate. This is
        the one accessor where "read the sensor" isn't just a relabeling
        of the same number - the frame transform is real work."""
        v_body = self.data.sensordata[self._sens_vel]
        return quat_rotate_vector(self.attitude(), v_body)

    def attitude(self) -> np.ndarray:
        """Hamilton scalar-first quaternion, read from the IMU's
        `framequat` sensor (not qpos directly) - numerically identical
        here (the site has no rotation offset from the body), but now a
        genuine sensor reading."""
        return self.data.sensordata[self._sens_att].copy()

    def angular_velocity(self) -> np.ndarray:
        """Body-frame angular velocity, read from the IMU's `gyro`
        sensor - the one IMU channel a real gyroscope actually measures
        directly, and already body-frame here exactly like qvel[3:6]
        was, so no transform needed."""
        return self.data.sensordata[self._sens_gyro].copy()

    def dispatch(self, command: str, measure_traj_latency: bool = True) -> dict:
        """Begin a flight to the waypoint for ``command`` (offset applied
        from the current position). Returns the trajectory-generation
        latency in ms (and 0.0 if not measured)."""
        t0 = time.perf_counter()
        p0 = self.position()
        p1_raw = p0 + target_offset(command)
        # Buildings are non-collidable (visual-only, see city.py) - nothing
        # in MuJoCo's physics stops a target from being set inside one.
        # This clamp is the safety net for CHAINED same-direction commands
        # (e.g. "forward" said twice) compounding past a building; a single
        # command from a safe starting point was already proven clear by
        # test_city_buildings_clear_flight_corridor, but that check said
        # nothing about repeats - confirmed live: two "forward"s landed a
        # target at x=3.0m, dead center of the obstacle tower.
        p1 = clamp_target_to_safe_zone(p0, p1_raw)
        if not np.allclose(p1, p1_raw, atol=1e-6):
            print(f"[safety] {command!r} clamped - target would enter a "
                  f"building; stopping short at {p1.round(3).tolist()} "
                  f"instead of {p1_raw.round(3).tolist()}", flush=True)
        self.traj = PointToPointTrajectory(p0, p1)
        self.t_traj = 0.0
        self.command = command
        dt_ms = 1000.0 * (time.perf_counter() - t0) if measure_traj_latency else 0.0
        return {"trajectory": dt_ms, "target": p1}

    def step(self) -> float:
        """Advance one control period. Returns the per-step control latency
        in ms. Reads current state, runs outer -> inner, applies rotor
        commands to MuJoCo, steps physics."""
        t0 = time.perf_counter()
        p = self.position()
        v = self.velocity()
        q = self.attitude()
        omega = self.angular_velocity()

        t = self.t_traj
        p_d = self.traj.position(t)
        v_d = self.traj.velocity(t)
        a_ff = self.traj.acceleration(t)
        q_d, thrust_d = self.outer.compute(p, v, p_d, v_d, a_ff)
        u, _ = self.inner.compute(q, omega, q_d, np.zeros(3), thrust_d)

        self.data.ctrl[:] = u
        self.data.qfrc_applied[0:3] = -C_TRANS * self.data.qvel[0:3]
        self.data.qfrc_applied[3:6] = -C_ROT * self.data.qvel[3:6]
        mujoco.mj_step(self.model, self.data)
        # mj_step() computes sensors from the PRE-integration state (its
        # own internal mj_forward runs before qpos/qvel are advanced), so
        # data.sensordata is one full step stale relative to the qpos/qvel
        # it just integrated to. mj_forward() here recomputes sensors
        # (cheap - no contacts on this model, no integration) from the
        # now-current qpos/qvel, so the NEXT position()/velocity()/etc.
        # call reads a fresh IMU reading, not a one-step-lagged one.
        mujoco.mj_forward(self.model, self.data)
        self.t_traj += DT
        return 1000.0 * (time.perf_counter() - t0)

    def attitude_error_deg(self, q_target: np.ndarray) -> float:
        from controller import quat_multiply, quat_conjugate
        qe = quat_multiply(quat_conjugate(q_target), self.attitude())
        w = float(np.clip(abs(qe[0]), -1.0, 1.0))
        return float(np.degrees(2 * np.arccos(w)))
