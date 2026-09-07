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
03_validation.run_stage_c.freeze_selected_model() rather than
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
for _p in ("01_simulation", "04_control",
           "02_identification", "03_validation"):
    sys.path.insert(0, str(ROOT / _p))

from sim_driver import C_ROT, C_TRANS, SCENE_PATH  # noqa: E402
from controller import QuaternionController, gains_from_identified_model  # noqa: E402
from position_controller import PositionController  # noqa: E402
from trajectory import PointToPointTrajectory  # noqa: E402
from commands import target_offset  # noqa: E402
from sindy_fit import SINDyModel  # noqa: E402

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
            "Run `03_validation/run_stage_c.py` first — it fits, "
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
        return self.data.qpos[0:3].copy()

    def velocity(self) -> np.ndarray:
        return self.data.qvel[0:3].copy()

    def attitude(self) -> np.ndarray:
        return self.data.qpos[3:7].copy()

    def angular_velocity(self) -> np.ndarray:
        return self.data.qvel[3:6].copy()

    def dispatch(self, command: str, measure_traj_latency: bool = True) -> dict:
        """Begin a flight to the waypoint for ``command`` (offset applied
        from the current position). Returns the trajectory-generation
        latency in ms (and 0.0 if not measured)."""
        t0 = time.perf_counter()
        p0 = self.position()
        p1 = p0 + target_offset(command)
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
        self.t_traj += DT
        return 1000.0 * (time.perf_counter() - t0)

    def attitude_error_deg(self, q_target: np.ndarray) -> float:
        from controller import quat_multiply, quat_conjugate
        qe = quat_multiply(quat_conjugate(q_target), self.attitude())
        w = float(np.clip(abs(qe[0]), -1.0, 1.0))
        return float(np.degrees(2 * np.arccos(w)))
