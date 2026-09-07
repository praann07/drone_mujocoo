"""Open-loop excitation trial runner.

Steps the MuJoCo model with hover-baseline + excitation-offset rotor
thrusts, no controller in the loop (per docs/TDD.md section 3 - a closed
loop would correlate control input with state error and bias the fit).
Logs the raw schema from docs/DATA_MODEL.md section 1 at every physics
step (500 Hz, matches docs/TDD.md section 1 sample rate).
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

from excitation import DEFAULT_AMPLITUDE_SCALE, HOVER_THRUST_PER_ROTOR, make_rotor_offsets

MODEL_PATH = Path(__file__).resolve().parent / "models" / "quad.xml"
# scene.xml wraps quad.xml with a ground plane, skybox, and lights - for
# rendering/visualization only (its own docstring says so). It adds no
# bodies, joints, or actuators, so qpos/qvel/ctrl indexing and all physics
# are byte-identical to quad.xml alone; any visualization consumer
# (interactive viewer, offscreen renderer) should load THIS, not
# MODEL_PATH, or it gets a scene with no floor, no lights beyond the
# default headlight, and a dark-gray (rgba 0.2 0.2 0.2) body on a black
# background - i.e. visually nothing.
SCENE_PATH = Path(__file__).resolve().parent / "models" / "scene.xml"
SAMPLE_RATE_HZ = 500.0
U_MAX = 6.0  # N, per docs/TDD.md section 2

# Numeric state/control columns, in the order they are packed into the
# log array below. Matches the raw schema in docs/DATA_MODEL.md section 1.
STATE_COLS = ["px", "py", "pz", "vx", "vy", "vz",
              "qw", "qx", "qy", "qz", "wx", "wy", "wz",
              "u1", "u2", "u3", "u4"]

# Aerodynamic drag, applied as an explicit generalized force each step
# (not in the MJCF - a freejoint's <joint damping="..."> would apply one
# scalar to all 6 DOFs uniformly, which can't give translation and
# rotation different time constants). Without this the rigid body is a
# pure undamped double integrator: any sustained low-frequency excitation
# component causes angular velocity to grow without bound instead of the
# bounded oscillation/decay the Stage A gate wants, and there is no decay
# behavior for SINDy to discover in Stage B. Coefficients chosen for an
# ~0.4-0.8s rotational time constant (a few oscillation cycles visible
# within a 20s trial) and a gentle ~10s translational time constant.
# See docs/TDD.md section 2.
C_ROT = 0.008    # N*m*s/rad, torque = -C_ROT * omega_body
C_TRANS = 0.1    # N*s/m,     force  = -C_TRANS * v_world


def run_trial(axis: str, signal_type: str, trial_id: str, duration_s: float = 20.0,
              amplitude_scale: float = DEFAULT_AMPLITUDE_SCALE, seed: int | None = None) -> pd.DataFrame:
    """Run one open-loop excitation trial and return the raw state log."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    t, offsets = make_rotor_offsets(axis, signal_type, duration_s, SAMPLE_RATE_HZ,
                                     amplitude_scale=amplitude_scale, seed=seed)
    n = len(t)
    log = np.zeros((n, len(STATE_COLS)))
    times = np.zeros(n)

    for i in range(n):
        ctrl = np.clip(HOVER_THRUST_PER_ROTOR + offsets[i], 0.0, U_MAX)
        data.ctrl[:] = ctrl

        times[i] = data.time
        # Free joint qvel: [0:3] linear velocity in WORLD frame,
        # [3:6] angular velocity in BODY frame - matches this project's
        # v (world) / omega (body) convention directly, no conversion.
        log[i, 0:3] = data.qpos[0:3]    # px py pz
        log[i, 3:6] = data.qvel[0:3]    # vx vy vz
        log[i, 6:10] = data.qpos[3:7]   # qw qx qy qz
        log[i, 10:13] = data.qvel[3:6]  # wx wy wz
        log[i, 13:17] = ctrl            # u1..u4

        # qfrc_applied DOF layout matches qvel exactly for a free joint:
        # [0:3] world-frame linear, [3:6] body-frame angular - so drag
        # opposing each is a direct elementwise multiply, no frame conversion.
        data.qfrc_applied[0:3] = -C_TRANS * data.qvel[0:3]
        data.qfrc_applied[3:6] = -C_ROT * data.qvel[3:6]

        mujoco.mj_step(model, data)

    df = pd.DataFrame(log, columns=STATE_COLS)
    df.insert(0, "signal_type", signal_type)
    df.insert(0, "axis", axis)
    df.insert(0, "trial_id", trial_id)
    df.insert(0, "step", np.arange(n))
    df.insert(0, "t", times)
    return df


def save_trial(df: pd.DataFrame, out_dir: Path) -> Path:
    """Write one trial's log. Sample rate is not stored per-file - it is a
    fixed project constant (500 Hz, docs/TDD.md section 1)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{df['trial_id'].iloc[0]}.parquet"
    df.to_parquet(path, index=False)
    return path
