"""Stage D part 2: full cascade (trajectory planner -> outer position loop
-> inner attitude loop) for all 6 voice-command waypoints, verified on
the identified model then sim-to-sim gap checked against ground-truth
MuJoCo physics (docs/TDD.md section 7b, ENGINEERING_PLAN.md Stage D
tasks 7-8).

    .venv\\Scripts\\python.exe 04_control\\run_stage_d_cascade.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_identification"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_validation"))
from closed_loop_sim import DT, rollout_cascade_on_mujoco, rollout_cascade_on_sindy  # noqa: E402
from controller import QuaternionController, gains_from_identified_model  # noqa: E402
from position_controller import (  # noqa: E402
    KD_POS, KP_POS, MASS_KG, POS_DAMPING_RATIO, POS_SETTLING_TIME_S, PositionController,
)
from run_stage_b import fit_models  # noqa: E402
from run_stage_c import linearize_sindy_at_hover  # noqa: E402
from trajectory import (  # noqa: E402
    MAX_ACCEL_MPS2, MAX_VELOCITY_MPS, POSITION_TOLERANCE_M, WAYPOINT_OFFSET_M, PointToPointTrajectory,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "data" / "processed" / "stage_d_plots"
C_TRANS = 0.1  # matches 01_simulation/sim_driver.py

# Waypoint semantics, world frame (docs/TDD.md section 7b): forward=+X,
# back=-X, left=+Y, right=-Y. hover/stop = stay at current position.
WAYPOINTS = {
    "forward": np.array([WAYPOINT_OFFSET_M, 0.0, 0.0]),
    "back": np.array([-WAYPOINT_OFFSET_M, 0.0, 0.0]),
    "left": np.array([0.0, WAYPOINT_OFFSET_M, 0.0]),
    "right": np.array([0.0, -WAYPOINT_OFFSET_M, 0.0]),
    "hover": np.array([0.0, 0.0, 0.0]),
    "stop": np.array([0.0, 0.0, 0.0]),
}
DURATION_S = 6.0

# Stage D cascade gate tolerance
POS_SETTLING_TOLERANCE_REL = 0.30
POS_FINAL_ERROR_TOLERANCE_M = 0.10


def position_settling_time(t, error_m, band=POSITION_TOLERANCE_M):
    within = error_m <= band
    for i in range(len(within)):
        if within[i:].all():
            return float(t[i])
    return None


def run_all():
    fitted = fit_models(verbose=False)
    sindy = fitted["sindy"]
    A_sindy, _ = linearize_sindy_at_hover(sindy)
    Kq, Komega = gains_from_identified_model(A_sindy)
    inner = QuaternionController(Kq, Komega)
    outer = PositionController()

    print(f"Outer-loop natural frequency: {np.sqrt(KP_POS):.3f} rad/s "
          f"(settling {POS_SETTLING_TIME_S}s, damping {POS_DAMPING_RATIO})")
    print(f"Inner-loop natural frequency: {np.sqrt(Kq[0] / 2):.3f} rad/s "
          f"(from Stage D part 1 gains)")
    print(f"Cascade bandwidth separation (inner/outer): "
          f"{np.sqrt(Kq[0] / 2) / np.sqrt(KP_POS):.2f}x "
          f"(docs/TDD.md section 7b requires outer < inner)")

    p0, v0 = np.zeros(3), np.zeros(3)
    q0, omega0 = np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)

    rows, logs = [], {}
    for name, offset in WAYPOINTS.items():
        p1 = p0 + offset
        traj = PointToPointTrajectory(p0, p1)
        print(f"\nWaypoint: {name} (offset {offset}, trajectory duration {traj.t_total:.2f}s)")

        log_s = rollout_cascade_on_sindy(inner, outer, sindy, MASS_KG, C_TRANS,
                                          p0, v0, q0, omega0, traj, DURATION_S)
        log_m = rollout_cascade_on_mujoco(inner, outer, p0, v0, q0, omega0, traj, DURATION_S)

        err_s = np.linalg.norm(log_s["p"] - p1, axis=1)
        err_m = np.linalg.norm(log_m["p"] - p1, axis=1)
        settle_s = position_settling_time(log_s["t"], err_s)
        settle_m = position_settling_time(log_m["t"], err_m)
        final_s, final_m = float(err_s[-1]), float(err_m[-1])

        print(f"  SINDy -driven cascade: settle={settle_s}, final_error={final_s:.4f} m")
        print(f"  MuJoCo-driven cascade: settle={settle_m}, final_error={final_m:.4f} m")

        rows.append({"waypoint": name, "sindy_settle_s": settle_s, "mujoco_settle_s": settle_m,
                     "sindy_final_error_m": final_s, "mujoco_final_error_m": final_m})
        logs[name] = (log_s, log_m, err_s, err_m)

    df = pd.DataFrame(rows)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PLOTS_DIR / "stage_d_cascade_results.csv", index=False)

    for name in ["forward", "left"]:
        plot_cascade(name, *logs[name], p0 + WAYPOINTS[name])
    return df


def plot_cascade(name, log_s, log_m, err_s, err_m, p_target):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(log_s["p"][:, 0], log_s["p"][:, 1], label="SINDy-driven")
    axes[0].plot(log_m["p"][:, 0], log_m["p"][:, 1], "--", label="MuJoCo-driven")
    axes[0].scatter([0], [0], color="black", marker="o", label="start", zorder=5)
    axes[0].scatter([p_target[0]], [p_target[1]], color="red", marker="x", label="target", zorder=5)
    axes[0].set_xlabel("world X (m)")
    axes[0].set_ylabel("world Y (m)")
    axes[0].set_title(f"{name}: horizontal path")
    axes[0].legend(fontsize=8)
    axes[0].set_aspect("equal")

    axes[1].plot(log_s["t"], err_s, label="SINDy-driven")
    axes[1].plot(log_m["t"], err_m, "--", label="MuJoCo-driven")
    axes[1].axhline(POSITION_TOLERANCE_M, color="gray", linestyle=":", label="tolerance")
    axes[1].set_xlabel("t (s)")
    axes[1].set_ylabel("position error (m)")
    axes[1].set_title(f"{name}: distance to target")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"cascade_{name}.png", dpi=120)
    plt.close(fig)


def gate_check(df: pd.DataFrame) -> bool:
    print("\n" + "=" * 70)
    print("STAGE D CASCADE GATE")
    print("=" * 70)
    passed = True
    for _, r in df.iterrows():
        settled_ok = r.sindy_settle_s is not None and r.mujoco_settle_s is not None
        if settled_ok:
            rel = abs(r.sindy_settle_s - r.mujoco_settle_s) / max(r.mujoco_settle_s, 1e-6)
            settle_close = rel <= POS_SETTLING_TOLERANCE_REL
        else:
            settle_close = False
        final_close = abs(r.sindy_final_error_m - r.mujoco_final_error_m) <= POS_FINAL_ERROR_TOLERANCE_M
        ok = settled_ok and settle_close and final_close
        passed &= ok
        print(f"  {r.waypoint:8s}: settled={settled_ok}  settle_times_close={settle_close}  "
              f"final_error_close={final_close} "
              f"(sindy={r.sindy_final_error_m:.4f}m mujoco={r.mujoco_final_error_m:.4f}m)  "
              f"-> {'PASS' if ok else 'FAIL'}")
    print(f"\nSTAGE D CASCADE GATE: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    df = run_all()
    passed = gate_check(df)
    raise SystemExit(0 if passed else 1)
