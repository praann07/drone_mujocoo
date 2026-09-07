"""Stage D part 1: inner-loop quaternion controller, designed against the
SINDy model selected in Stage C, verified on a battery of representative
setpoint trajectories (docs/TDD.md section 7), then sim-to-sim gap
checked against ground-truth MuJoCo physics.

    .venv\\Scripts\\python.exe 04_control\\run_stage_d_inner.py
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
from closed_loop_sim import DT, rollout_on_mujoco, rollout_on_sindy  # noqa: E402
from controller import QuaternionController, gains_from_identified_model  # noqa: E402
from metrics import attitude_error_series, step_response_metrics  # noqa: E402
from mixer import HOVER_THRUST_PER_ROTOR  # noqa: E402
from run_stage_b import fit_models  # noqa: E402
from run_stage_c import linearize_sindy_at_hover  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "data" / "processed" / "stage_d_plots"
THRUST_HOVER = 4 * HOVER_THRUST_PER_ROTOR

# --- Stage D inner-loop gate tolerance --------------------------------
# How far the SINDy-driven controller's metrics may differ from the same
# controller driving ground-truth MuJoCo physics, per docs/TDD.md
# section 7 gate. Settling time in relative terms (the timescale itself
# is what's being checked for transfer); steady-state error in absolute
# degrees (near zero either way, so relative comparison is not meaningful).
SETTLING_TIME_TOLERANCE_REL = 0.30
STEADY_STATE_TOLERANCE_DEG = 1.0


def axis_angle_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return np.array([np.cos(angle_rad / 2), *(np.sin(angle_rad / 2) * axis)])


def step_setpoint(q_target: np.ndarray):
    return lambda t: q_target


def ramp_then_hold_setpoint(axis: np.ndarray, final_angle_rad: float, ramp_duration_s: float):
    def f(t):
        angle = final_angle_rad * min(t / ramp_duration_s, 1.0)
        return axis_angle_quat(axis, angle)
    return f


SCENARIOS = {
    "roll_step_15deg": (step_setpoint(axis_angle_quat(np.array([1, 0, 0]), np.radians(15))), 4.0),
    "pitch_step_15deg": (step_setpoint(axis_angle_quat(np.array([0, 1, 0]), np.radians(15))), 4.0),
    "yaw_step_15deg": (step_setpoint(axis_angle_quat(np.array([0, 0, 1]), np.radians(15))), 4.0),
    "combined_step_15deg": (step_setpoint(axis_angle_quat(np.array([1, 1, 1]), np.radians(15))), 4.0),
    "large_step_60deg": (step_setpoint(axis_angle_quat(np.array([1, 0, 0]), np.radians(60))), 4.0),
    "ramp_roll_to_30deg": (ramp_then_hold_setpoint(np.array([1, 0, 0]), np.radians(30), 3.0), 6.0),
}


def run_all() -> pd.DataFrame:
    fitted = fit_models(verbose=False)
    if not fitted["gate_passed"]:
        raise RuntimeError("Stage B gate not passed - refusing to build Stage D on it.")
    sindy = fitted["sindy"]
    A_sindy, _ = linearize_sindy_at_hover(sindy)
    Kq, Komega = gains_from_identified_model(A_sindy)
    print(f"Gains derived from identified model: Kq={Kq}, Komega={Komega}")
    controller = QuaternionController(Kq, Komega)

    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    omega0 = np.zeros(3)

    rows = []
    example_logs = {}
    for name, (q_fn, duration) in SCENARIOS.items():
        print(f"\nScenario: {name}")
        log_sindy = rollout_on_sindy(controller, sindy, q0, omega0, q_fn, THRUST_HOVER, duration)
        log_mujoco = rollout_on_mujoco(controller, q0, omega0, q_fn, THRUST_HOVER, duration)

        err_sindy = attitude_error_series(controller, log_sindy)
        err_mujoco = attitude_error_series(controller, log_mujoco)
        m_sindy = step_response_metrics(log_sindy["t"], err_sindy)
        m_mujoco = step_response_metrics(log_mujoco["t"], err_mujoco)

        print(f"  SINDy -driven : settle={m_sindy['settling_time_s']}, "
              f"sse={m_sindy['steady_state_error_deg']:.3f} deg, "
              f"min_err={m_sindy['min_error_deg']:.3f} deg")
        print(f"  MuJoCo-driven : settle={m_mujoco['settling_time_s']}, "
              f"sse={m_mujoco['steady_state_error_deg']:.3f} deg, "
              f"min_err={m_mujoco['min_error_deg']:.3f} deg")
        print(f"  Rotor saturation: SINDy={log_sindy['saturated'].any()}, "
              f"MuJoCo={log_mujoco['saturated'].any()}")

        rows.append({"scenario": name,
                     "sindy_settling_s": m_sindy["settling_time_s"],
                     "mujoco_settling_s": m_mujoco["settling_time_s"],
                     "sindy_sse_deg": m_sindy["steady_state_error_deg"],
                     "mujoco_sse_deg": m_mujoco["steady_state_error_deg"],
                     "sindy_settled": m_sindy["settled"],
                     "mujoco_settled": m_mujoco["settled"],
                     "sindy_saturated": bool(log_sindy["saturated"].any()),
                     "mujoco_saturated": bool(log_mujoco["saturated"].any())})
        example_logs[name] = (log_sindy, log_mujoco, err_sindy, err_mujoco)

    df = pd.DataFrame(rows)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PLOTS_DIR / "stage_d_inner_loop_results.csv", index=False)

    for name in ["roll_step_15deg", "large_step_60deg", "ramp_roll_to_30deg"]:
        plot_comparison(name, *example_logs[name])

    return df


def plot_comparison(name, log_sindy, log_mujoco, err_sindy, err_mujoco):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_sindy["t"], err_sindy, label="SINDy-driven")
    ax.plot(log_mujoco["t"], err_mujoco, label="MuJoCo-driven (ground truth)", linestyle="--")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.6, label="settle band (1 deg)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("attitude error (deg)")
    ax.set_title(f"Inner-loop step response: {name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"inner_loop_{name}.png", dpi=120)
    plt.close(fig)


def gate_check(df: pd.DataFrame) -> bool:
    print("\n" + "=" * 70)
    print("STAGE D INNER-LOOP GATE")
    print("=" * 70)
    passed = True
    for _, r in df.iterrows():
        stabilized = r.sindy_settled and r.mujoco_settled
        if r.sindy_settling_s and r.mujoco_settling_s:
            rel_diff = abs(r.sindy_settling_s - r.mujoco_settling_s) / r.mujoco_settling_s
            settling_ok = rel_diff <= SETTLING_TIME_TOLERANCE_REL
        else:
            settling_ok = False
        sse_ok = abs(r.sindy_sse_deg - r.mujoco_sse_deg) <= STEADY_STATE_TOLERANCE_DEG
        ok = stabilized and settling_ok and sse_ok
        passed &= ok
        print(f"  {r.scenario:22s}: stabilized={stabilized}  "
              f"settling_time SINDy={r.sindy_settling_s} MuJoCo={r.mujoco_settling_s} "
              f"(within {SETTLING_TIME_TOLERANCE_REL:.0%}: {settling_ok})  "
              f"sse_diff_ok={sse_ok}  -> {'PASS' if ok else 'FAIL'}")
    print(f"\nSTAGE D INNER-LOOP GATE: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    df = run_all()
    passed = gate_check(df)
    raise SystemExit(0 if passed else 1)
