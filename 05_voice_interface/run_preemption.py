"""Stage E command-preemption (I/O robustness) trial.

Demonstrates that a NEW voice command arriving MID-FLIGHT immediately
overrides the in-progress maneuver - it is not queued behind it. In the
city scene (01_simulation/models/city.py) the forward path now contains a
visual obstacle tower whose face sits at x = +2.3 m, just beyond the +1.5 m
forward waypoint. The script dispatches ``forward``, then re-dispatches
``back`` at t = 0.7 s while the drone is still in transit, and verifies:

  1. the drone reverses immediately (velocity sign-flip latency), never
     reaching the waypoint, let alone the tower face;
  2. it holds the NEW (back) waypoint at the end.

This is the documented behavior of FlightController.dispatch() - it
re-plans PointToPointTrajectory from the CURRENT position and resets the
trajectory clock - so the obstacle is the scenario that makes the property
visible and measurable, not a controller change. demo.py already dispatches
every control step the same way, so interactive "forward... back before I
hit that tower" works identically.

    .venv\\Scripts\\python.exe 05_voice_interface\\run_preemption.py

Writes data/processed/stage_e_voice_sessions/stage_e_preemption_*.parquet
(per-step telemetry) and data/processed/stage_e_demo/robustness_preemption.png.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in ("05_voice_interface", "01_simulation", "01_simulation/models"):
    sys.path.insert(0, str(ROOT / _p))

from flight import DT, FlightController  # noqa: E402
from trajectory import POSITION_TOLERANCE_M  # noqa: E402
from commands import WAYPOINT_OFFSET_M  # noqa: E402
from city import OBSTACLE_FACE_X  # noqa: E402

PREEMPT_AT_S = 0.7       # dispatch "back" this far into the forward leg
FLIGHT_DURATION_S = 8.0  # long enough to reverse AND settle on the new waypoint
REVERSAL_LATENCY_TOL_S = 0.5


def _utc():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_preemption(preempt_at_s: float = PREEMPT_AT_S,
                   flight_s: float = FLIGHT_DURATION_S,
                   out_dir: Path | None = None) -> dict:
    out_dir = out_dir or (ROOT / "data" / "processed" / "stage_e_voice_sessions")
    out_dir.mkdir(parents=True, exist_ok=True)

    fc = FlightController()
    fc.reset()
    fc.dispatch("forward")
    fwd_traj = fc.traj  # keep the original plan for the counterfactual line

    rows = []
    dispatch_step = None
    reversal_step = None
    preempted = False
    back_target = None

    n_steps = int(round(flight_s / DT))
    for i in range(n_steps):
        t = i * DT
        if not preempted and t >= preempt_at_s:
            lat = fc.dispatch("back")  # <-- the mid-flight override
            back_target = lat["target"]
            preempted = True
            dispatch_step = i
        fc.step()
        p = fc.position()
        v = fc.velocity()
        if preempted and reversal_step is None and v[0] <= 0.0:
            reversal_step = i
        rows.append({
            "t": t, "px": p[0], "py": p[1], "pz": p[2],
            "vx": v[0], "vy": v[1], "vz": v[2],
            "command": "back" if preempted else "forward",
        })

    telemetry = pd.DataFrame(rows)
    stamp = _utc()
    log_path = out_dir / f"stage_e_preemption_{stamp}.parquet"
    telemetry.to_parquet(log_path, index=False)

    peak_x = float(telemetry["px"].max())
    t_dispatch = dispatch_step * DT if dispatch_step is not None else None
    t_reversal = reversal_step * DT if reversal_step is not None else None
    reversal_latency = (t_reversal - t_dispatch) if (t_reversal is not None and t_dispatch is not None) else None

    final = telemetry.iloc[-1]
    final_pos = np.array([final.px, final.py, final.pz])
    final_err = float(np.linalg.norm(final_pos - back_target)) if back_target is not None else float("inf")

    summary = {
        "preempt_at_s": preempt_at_s,
        "peak_x_m": peak_x,
        "tower_face_x_m": OBSTACLE_FACE_X,
        "clearance_to_tower_m": OBSTACLE_FACE_X - peak_x,
        "forward_waypoint_m": WAYPOINT_OFFSET_M,
        "dispatch_back_at_s": t_dispatch,
        "reversal_at_s": t_reversal,
        "reversal_latency_s": reversal_latency,
        "final_position_m": final_pos,
        "final_position_error_m": final_err,
        "success": bool(
            peak_x < OBSTACLE_FACE_X
            and peak_x < WAYPOINT_OFFSET_M
            and reversal_latency is not None and reversal_latency <= REVERSAL_LATENCY_TOL_S
            and final_err <= POSITION_TOLERANCE_M
        ),
        "log_path": str(log_path),
        "telemetry": telemetry,
        "fwd_traj": fwd_traj,
        "back_target": back_target,
    }
    return summary


def report(s: dict) -> bool:
    print("\n" + "=" * 70)
    print("STAGE E COMMAND PREEMPTION (mid-flight override)")
    print("=" * 70)
    print(f"sequence: forward at t=0  ->  back at t={s['dispatch_back_at_s']:.2f}s (mid-flight)")
    print(f"peak forward excursion: {s['peak_x_m']:.3f} m  (forward waypoint {s['forward_waypoint_m']:.2f} m,"
          f" tower face {s['tower_face_x_m']:.2f} m)")
    print(f"clearance to tower face at peak: {s['clearance_to_tower_m']:.3f} m")
    rev = s["reversal_latency_s"]
    print(f"reversal latency (back dispatch -> vx<=0): {rev if rev is None else round(rev, 4)} s"
          f"  (tol {REVERSAL_LATENCY_TOL_S} s)")
    print(f"final position: {np.round(s['final_position_m'], 3)}  "
          f"error vs back waypoint: {s['final_position_error_m']:.4f} m  (tol {POSITION_TOLERANCE_M} m)")
    print(f"-> {'PASS' if s['success'] else 'FAIL'}")
    print(f"log: {s['log_path']}")
    return s["success"]


def plot(s: dict, out_dir: Path | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir or (ROOT / "data" / "processed" / "stage_e_demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    telemetry = s["telemetry"]
    back_target = s["back_target"]

    t_disp = s["dispatch_back_at_s"]
    t_rev = s["reversal_at_s"]

    # settle = first time all velocity components drop below 2 cm/s
    sp = np.where((np.abs(telemetry.vx) < 0.02) & (np.abs(telemetry.vy) < 0.02))[0]
    t_end = min(float(telemetry["t"].iloc[-1]), telemetry["t"].iloc[sp[0]] + 1.5) if len(sp) else float(telemetry["t"].iloc[-1])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                   facecolor="#0b0e14")
    for ax in (ax1, ax2):
        ax.set_facecolor("#121722")
        ax.tick_params(colors="#899bb0", labelsize=8)
        for spn in ax.spines.values():
            spn.set_color("#2a364f")
        ax.grid(True, color="#232d3f", linestyle="--", linewidth=0.7, alpha=0.8)

    # x(t) with the forward plan (counterfactual) and the tower's zone.
    tt = telemetry["t"].to_numpy()
    plan_x = np.array([s["fwd_traj"].position(t)[0] for t in tt])
    ax1.fill_between(tt, 0, OBSTACLE_FACE_X, color="#ff5370", alpha=0.18,
                     label="obstacle-tower zone (x >= 2.30 m)")
    ax1.plot(tt, plan_x, color="#899bb0", ls="--", lw=1.2,
             label="forward plan (counterfactual, no preemption)")
    ax1.plot(tt, telemetry["px"], color="#22da6e", lw=2.0, label="actual x(t)")
    ax1.axhline(WAYPOINT_OFFSET_M, color="#00e5ff", ls=":", lw=1.2,
                label=f"forward waypoint (x = {WAYPOINT_OFFSET_M:.1f} m)")
    ax1.axvline(t_disp, color="#ffb86c", ls="-", lw=1.2, label=f"BACK dispatched (t={t_disp:.1f}s)")
    if t_rev is not None:
        ax1.axvline(t_rev, color="#ff5370", ls="-.", lw=1.1, label=f"reversal (t={t_rev:.2f}s)")
    ax1.scatter([t_disp], [s["peak_x_m"]], color="#ffb86c", s=70, zorder=6)
    ax1.annotate(f"peak x = {s['peak_x_m']:.2f} m\n(clearance {s['clearance_to_tower_m']:.2f} m)",
                 (t_disp, s["peak_x_m"]), xytext=(t_disp + 0.35, s["peak_x_m"] + 0.35),
                 color="#cdd6f4", fontsize=8, arrowprops=dict(arrowstyle="->", color="#899bb0"))
    ax1.set_ylabel("X position (m)", color="#cdd6f4", fontsize=9, fontweight="bold")
    ax1.set_ylim(-0.2, OBSTACLE_FACE_X + 0.5)
    ax1.legend(fontsize=7, loc="lower right", facecolor="#121722", labelcolor="#cdd6f4",
               edgecolor="#2a364f")

    # vx(t) with the reversal-latency window shaded.
    ax2.plot(tt, telemetry["vx"], color="#00e5ff", lw=1.8, label="vx(t)")
    if t_rev is not None:
        ax2.axvspan(t_disp, t_rev, color="#ffb86c", alpha=0.25,
                    label=f"reversal latency {s['reversal_latency_s']:.2f} s")
    ax2.axvline(t_disp, color="#ffb86c", ls="-", lw=1.2)
    if t_rev is not None:
        ax2.axvline(t_rev, color="#ff5370", ls="-.", lw=1.1)
    ax2.axhline(0.0, color="#899bb0", lw=0.8)
    ax2.set_ylabel("vx (m/s)", color="#cdd6f4", fontsize=9, fontweight="bold")
    ax2.set_xlabel("Time (s)", color="#cdd6f4", fontsize=9, fontweight="bold")
    ax2.set_xlim(0, t_end)
    ax2.legend(fontsize=7, loc="lower right", facecolor="#121722", labelcolor="#cdd6f4",
               edgecolor="#2a364f")

    fig.suptitle("Command preemption: 'forward' overridden by 'back' mid-flight\n"
                 f"Peak excursion {s['peak_x_m']:.2f} m — never entered the obstacle zone "
                 f"(face at {OBSTACLE_FACE_X:.2f} m)",
                 color="#cdd6f4", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = out_dir / "robustness_preemption.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")
    return out_path


if __name__ == "__main__":
    s = run_preemption()
    ok = report(s)
    plot(s)
    raise SystemExit(0 if ok else 1)