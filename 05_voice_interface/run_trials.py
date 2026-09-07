"""Stage E headless trial runner (docs/ENGINEERING_PLAN.md Stage E tasks
5-6). Runs >= 20 end-to-end command trials driving the MuJoCo plant through
the Stage D cascade, logs one row per trial per docs/DATA_MODEL.md section
5, and reports success rate + latency breakdown.

Uses a scripted command source (no microphone needed) so the pipeline and
the identified-model-derived cascade can be validated in this environment;
the live microphone path lives in demo.py.

    .venv\\Scripts\\python.exe 05_voice_interface\\run_trials.py
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commands import COMMANDS, classify  # noqa: E402
from flight import DT, FlightController, IDENTITY_Q  # noqa: E402
from trajectory import POSITION_TOLERANCE_M  # noqa: E402

FLIGHT_DURATION_S = 6.0
N_TRIALS = 24  # 3 per command (8 commands, since up/down were added)
SEED = 20260824

_TRANSCRIPTS = {
    "forward": "go forward please",
    "back": "move back",
    "left": "go left now",
    "right": "turn right and go",
    "up": "go up now",
    "down": "go down now",
    "hover": "hover in place",
    "stop": "stop moving",
}


def _utc():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_all(n_trials: int = N_TRIALS, seed: int = SEED,
            out_dir: Path | None = None) -> pd.DataFrame:
    out_dir = out_dir or (Path(__file__).resolve().parent.parent / "data" / "processed"
                          / "stage_e_voice_sessions")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    command_seq = list(COMMANDS) * (n_trials // len(COMMANDS))
    while len(command_seq) < n_trials:
        command_seq.append(rng.choice(COMMANDS))
    command_seq = list(rng.permutation(command_seq))

    fc = FlightController()
    rows = []
    step_latencies = []

    for i, cmd in enumerate(command_seq):
        transcript = _TRANSCRIPTS[cmd]
        fc.reset()  # each trial starts at origin for a clean measurement

        t0 = time.perf_counter()
        classified = classify(transcript)
        classify_ms = 1000.0 * (time.perf_counter() - t0)

        lat = fc.dispatch(cmd)
        traj_ms = lat["trajectory"]
        target = lat["target"]

        n_steps = int(round(FLIGHT_DURATION_S / DT))
        for _ in range(n_steps):
            step_latencies.append(fc.step())

        achieved_p = fc.position()
        achieved_q = fc.attitude()
        pos_err = float(np.linalg.norm(achieved_p - target))
        att_err = fc.attitude_error_deg(IDENTITY_Q)
        success = pos_err <= POSITION_TOLERANCE_M

        ctrl_ms = float(np.mean(step_latencies)) if step_latencies else 0.0
        step_latencies.clear()

        rows.append({
            "trial_id": f"E_trial_{i:03d}",
            "timestamp_utc": _utc(),
            "transcript": transcript,
            "asr_confidence": 1.0,
            "dispatched_command": classified,
            "start_px": 0.0, "start_py": 0.0, "start_pz": 0.0,
            "target_px": float(target[0]), "target_py": float(target[1]), "target_pz": float(target[2]),
            "target_qw": 1.0, "target_qx": 0.0, "target_qy": 0.0, "target_qz": 0.0,
            "achieved_px": float(achieved_p[0]), "achieved_py": float(achieved_p[1]), "achieved_pz": float(achieved_p[2]),
            "achieved_qw": float(achieved_q[0]), "achieved_qx": float(achieved_q[1]),
            "achieved_qy": float(achieved_q[2]), "achieved_qz": float(achieved_q[3]),
            "position_error_m": pos_err,
            "attitude_error_deg": att_err,
            "success": bool(success),
            "latency_vad_ms": 0.0,
            "latency_asr_ms": 0.0,
            "latency_classify_ms": classify_ms,
            "latency_trajectory_gen_ms": traj_ms,
            "latency_control_dispatch_ms": ctrl_ms,
            "latency_total_ms": classify_ms + traj_ms + ctrl_ms,
        })
        print(f"  trial {i:02d}: cmd={cmd:7s} classified={classified} "
              f"pos_err={pos_err:.4f} m att_err={att_err:.2f} deg -> "
              f"{'PASS' if success else 'FAIL'}")

    df = pd.DataFrame(rows)
    stamp = _utc()
    path = out_dir / f"stage_e_trials_{stamp}.parquet"
    df.to_parquet(path, index=False)
    return df, path


def report(df: pd.DataFrame, path: Path):
    n = len(df)
    n_ok = int(df["success"].sum())
    print("\n" + "=" * 70)
    print("STAGE E VOICE NAVIGATION TRIALS (isolated: fc.reset() before each)")
    print("=" * 70)
    print(f"trials: {n}   success: {n_ok}/{n}  =  {100.0 * n_ok / n:.1f}%")
    print(f"position tolerance: {POSITION_TOLERANCE_M} m")
    print("\nper-command success:")
    for cmd in COMMANDS:
        sub = df[df["dispatched_command"] == cmd]
        if len(sub):
            ok = int(sub["success"].sum())
            print(f"  {cmd:7s}: {ok}/{len(sub)}")
    print("\nlatency breakdown (mean ms):")
    for col in ["latency_classify_ms", "latency_trajectory_gen_ms",
                "latency_control_dispatch_ms", "latency_total_ms"]:
        print(f"  {col:32s}: {df[col].mean():.3f}")
    print(f"\nlog: {path}")
    return n_ok == n


DEFAULT_CHAIN = ("forward", "left", "back", "right", "hover")


def run_chained(sequence=DEFAULT_CHAIN, hold_s: float = FLIGHT_DURATION_S,
                 out_dir: Path | None = None) -> tuple[pd.DataFrame, Path]:
    """Fly `sequence` as ONE continuous flight with NO fc.reset() between
    commands - each command's waypoint is offset from wherever the drone
    actually is when the previous command's hold period ends, exactly
    like render_offscreen.py's COMMAND_SEQUENCE demo.

    This exists because run_all() resets to a clean hover before every
    trial, so its 24/24 headless success rate only ever tests a single
    hop from steady state. render_offscreen.py's navigation GIF - the
    thing anyone watching the demo actually judges - chains five commands
    back-to-back with whatever attitude/velocity error is left over from
    the previous leg. Those are two different tests, and only this one
    matches the qualitative demo, so the quantitative gate needs both.
    """
    out_dir = out_dir or (Path(__file__).resolve().parent.parent / "data" / "processed"
                          / "stage_e_voice_sessions")
    out_dir.mkdir(parents=True, exist_ok=True)

    fc = FlightController()
    fc.reset()  # one clean start for the whole chain, not per-leg
    rows = []

    for i, cmd in enumerate(sequence):
        transcript = _TRANSCRIPTS[cmd]
        p_before = fc.position()

        t0 = time.perf_counter()
        classified = classify(transcript)
        classify_ms = 1000.0 * (time.perf_counter() - t0)

        lat = fc.dispatch(cmd)  # offset applied from CURRENT position, not origin
        target = lat["target"]

        n_steps = int(round(hold_s / DT))
        step_latencies = [fc.step() for _ in range(n_steps)]

        achieved_p = fc.position()
        achieved_q = fc.attitude()
        pos_err = float(np.linalg.norm(achieved_p - target))
        att_err = fc.attitude_error_deg(IDENTITY_Q)
        success = pos_err <= POSITION_TOLERANCE_M

        rows.append({
            "leg_index": i,
            "command": cmd,
            "transcript": transcript,
            "start_position": p_before,
            "target_position": target,
            "achieved_position": achieved_p,
            "achieved_quat": achieved_q,
            "position_error_m": pos_err,
            "attitude_error_deg": att_err,
            "success": bool(success),
            "latency_classify_ms": classify_ms,
            "latency_trajectory_gen_ms": lat["trajectory"],
            "latency_control_dispatch_ms": float(np.mean(step_latencies)),
        })
        print(f"  leg {i} ({cmd:7s}): start={p_before} -> target={target} "
              f"achieved={achieved_p} pos_err={pos_err:.4f} m -> "
              f"{'PASS' if success else 'FAIL'}")

    df = pd.DataFrame(rows)
    stamp = _utc()
    path = out_dir / f"stage_e_chained_{stamp}.parquet"
    # position/quat columns are fixed-length arrays; store as separate
    # scalar columns so the parquet round-trips cleanly like run_all()'s log.
    flat = pd.DataFrame({
        "leg_index": df.leg_index, "command": df.command, "transcript": df.transcript,
        "target_px": df.target_position.apply(lambda a: a[0]),
        "target_py": df.target_position.apply(lambda a: a[1]),
        "target_pz": df.target_position.apply(lambda a: a[2]),
        "achieved_px": df.achieved_position.apply(lambda a: a[0]),
        "achieved_py": df.achieved_position.apply(lambda a: a[1]),
        "achieved_pz": df.achieved_position.apply(lambda a: a[2]),
        "position_error_m": df.position_error_m, "attitude_error_deg": df.attitude_error_deg,
        "success": df.success, "latency_classify_ms": df.latency_classify_ms,
        "latency_trajectory_gen_ms": df.latency_trajectory_gen_ms,
        "latency_control_dispatch_ms": df.latency_control_dispatch_ms,
    })
    flat.to_parquet(path, index=False)
    return df, path


def report_chained(df: pd.DataFrame, path: Path):
    n = len(df)
    n_ok = int(df["success"].sum())
    final_err = float(df["position_error_m"].iloc[-1])
    max_err = float(df["position_error_m"].max())
    print("\n" + "=" * 70)
    print("STAGE E CHAINED FLIGHT (no reset between legs - matches the GIF demo)")
    print("=" * 70)
    print(f"sequence: {' -> '.join(df['command'])}")
    print(f"legs within tolerance: {n_ok}/{n}")
    print(f"final-leg position error: {final_err:.4f} m  (tolerance {POSITION_TOLERANCE_M} m)")
    print(f"max position error across chain: {max_err:.4f} m")
    print(f"\nlog: {path}")
    return n_ok == n


if __name__ == "__main__":
    df, path = run_all()
    ok_isolated = report(df, path)

    df_chain, path_chain = run_chained()
    ok_chained = report_chained(df_chain, path_chain)

    raise SystemExit(0 if (ok_isolated and ok_chained) else 1)
