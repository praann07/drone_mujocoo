"""Maneuver gauntlet: rigorous per-maneuver logging, plots, summary table,
and full-gauntlet GIF. Saves everything to results/maneuvers/.

Sections:
  1. SINGLE MOVES  — 8 commands x 3 runs each from clean hover
  2. CHAINED SEQUENCES — square, stairs, full-mix x 2 runs each
  3. EDGE CASES — rapid-fire, mid-move preemption, stop-resume, altitude limits
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
for _p in ("05_voice_interface", "01_simulation", "01_simulation/models",
           "04_control", "02_identification", "03_validation"):
    sys.path.insert(0, str(ROOT / _p))

from flight import DT, FlightController, IDENTITY_Q  # noqa
from commands import COMMANDS, target_offset, WAYPOINT_OFFSET_M, VERTICAL_OFFSET_M  # noqa
from trajectory import POSITION_TOLERANCE_M  # noqa
import mujoco  # noqa

OUT = ROOT / "results" / "maneuvers"
OUT.mkdir(parents=True, exist_ok=True)

FLIGHT_S = 6.0
TOLERANCE_M = POSITION_TOLERANCE_M  # 0.15 m

# ── helpers ──────────────────────────────────────────────────────────────────

def _fly(fc: FlightController, cmd: str, duration_s: float = FLIGHT_S) -> dict:
    """Dispatch cmd, step for duration_s, return per-step telemetry dict."""
    p0 = fc.position().copy()
    lat = fc.dispatch(cmd)
    target = lat["target"]
    rows = []
    n = int(round(duration_s / DT))
    for i in range(n):
        fc.step()
        p = fc.position()
        v = fc.velocity()
        q = fc.attitude()
        rows.append({"t": i * DT, "px": p[0], "py": p[1], "pz": p[2],
                     "vx": v[0], "vy": v[1], "vz": v[2],
                     "qw": q[0], "qx": q[1], "qy": q[2], "qz": q[3]})
    df = pd.DataFrame(rows)
    achieved = fc.position()
    pos_err = float(np.linalg.norm(achieved - target))
    att_err = fc.attitude_error_deg(IDENTITY_Q)

    # settling time: first index where pos error stays < TOLERANCE_M
    tgt_arr = np.array(target)
    errs = np.linalg.norm(df[["px","py","pz"]].values - tgt_arr, axis=1)
    settle_idx = None
    for i in range(len(errs)):
        if (errs[i:] < TOLERANCE_M).all():
            settle_idx = i
            break
    settle_s = float(df["t"].iloc[settle_idx]) if settle_idx is not None else None

    # overshoot: max excursion beyond target along command axis
    offset = target_offset(cmd)
    axis = np.argmax(np.abs(offset)) if np.any(offset != 0) else None
    if axis is not None and offset[axis] != 0:
        sign = np.sign(offset[axis])
        col = ["px","py","pz"][axis]
        overshoot = float(max(0.0, sign * (df[col].max() if sign > 0 else -df[col].min())
                              - abs(offset[axis])))
    else:
        overshoot = 0.0

    return {
        "df": df, "target": target, "achieved": achieved,
        "pos_err": pos_err, "att_err": att_err,
        "settle_s": settle_s, "overshoot_m": overshoot,
        "success": pos_err <= TOLERANCE_M,
    }


def _plot_xyz(df: pd.DataFrame, target: np.ndarray, title: str, path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    labels = ["X (m)", "Y (m)", "Z (m)"]
    cols = ["px", "py", "pz"]
    colors = ["#22da6e", "#00e5ff", "#ffb86c"]
    for ax, col, lbl, clr, tgt in zip(axes, cols, labels, colors, target):
        ax.plot(df["t"], df[col], color=clr, lw=1.8, label="achieved")
        ax.axhline(tgt, color="#ff5370", ls="--", lw=1.2, label=f"target {tgt:.3f}")
        ax.set_ylabel(lbl, fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.4)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ── Section 1: Single moves ───────────────────────────────────────────────────

print("\n" + "="*70)
print("SECTION 1: SINGLE MOVES (8 commands x 3 runs)")
print("="*70)

single_rows = []
fc = FlightController()

for cmd in COMMANDS:
    for run in range(3):
        fc.reset()
        r = _fly(fc, cmd)
        tag = f"single_{cmd}_run{run}"
        r["df"].to_csv(OUT / f"{tag}.csv", index=False)
        _plot_xyz(r["df"], r["target"], f"{cmd.upper()} run {run+1}  err={r['pos_err']:.4f}m",
                  OUT / f"{tag}.png")
        status = "PASS" if r["success"] else "FAIL"
        print(f"  {cmd:7s} run{run+1}: err={r['pos_err']:.4f}m  settle={r['settle_s']}s"
              f"  overshoot={r['overshoot_m']:.4f}m  att={r['att_err']:.2f}deg  {status}")
        single_rows.append({
            "section": "single", "maneuver": cmd, "run": run+1,
            "pos_err_m": r["pos_err"], "settle_s": r["settle_s"],
            "overshoot_m": r["overshoot_m"], "att_err_deg": r["att_err"],
            "success": r["success"],
        })

# ── Section 2: Chained sequences ─────────────────────────────────────────────

print("\n" + "="*70)
print("SECTION 2: CHAINED SEQUENCES (3 sequences x 2 runs)")
print("="*70)

CHAINS = {
    "square":   ("forward", "left", "back", "right", "hover"),
    "stairs":   ("up", "forward", "up", "forward", "down", "back", "hover"),
    "full_mix": ("left", "up", "right", "down", "forward", "back", "hover"),
}

chain_rows = []

for chain_name, seq in CHAINS.items():
    for run in range(2):
        fc.reset()
        all_dfs = []
        t_offset = 0.0
        leg_results = []
        for cmd in seq:
            r = _fly(fc, cmd)
            df_leg = r["df"].copy()
            df_leg["t"] += t_offset
            df_leg["cmd"] = cmd
            all_dfs.append(df_leg)
            t_offset += FLIGHT_S
            leg_results.append(r)
            status = "PASS" if r["success"] else "FAIL"
            print(f"  {chain_name} run{run+1} [{cmd:7s}]: err={r['pos_err']:.4f}m  {status}")

        full_df = pd.concat(all_dfs, ignore_index=True)
        tag = f"chain_{chain_name}_run{run+1}"
        full_df.to_csv(OUT / f"{tag}.csv", index=False)

        # Plot all 3 axes over the full chain
        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        cols = ["px","py","pz"]; labels = ["X (m)","Y (m)","Z (m)"]
        colors = ["#22da6e","#00e5ff","#ffb86c"]
        for ax, col, lbl, clr in zip(axes, cols, labels, colors):
            ax.plot(full_df["t"], full_df[col], color=clr, lw=1.5)
            ax.set_ylabel(lbl, fontsize=8); ax.grid(True, alpha=0.4)
        # mark command transitions
        for i, cmd in enumerate(seq):
            for ax in axes:
                ax.axvline(i * FLIGHT_S, color="#899bb0", ls=":", lw=0.8)
                ax.text(i * FLIGHT_S + 0.1, ax.get_ylim()[1]*0.9, cmd, fontsize=6, color="#cdd6f4")
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"Chain: {chain_name.upper()} run {run+1}", fontsize=10, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT / f"{tag}.png", dpi=100)
        plt.close(fig)

        final_err = leg_results[-1]["pos_err"]
        all_pass = all(r["success"] for r in leg_results)
        chain_rows.append({
            "section": "chain", "maneuver": chain_name, "run": run+1,
            "final_pos_err_m": final_err,
            "all_legs_pass": all_pass,
            "n_legs": len(seq),
            "n_pass": sum(r["success"] for r in leg_results),
        })

# ── Section 3: Edge cases ─────────────────────────────────────────────────────

print("\n" + "="*70)
print("SECTION 3: EDGE CASES")
print("="*70)

edge_rows = []

# 3a: Rapid-fire — two commands back-to-back with no settle time (0.3s only)
print("\n  3a: Rapid-fire (forward then left, 0.3s between)")
fc.reset()
r1 = _fly(fc, "forward", duration_s=0.3)
r2 = _fly(fc, "left", duration_s=FLIGHT_S)
print(f"     forward(0.3s): err={r1['pos_err']:.4f}m  left(6s): err={r2['pos_err']:.4f}m  "
      f"{'PASS' if r2['success'] else 'FAIL'}")
r2["df"].to_csv(OUT / "edge_rapid_fire.csv", index=False)
_plot_xyz(r2["df"], r2["target"], "Edge: Rapid-fire (fwd 0.3s -> left 6s)", OUT / "edge_rapid_fire.png")
edge_rows.append({"edge": "rapid_fire", "final_err_m": r2["pos_err"], "success": r2["success"]})

# 3b: New command mid-move (preemption)
print("\n  3b: Mid-move preemption (forward 1s -> back 6s)")
fc.reset()
r_fwd = _fly(fc, "forward", duration_s=1.0)
r_back = _fly(fc, "back", duration_s=FLIGHT_S)
print(f"     forward(1s): pos={fc.position()[:2]}  back(6s): err={r_back['pos_err']:.4f}m  "
      f"{'PASS' if r_back['success'] else 'FAIL'}")
r_back["df"].to_csv(OUT / "edge_preemption.csv", index=False)
_plot_xyz(r_back["df"], r_back["target"], "Edge: Mid-move preemption (fwd->back)", OUT / "edge_preemption.png")
edge_rows.append({"edge": "mid_move_preemption", "final_err_m": r_back["pos_err"], "success": r_back["success"]})

# 3c: Stop mid-move then resume
print("\n  3c: Stop mid-move then resume (forward 1s -> stop 2s -> forward 6s)")
fc.reset()
_fly(fc, "forward", duration_s=1.0)
r_stop = _fly(fc, "stop", duration_s=2.0)
r_resume = _fly(fc, "forward", duration_s=FLIGHT_S)
print(f"     stop: err={r_stop['pos_err']:.4f}m  resume forward: err={r_resume['pos_err']:.4f}m  "
      f"{'PASS' if r_resume['success'] else 'FAIL'}")
r_resume["df"].to_csv(OUT / "edge_stop_resume.csv", index=False)
_plot_xyz(r_resume["df"], r_resume["target"], "Edge: Stop mid-move then resume", OUT / "edge_stop_resume.png")
edge_rows.append({"edge": "stop_resume", "final_err_m": r_resume["pos_err"], "success": r_resume["success"]})

# 3d: Down at lowest safe altitude (start at z=1.0, go down 0.5m -> z=0.5m, safe)
print("\n  3d: Down at lowest safe altitude (z=1.0 -> z=0.5m)")
fc.reset()
r_down = _fly(fc, "down", duration_s=FLIGHT_S)
z_min = float(r_down["df"]["pz"].min())
print(f"     z_min={z_min:.4f}m  err={r_down['pos_err']:.4f}m  "
      f"above_floor={'YES' if z_min > 0.05 else 'NO'}  "
      f"{'PASS' if r_down['success'] and z_min > 0.05 else 'FAIL'}")
r_down["df"].to_csv(OUT / "edge_down_low.csv", index=False)
_plot_xyz(r_down["df"], r_down["target"], "Edge: Down at lowest altitude", OUT / "edge_down_low.png")
edge_rows.append({"edge": "down_at_low_alt", "final_err_m": r_down["pos_err"],
                  "z_min_m": z_min, "success": r_down["success"] and z_min > 0.05})

# 3e: Up at highest safe altitude (start at z=1.0, go up 0.5m -> z=1.5m, safe)
print("\n  3e: Up at highest safe altitude (z=1.0 -> z=1.5m)")
fc.reset()
r_up = _fly(fc, "up", duration_s=FLIGHT_S)
z_max = float(r_up["df"]["pz"].max())
print(f"     z_max={z_max:.4f}m  err={r_up['pos_err']:.4f}m  "
      f"no_runaway={'YES' if z_max < 3.0 else 'NO'}  "
      f"{'PASS' if r_up['success'] and z_max < 3.0 else 'FAIL'}")
r_up["df"].to_csv(OUT / "edge_up_high.csv", index=False)
_plot_xyz(r_up["df"], r_up["target"], "Edge: Up at highest altitude", OUT / "edge_up_high.png")
edge_rows.append({"edge": "up_at_high_alt", "final_err_m": r_up["pos_err"],
                  "z_max_m": z_max, "success": r_up["success"] and z_max < 3.0})

# ── Summary table ─────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("GAUNTLET SUMMARY")
print("="*70)

df_single = pd.DataFrame(single_rows)
df_chain = pd.DataFrame(chain_rows)
df_edge = pd.DataFrame(edge_rows)

df_single.to_csv(OUT / "summary_single.csv", index=False)
df_chain.to_csv(OUT / "summary_chain.csv", index=False)
df_edge.to_csv(OUT / "summary_edge.csv", index=False)

n_single = len(df_single); n_single_pass = int(df_single["success"].sum())
n_chain_legs = df_chain["n_legs"].sum(); n_chain_pass = df_chain["n_pass"].sum()
n_edge = len(df_edge); n_edge_pass = int(df_edge["success"].sum())
total = n_single + n_chain_legs + n_edge
total_pass = n_single_pass + n_chain_pass + n_edge_pass

print(f"\nSingle moves:      {n_single_pass}/{n_single} PASS")
print(f"Chain legs:        {n_chain_pass}/{n_chain_legs} PASS")
print(f"Edge cases:        {n_edge_pass}/{n_edge} PASS")
print(f"TOTAL:             {total_pass}/{total} PASS  ({100*total_pass/total:.1f}%)")

# Per-command stats for single moves
print("\nPer-command (single moves, 3 runs each):")
for cmd in COMMANDS:
    sub = df_single[df_single["maneuver"] == cmd]
    mean_err = sub["pos_err_m"].mean()
    mean_settle = sub["settle_s"].mean()
    mean_over = sub["overshoot_m"].mean()
    n_ok = int(sub["success"].sum())
    print(f"  {cmd:7s}: err={mean_err:.4f}m  settle={mean_settle:.3f}s  "
          f"overshoot={mean_over:.4f}m  {n_ok}/3")

gauntlet_pass = (total_pass == total)
print(f"\nGAUNTLET: {'PASS' if gauntlet_pass else 'FAIL'}")

# ── Full-gauntlet GIF (bird's-eye + chase) ────────────────────────────────────

print("\n" + "="*70)
print("RENDERING FULL-GAUNTLET GIF")
print("="*70)

WIDTH, HEIGHT = 640, 480
FPS = 20
STEPS_PER_FRAME = max(1, int(round(1.0 / (FPS * DT))))

# Sequence: all 8 single commands (1 run each) + square chain
GIF_SEQUENCE = list(COMMANDS) + ["forward", "left", "back", "right", "hover"]
GIF_FLIGHT_S = 3.0  # shorter per leg for a watchable GIF

fc2 = FlightController()
renderer = mujoco.Renderer(fc2.model, width=WIDTH, height=HEIGHT)
frames = []
step_count = 0

def _font():
    try: return ImageFont.load_default()
    except: return None

font = _font()

for i, cmd in enumerate(GIF_SEQUENCE):
    # Reset at the very start, and again right before the trailing square
    # chain begins (index len(COMMANDS)) so it starts from a clean origin
    # instead of wherever the 8 single-move tests scattered the drone.
    # (Previous version used `list.index(cmd)`, which always returns the
    # FIRST occurrence of "forward" - i.e. always 0 - so the intended
    # index==len(COMMANDS) branch could never fire; it only "worked" by
    # accident because `cmd == GIF_SEQUENCE[0]` (cmd == "forward") is also
    # true on forward's second occurrence. Same net effect, but dead
    # condition = fragile: fixed to use the loop index directly.)
    if i == 0 or i == len(COMMANDS):
        fc2.reset()
    fc2.dispatch(cmd)
    n_steps = int(round(GIF_FLIGHT_S / DT))
    for _ in range(n_steps):
        fc2.step()
        step_count += 1
        if step_count % STEPS_PER_FRAME == 0:
            renderer.update_scene(fc2.data, camera="chase")
            chase = renderer.render()
            renderer.update_scene(fc2.data, camera="birdseye")
            bird = renderer.render()
            chase_img = Image.fromarray(chase).convert("RGB")
            bird_img = Image.fromarray(bird).convert("RGB")
            combined = Image.new("RGB", (WIDTH * 2, HEIGHT))
            combined.paste(chase_img, (0, 0))
            combined.paste(bird_img, (WIDTH, 0))
            p = fc2.position()
            draw = ImageDraw.Draw(combined)
            draw.text((8, 8), f"cmd: {cmd}", fill=(0,255,0), font=font)
            draw.text((8, 24), f"pos x={p[0]:+.2f} y={p[1]:+.2f} z={p[2]:+.2f}", fill=(0,255,0), font=font)
            draw.text((WIDTH+8, 8), "CHASE (left) | BIRD'S-EYE (right)", fill=(0,255,0), font=font)
            frames.append(combined)

gif_path = OUT / "gauntlet_full.gif"
frames[0].save(gif_path, save_all=True, append_images=frames[1:],
               duration=int(1000/FPS), loop=0)
print(f"  wrote {len(frames)} frames -> {gif_path}")

print(f"\nGAUNTLET {'PASS' if gauntlet_pass else 'FAIL'} — results in {OUT}")
raise SystemExit(0 if gauntlet_pass else 1)
