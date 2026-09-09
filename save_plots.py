"""Collects every artifact shown in results/RESULTS.md and README.md's
Visual Evidence gallery into results/, from the canonical files each stage
script already writes under data/processed/. Run this AFTER the full
pipeline (Stage A-E + run_gauntlet.py) so every source file exists - see
REBUILD_RESULTS.bat for the one-click "run everything, then collect it
all" sequence.

Previously this script only handled 3 of the ~24 files that actually live
under results/ - the rest had been placed there by hand in an earlier,
unrepeatable pass. This is the fix: every file in results/ now has an
explicit, traceable source line here, so "how do I get all these results"
has a real, single-command answer.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "05_voice_interface"))

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
for _sub in ("plots_2d", "plots_3d", "videos", "data"):
    (RESULTS / _sub).mkdir(parents=True, exist_ok=True)

_missing: list[str] = []


def _copy(src: Path, dst: Path) -> None:
    if not src.exists():
        _missing.append(str(src.relative_to(ROOT)))
        return
    shutil.copy2(src, dst)
    print(f"  copied {dst.relative_to(ROOT)}")


def _newest(pattern: str) -> Path | None:
    files = sorted((PROCESSED / "voice_sessions").glob(pattern),
                    key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


# ── Stage A/C/D static plots -> results/plots_2d/ ───────────────────────────
print("Stage A/C/D plots:")
for name in ("response_combined.png", "response_pitch.png",
             "response_roll.png", "response_yaw.png"):
    _copy(PROCESSED / "excitation_plots" / name, RESULTS / "plots_2d" / name)

for name in ("eigenvalue_spectrum.png", "head_to_head_near_hover.png",
             "input_output_data_definition.png", "noise_ablation.png",
             "rollout_dmdc_long.png", "rollout_dmdc_short.png",
             "rollout_sindy_long.png", "rollout_sindy_short.png",
             "sparsity_ablation.png"):
    _copy(PROCESSED / "validation_plots" / name, RESULTS / "plots_2d" / name)
_copy(PROCESSED / "validation_plots" / "rollout_sindy_long_3d_phase.png",
      RESULTS / "plots_3d" / "rollout_sindy_long_3d_phase.png")

for name in ("cascade_forward.png", "cascade_left.png",
             "inner_loop_large_step_60deg.png", "inner_loop_ramp_roll_to_30deg.png",
             "inner_loop_roll_step_15deg.png"):
    _copy(PROCESSED / "control_plots" / name, RESULTS / "plots_2d" / name)

_copy(PROCESSED / "voice_demo" / "robustness_preemption.png",
      RESULTS / "plots_2d" / "robustness_preemption.png")
_copy(PROCESSED / "voice_demo" / "robustness_preemption.gif",
      RESULTS / "videos" / "robustness_preemption.gif")
_copy(PROCESSED / "voice_demo" / "flight_3d_trajectory.png",
      RESULTS / "plots_3d" / "flight_3d_trajectory.png")

# ── Raw numeric results -> results/data/ ────────────────────────────────────
print("Numeric results:")
_copy(PROCESSED / "sindy_fitted_model.npz", RESULTS / "data" / "sindy_fitted_model.npz")
_copy(PROCESSED / "identification_results.csv", RESULTS / "data" / "identification_results.csv")
_copy(PROCESSED / "validation_plots" / "rollout_results.csv",
      RESULTS / "data" / "rollout_results.csv")
_copy(PROCESSED / "control_plots" / "cascade_results.csv",
      RESULTS / "data" / "cascade_results.csv")
_copy(PROCESSED / "control_plots" / "inner_loop_results.csv",
      RESULTS / "data" / "inner_loop_results.csv")

for prefix, pattern in (("trials", "trials_*.parquet"),
                        ("chained", "chained_*.parquet"),
                        ("preemption", "preemption_*.parquet")):
    # Fixed "_latest" destination name, not the source's own timestamped
    # name - copying with the source name meant every rerun of this script
    # added ANOTHER file instead of replacing the old one, silently piling
    # up near-duplicate parquet logs under results/data/ forever.
    for stale in (RESULTS / "data").glob(f"{prefix}_*.parquet"):
        stale.unlink()
    src = _newest(pattern)
    if src is not None:
        _copy(src, RESULTS / "data" / f"{prefix}_latest.parquet")
    else:
        _missing.append(f"data/processed/voice_sessions/{pattern} (none found)")

# ── Stage E 2D telemetry - regenerated fresh, not copied ────────────────────
print("Stage E telemetry (regenerated):")
trials_src = _newest("trials_*.parquet")
if trials_src is not None:
    df = pd.read_parquet(trials_src)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(df.index, df["position_error_m"], color="tab:blue", marker="o", markersize=3)
    axes[0].axhline(0.15, color="tab:orange", ls=":", label="0.15m tolerance")
    axes[0].set_ylabel("position error (m)")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"Stage E telemetry - {trials_src.name}")
    axes[1].plot(df.index, df["attitude_error_deg"], color="tab:red", marker="o", markersize=3)
    axes[1].set_ylabel("attitude error (deg)")
    lat_cols = [c for c in df.columns if c.startswith("latency_")]
    for c in lat_cols:
        axes[2].plot(df.index, df[c], marker=".", markersize=3,
                     label=c.replace("latency_", "").replace("_ms", ""))
    axes[2].set_ylabel("latency (ms)")
    axes[2].set_xlabel("trial index")
    axes[2].legend(fontsize=7, ncol=2)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out2d = RESULTS / "plots_2d" / "telemetry_2d.png"
    fig.savefig(out2d, dpi=120)
    plt.close(fig)
    print(f"  saved {out2d.relative_to(ROOT)}")
else:
    _missing.append("data/processed/voice_sessions/trials_*.parquet (none found)")

# ── Summary ──────────────────────────────────────────────────────────────────
if _missing:
    print(f"\n{len(_missing)} source file(s) not found (run the matching stage "
          f"script first - see README's Reproducing the Results section):")
    for m in _missing:
        print(f"  - {m}")
else:
    print("\nAll results/ artifacts collected - nothing missing.")
