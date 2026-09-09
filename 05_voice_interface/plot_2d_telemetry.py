"""Standalone 2D telemetry plot, launched as a separate process from a
Mission Control "Graph Launcher" button (docs/CLAUDE.md Stage E "Mission
Control" entry). Loads the MOST RECENT logged trial/chained-flight parquet
under data/processed/stage_e_voice_sessions/ and plots position error,
attitude error, and latency breakdown as a static (not live-streamed)
matplotlib figure.

This is deliberately NOT wired to the live sim via any queue/socket: the
live 3-strip-chart view already exists inside Mission Control's own
browser page (dcc.Graph, updated every ~200ms from the in-process
telemetry bus). This script's job is different - a bigger, standalone
look at the numbers from the most recently COMPLETED run, reusing the
same "read parquet -> matplotlib -> plt.show()" pattern already
established in 03_validation/run_stage_c.py and
04_control/run_stage_d_cascade.py, rather than inventing new
cross-process live telemetry streaming for a secondary feature.

    .venv\\Scripts\\python.exe 05_voice_interface\\plot_2d_telemetry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = ROOT / "data" / "processed" / "stage_e_voice_sessions"


def _latest_parquet() -> Path | None:
    """Most RECENTLY WRITTEN log, by actual file modification time - not
    alphabetical filename sort. This directory holds three different
    filename prefixes (stage_e_trials_, stage_e_chained_, stage_e_
    preemption_), and 't' > 'c' > 'p' alphabetically regardless of the
    timestamp each embeds, so a plain `sorted()` on the whole filename
    would always return a stage_e_trials_* file (if one exists) even when
    a stage_e_chained_* file from a live demo session was written seconds
    ago - silently showing an old headless test run instead of what was
    actually just flown."""
    files = list(SESSIONS_DIR.glob("*.parquet"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def main() -> int:
    path = _latest_parquet()
    if path is None:
        print(f"No logged trial data found under {SESSIONS_DIR}. "
              "Run run_trials.py or fly a Mission Control session first.")
        return 1

    df = pd.read_parquet(path)
    print(f"Plotting {path.name} ({len(df)} rows)")

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    if fig.canvas.manager is not None:
        try:
            fig.canvas.manager.set_window_title("2D Telemetry")
        except Exception:
            pass
    x = df.index

    if "position_error_m" in df.columns:
        axes[0].plot(x, df["position_error_m"], color="tab:blue", marker="o", markersize=3)
        if "latency_total_ms" not in df.columns:
            pass
        axes[0].axhline(0.15, color="tab:orange", linestyle=":", label="0.15m tolerance")
        axes[0].legend(fontsize=8)
    axes[0].set_ylabel("position error (m)")
    axes[0].set_title(f"Stage E telemetry - {path.name}")

    if "attitude_error_deg" in df.columns:
        axes[1].plot(x, df["attitude_error_deg"], color="tab:red", marker="o", markersize=3)
    axes[1].set_ylabel("attitude error (deg)")

    lat_cols = [c for c in df.columns if c.startswith("latency_")]
    for c in lat_cols:
        axes[2].plot(x, df[c], marker=".", markersize=3, label=c.replace("latency_", "").replace("_ms", ""))
    axes[2].set_ylabel("latency (ms)")
    axes[2].set_xlabel("row index (chronological)")
    if lat_cols:
        axes[2].legend(fontsize=7, ncol=2)

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
