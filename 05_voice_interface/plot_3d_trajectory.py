"""Standalone 3D trajectory plot ("3D trajectory over the city model"), run
directly as its own process/window (docs/CLAUDE.md Stage E history).

This is an independent MATPLOTLIB analysis artifact, not a re-render or
stream of MuJoCo's own 3D scene - the native MuJoCo passive viewer stays
the ONLY live 3D view of the sim. This script instead plots the flight
PATH from the most recently logged chained-flight parquet(s)
(data/processed/voice_sessions/chained_*.parquet - a
genuine continuous multi-leg path, unlike the isolated-trial log where
every trial resets to the origin), overlaying up to the last
MAX_FLIGHTS_OVERLAID flights in different colors, together with the SAME
city model that scene.xml now renders (imported from
01_simulation/models/city.py - buildings + obstacle tower as simple bars),
so the plot has the identical spatial layout/scale the 3D scene shows and
the drone visibly flies the central plaza clear of every building footprint.
Shows both an isometric view and a top-down/bird's-eye view side by side.

    .venv\\Scripts\\python.exe 05_voice_interface\\plot_3d_trajectory.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

ROOT = Path(__file__).resolve().parent.parent
for _p in ("05_voice_interface", "01_simulation/models"):
    sys.path.insert(0, str(ROOT / _p))
SESSIONS_DIR = ROOT / "data" / "processed" / "voice_sessions"
from city import CITY_OBJECTS  # noqa: E402


MAX_FLIGHTS_OVERLAID = 3
_FLIGHT_COLORS = ["tab:blue", "tab:purple", "tab:cyan"]


def _recent_chained_parquets(n: int = MAX_FLIGHTS_OVERLAID) -> tuple[list[Path], bool]:
    """Returns (paths, is_continuous), most recent LAST. Chained-flight
    logs are one genuine continuous path; isolated-trial logs reset to
    the origin before every trial, so plotting a line through consecutive
    rows there would show misleading teleport jumps rather than real
    flight - used only as a fallback when no chained log exists at all,
    and flagged as such (and only the single most recent one, since
    overlaying multiple discontinuous isolated-trial logs would compound
    the misleading-jump problem)."""
    files = sorted(SESSIONS_DIR.glob("chained_*.parquet"))
    if files:
        return files[-n:], True
    files = sorted(SESSIONS_DIR.glob("trials_*.parquet"))
    return ([files[-1]], False) if files else ([], False)


def _draw_city(ax):
    for obj in CITY_OBJECTS:
        if obj.kind == "tower":
            color = "#e8590c"
            label = "obstacle tower"
        elif obj.kind == "building":
            color = "#495057"
            label = None
        else:
            continue
        ax.bar3d(obj.x - obj.half_x, obj.y - obj.half_y, 0,
                 2 * obj.half_x, 2 * obj.half_y, obj.height,
                 color=color, alpha=0.55, shade=True, label=label)


def _plot_flights(ax, flights: list[tuple[pd.DataFrame, str, str]]):
    _draw_city(ax)
    for df, label, color in flights:
        xs, ys, zs = df["achieved_px"], df["achieved_py"], df["achieved_pz"]
        ax.plot(xs, ys, zs, color=color, linewidth=2, marker="o", markersize=3, label=label)
        ax.scatter([xs.iloc[0]], [ys.iloc[0]], [zs.iloc[0]], color="green", s=70,
                   zorder=5, label="start" if df is flights[0][0] else None)
        ax.scatter([xs.iloc[-1]], [ys.iloc[-1]], [zs.iloc[-1]], color="red", s=70,
                   zorder=5, label="end" if df is flights[0][0] else None)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")


def main() -> int:
    paths, is_continuous = _recent_chained_parquets()
    if not paths:
        print(f"No logged flight data found under {SESSIONS_DIR}. "
              "Run run_trials.py or fly a demo.py session first.")
        return 1
    if not is_continuous:
        print("WARNING: no chained-flight log found - falling back to an "
              "isolated-trial log, where each trial resets to the origin. "
              "The plotted line will show teleport jumps between trials, "
              "not a real continuous path. Run run_chained() (or fly "
              "several commands in one demo.py session) for a real "
              "trajectory.")

    flights = []
    for i, path in enumerate(paths):
        df = pd.read_parquet(path)
        label = f"{path.name} ({len(df)} rows)"
        color = _FLIGHT_COLORS[i % len(_FLIGHT_COLORS)]
        flights.append((df, label, color))
        print(f"  overlay {i + 1}/{len(paths)}: {label}")

    fig = plt.figure(figsize=(14, 7))
    if fig.canvas.manager is not None:
        try:
            fig.canvas.manager.set_window_title("3D Trajectory")
        except Exception:
            pass

    # Left: isometric view (the default 3D matplotlib angle) - gives depth
    # and altitude a sense of scale. Right: bird's-eye/top-down view
    # (elev=90 looks straight down the Z axis) - reads like a map, makes
    # the actual XY path shape and city-bar footprints unambiguous, which
    # the isometric view alone can foreshorten or occlude.
    ax_iso = fig.add_subplot(121, projection="3d")
    _plot_flights(ax_iso, flights)
    ax_iso.set_title("Isometric view")
    ax_iso.legend(fontsize=7, loc="upper left")

    ax_top = fig.add_subplot(122, projection="3d")
    _plot_flights(ax_top, flights)
    ax_top.view_init(elev=90, azim=-90)
    ax_top.set_title("Top-down (bird's-eye) view")

    fig.suptitle(f"3D flight trajectory over the city model"
                 + (f" ({len(flights)} overlaid flights)" if len(flights) > 1 else ""))
    fig.tight_layout()

    # Save an artifact copy in addition to showing it live, matching every
    # other analysis plot in the project (validation_plots/, control_plots/)
    # - so there's always a reproducible file to embed/inspect even if no
    # one is watching the window when it's generated.
    out_dir = ROOT / "data" / "processed" / "voice_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "flight_3d_trajectory.png"
    fig.savefig(out_path, dpi=120)
    print(f"  wrote {out_path}")

    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
