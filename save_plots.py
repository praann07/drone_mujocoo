"""Save 2D telemetry and 3D trajectory plots to results/ without plt.show()."""
import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "05_voice_interface"))

# ── 2D telemetry ──────────────────────────────────────────────────────────────
import pandas as pd
SESSIONS = ROOT / "data" / "processed" / "stage_e_voice_sessions"
latest = sorted(SESSIONS.glob("stage_e_trials_*.parquet"))[-1]
df = pd.read_parquet(latest)

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
axes[0].plot(df.index, df["position_error_m"], color="tab:blue", marker="o", markersize=3)
axes[0].axhline(0.15, color="tab:orange", ls=":", label="0.15m tolerance")
axes[0].set_ylabel("position error (m)"); axes[0].legend(fontsize=8)
axes[0].set_title(f"Stage E telemetry — {latest.name}")
axes[1].plot(df.index, df["attitude_error_deg"], color="tab:red", marker="o", markersize=3)
axes[1].set_ylabel("attitude error (deg)")
lat_cols = [c for c in df.columns if c.startswith("latency_")]
for c in lat_cols:
    axes[2].plot(df.index, df[c], marker=".", markersize=3,
                 label=c.replace("latency_","").replace("_ms",""))
axes[2].set_ylabel("latency (ms)"); axes[2].set_xlabel("trial index")
axes[2].legend(fontsize=7, ncol=2)
for ax in axes: ax.grid(alpha=0.3)
fig.tight_layout()
out2d = ROOT / "results" / "plots_2d" / "stage_e_telemetry_2d.png"
fig.savefig(out2d, dpi=120)
plt.close(fig)
print(f"saved {out2d}")

# ── 3D trajectory (copy already-generated PNG) ────────────────────────────────
import shutil
src = ROOT / "data" / "processed" / "stage_e_demo" / "flight_3d_trajectory.png"
dst = ROOT / "results" / "plots_3d" / "flight_3d_trajectory.png"
shutil.copy2(src, dst)
print(f"copied {dst}")

# ── Stage C phase portrait ────────────────────────────────────────────────────
src2 = ROOT / "data" / "processed" / "stage_c_plots" / "rollout_sindy_long_3d_phase.png"
dst2 = ROOT / "results" / "plots_3d" / "rollout_sindy_long_3d_phase.png"
shutil.copy2(src2, dst2)
print(f"copied {dst2}")
