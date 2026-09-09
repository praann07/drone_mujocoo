"""Stage A entry point: run all excitation trials, log raw data, build the
trial-level fit/held-out split, and produce the gate-check response plots.

Run from the project root:
    .venv\\Scripts\\python.exe 01_simulation\\run_excitation.py
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
from sim_driver import run_trial, save_trial  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PLOTS_DIR = PROCESSED_DIR / "excitation_plots"

AXES = ["roll", "pitch", "yaw", "combined"]
SIGNAL_TYPES = ["chirp", "prbs"]
TRIALS_PER_COMBO = 4  # 2 fit + 2 held-out, per docs/DATA_MODEL.md section 3
CHIRP_AMPLITUDE_SCALES = [1.0, 0.85, 0.7, 0.55]

# --- Near-hover trial family (for DMDc) -------------------------------
# DMDc fits a LINEAR operator and is declared valid only near hover
# (docs/TDD.md section 5), but the full-envelope trials above deliberately
# reach 90-180 deg - far outside any small-angle regime. Fitting DMDc on
# those and calling it valid would be exactly the "apples-to-oranges"
# comparison TDD.md section 5 forbids, so a dedicated small-amplitude
# family is generated instead. Scales are calibrated PER AXIS because yaw
# and combined excite more total rotation per unit amplitude than
# roll/pitch; 'prbs_fast' (short hold) is used instead of 'prbs' because a
# long-hold PRBS integrates past the bound at any usable amplitude.
# Verified against NEAR_HOVER_MAX_ANGLE_DEG by assert_near_hover_bound().
NEAR_HOVER_SIGNAL_TYPES = ["chirp", "prbs_fast"]
NEAR_HOVER_MAX_ANGLE_DEG = 20.0
NEAR_HOVER_SCALES = {
    "roll": {"chirp": 0.18, "prbs_fast": 0.032},
    "pitch": {"chirp": 0.18, "prbs_fast": 0.032},
    "yaw": {"chirp": 0.10, "prbs_fast": 0.022},
    "combined": {"chirp": 0.08, "prbs_fast": 0.018},
}
# 12s, not the originally-planned 20s: with drag added but no closed-loop
# stabilization (see docs/TDD.md section 3), sustained rotation over a
# full 20s trial triggers many local-coordinate resets (TDD.md section 4).
# 12s keeps that manageable while still covering >1 full cycle of the
# 0.1Hz chirp start frequency. Position/altitude drift over the trial is
# unaffected by this and is out of scope regardless - Stage B fits
# attitude (theta, omega) only, never position.
DURATION_S = 12.0


def quat_rotation_angle_deg(qw, qx, qy, qz):
    """Total rotation angle (deg) from the identity quaternion, via
    theta = 2*acos(|qw|) - deliberately NOT roll/pitch/yaw Euler angles.

    Euler-angle extraction has a gimbal-lock singularity at pitch=+-90 deg
    where the roll/yaw formulas become ill-conditioned and can show large
    spurious swings for what is actually smooth, well-behaved rotation (a
    large-amplitude pitch excitation trial hit exactly this during Stage A
    plotting and showed a spurious 360-degree "roll"/"yaw" excursion for a
    pure-pitch, zero-cross-coupling input). The quaternion itself has no
    such singularity, so a scalar rotation-angle magnitude is used for
    plotting instead - consistent with why Stage B (docs/TDD.md section 4)
    also avoids Euler angles in favor of a quaternion-native coordinate.
    """
    return np.degrees(2 * np.arccos(np.clip(np.abs(qw), -1.0, 1.0)))


ALL_SIGNAL_TYPES = ["chirp", "prbs", "prbs_fast"]


def seed_for_trial(axis: str, signal_type: str, trial_index: int) -> int:
    """Deterministic seed from a stable enumeration, NOT hash().

    Python salts string hashing per process (PYTHONHASHSEED), so
    hash-derived seeds silently change every run - the PRBS dataset would
    not be reproducible, which would make every Stage B/C number
    unreproducible and the stored fit/held-out split meaningless as a
    fixed artifact. Verified cross-process by tests/test_reproducibility.py.
    """
    return (AXES.index(axis) * len(ALL_SIGNAL_TYPES) + ALL_SIGNAL_TYPES.index(signal_type)) \
        * TRIALS_PER_COMBO + trial_index


def run_all_trials() -> pd.DataFrame:
    """Run every trial, save raw parquet logs, return the split table.

    Two regimes are generated: 'full_envelope' (large-angle, for SINDy)
    and 'near_hover' (small-angle, for DMDc - see docs/TDD.md section 5).
    Both use the same trial-level fit/held-out rule.
    """
    split_rows = []
    for axis in AXES:
        for signal_type in SIGNAL_TYPES:
            for trial_index in range(TRIALS_PER_COMBO):
                trial_id = f"A_{axis}_{signal_type}_{trial_index:03d}"
                # Independent chirp trials via distinct amplitude scale
                # (chirp itself is deterministic given f0/f1/duration).
                amplitude_scale = CHIRP_AMPLITUDE_SCALES[trial_index] if signal_type == "chirp" else 1.0
                print(f"running {trial_id} ...")
                df = run_trial(axis, signal_type, trial_id, duration_s=DURATION_S,
                                amplitude_scale=amplitude_scale,
                                seed=seed_for_trial(axis, signal_type, trial_index))
                save_trial(df, RAW_DIR)
                split_rows.append({"trial_id": trial_id, "axis": axis,
                                    "signal_type": signal_type, "regime": "full_envelope",
                                    "split": "fit" if trial_index < 2 else "held_out"})

    for axis in AXES:
        for signal_type in NEAR_HOVER_SIGNAL_TYPES:
            for trial_index in range(TRIALS_PER_COMBO):
                trial_id = f"A_hover_{axis}_{signal_type}_{trial_index:03d}"
                base = NEAR_HOVER_SCALES[axis][signal_type]
                # Mild per-trial variation so the 4 trials are independent
                # rather than 4 identical chirps.
                amplitude_scale = base * (1.0 - 0.15 * trial_index)
                print(f"running {trial_id} ...")
                df = run_trial(axis, signal_type, trial_id, duration_s=DURATION_S,
                                amplitude_scale=amplitude_scale,
                                seed=seed_for_trial(axis, signal_type, trial_index))
                save_trial(df, RAW_DIR)
                split_rows.append({"trial_id": trial_id, "axis": axis,
                                    "signal_type": signal_type, "regime": "near_hover",
                                    "split": "fit" if trial_index < 2 else "held_out"})

    split_df = pd.DataFrame(split_rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    split_df.to_parquet(PROCESSED_DIR / "trial_split.parquet", index=False)
    return split_df


def assert_near_hover_bound(split_df: pd.DataFrame) -> float:
    """Every near_hover trial must stay within NEAR_HOVER_MAX_ANGLE_DEG.

    This is what makes DMDc's declared validity regime (docs/TDD.md
    section 5) an enforced property of the data rather than a claim in a
    document - if the bound is violated, Stage B must not proceed to fit
    DMDc on this data and call it near-hover.
    """
    worst, worst_id = 0.0, None
    for trial_id in split_df.loc[split_df.regime == "near_hover", "trial_id"]:
        df = pd.read_parquet(RAW_DIR / f"{trial_id}.parquet")
        peak = quat_rotation_angle_deg(df.qw.values, df.qx.values, df.qy.values, df.qz.values).max()
        if peak > worst:
            worst, worst_id = peak, trial_id
    if worst > NEAR_HOVER_MAX_ANGLE_DEG:
        raise AssertionError(
            f"near-hover regime violated: {worst_id} peaks at {worst:.1f} deg "
            f"> {NEAR_HOVER_MAX_ANGLE_DEG} deg. Reduce NEAR_HOVER_SCALES for that axis."
        )
    return worst


def plot_axis_response(axis: str):
    """One figure per axis: quaternion components, rotation-angle
    magnitude, and angular velocity for a representative chirp and PRBS
    trial - the Stage A gate check. Quaternion-native throughout (no
    Euler angles) to avoid gimbal-lock plotting artifacts - see
    quat_rotation_angle_deg."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    fig.suptitle(f"Stage A excitation response — axis: {axis}")

    for col, signal_type in enumerate(SIGNAL_TYPES):
        trial_id = f"A_{axis}_{signal_type}_000"
        df = pd.read_parquet(RAW_DIR / f"{trial_id}.parquet")
        angle_deg = quat_rotation_angle_deg(df.qw.values, df.qx.values, df.qy.values, df.qz.values)

        def panel(ax, cols, title, ylabel):
            for c in cols:
                ax.plot(df.t, df[c], label=c)
            ax.set_title(f"{signal_type} — {title}")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)

        panel(axes[0, col], ["qx", "qy", "qz"], "quaternion vector part", "(unitless)")

        axes[1, col].plot(df.t, angle_deg, color="black")
        axes[1, col].set_title(f"{signal_type} — total rotation angle from identity")
        axes[1, col].set_ylabel("deg")

        panel(axes[2, col], ["wx", "wy", "wz"], "angular velocity", "rad/s")
        axes[2, col].set_xlabel("t (s)")

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / f"response_{axis}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def summarize_gate_stats(split_df: pd.DataFrame) -> pd.DataFrame:
    """Peak-to-peak rotation angle and angular-velocity RMS per trial -
    quantitative backing for the 'rich oscillation, not flat' gate.
    Rotation angle uses quat_rotation_angle_deg (singularity-free), not
    Euler angles."""
    stats = []
    for _, row in split_df.iterrows():
        df = pd.read_parquet(RAW_DIR / f"{row.trial_id}.parquet")
        angle_deg = quat_rotation_angle_deg(df.qw.values, df.qx.values, df.qy.values, df.qz.values)
        stats.append({
            "trial_id": row.trial_id, "axis": row.axis, "signal_type": row.signal_type,
            "split": row.split,
            "rotation_angle_ptp_deg": np.ptp(angle_deg),
            "wx_rms": np.sqrt(np.mean(df.wx**2)), "wy_rms": np.sqrt(np.mean(df.wy**2)),
            "wz_rms": np.sqrt(np.mean(df.wz**2)),
        })
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(PLOTS_DIR / "gate_stats.csv", index=False)
    return stats_df


if __name__ == "__main__":
    split_df = run_all_trials()
    print(f"\n{len(split_df)} trials complete. Split:\n"
          f"{split_df.groupby(['regime', 'split']).size()}")

    worst = assert_near_hover_bound(split_df)
    print(f"\nnear-hover regime OK: worst peak rotation {worst:.1f} deg "
          f"(bound {NEAR_HOVER_MAX_ANGLE_DEG} deg)")

    for axis in AXES:
        print(f"wrote {plot_axis_response(axis)}")

    stats_df = summarize_gate_stats(split_df)
    print("\nGate stats (peak-to-peak rotation angle, angular velocity RMS):")
    print(stats_df.to_string(index=False))
