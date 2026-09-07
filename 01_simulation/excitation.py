"""PRBS and chirp excitation signal generators.

Signals are generated as *differential rotor thrust offsets* around hover,
per docs/TDD.md section 3: frequency range 0.1-15 Hz, 20s trials.

Per-axis base amplitudes are calibrated (not a single flat fraction of
hover thrust) because roll/pitch and yaw have very different control
authority - roll/pitch torque comes from the full moment arm l/sqrt(2),
yaw only from the much smaller reaction-torque coefficient k_m. A flat
amplitude either saturates roll/pitch into repeated full inversions or
leaves yaw producing a barely-visible response. Each axis's base amplitude
is chosen so peak angular acceleration is ~12 rad/s^2 (a few hundred
deg/s^2 - aggressive but physically plausible, not a full-flip regime).
See docs/TDD.md section 3 for the derivation.

All signals returned as (t, offsets) where offsets has shape (len(t), 4)
- one differential thrust offset per rotor (u1..u4), added to the hover
baseline by the sim driver, never a fixed absolute thrust command here.
"""
from __future__ import annotations

import numpy as np

HOVER_THRUST_PER_ROTOR = 1.0 * 9.81 / 4.0  # N, mass=1.0kg per TDD.md section 2

# Differential mixing directions: how a scalar excitation signal e(t) maps
# onto the 4 rotors to excite roll, pitch, yaw, or all three at once.
# Signs chosen to directly excite the corresponding body torque per the
# mixer in docs/TDD.md section 2 (tau_x ~ u1-u2-u3+u4, tau_y ~ -u1+u2-u3+u4,
# tau_z ~ u1+u2-u3-u4).
AXIS_MIX = {
    "roll": np.array([1.0, -1.0, -1.0, 1.0]),
    "pitch": np.array([-1.0, 1.0, -1.0, 1.0]),
    "yaw": np.array([1.0, 1.0, -1.0, -1.0]),
}

# Base amplitude (N) per axis, calibrated per docs/TDD.md section 3 so peak
# angular acceleration is ~12 rad/s^2 on each axis despite yaw's much
# weaker control authority (k_m=0.02 vs roll/pitch moment arm 0.106m).
AXIS_BASE_AMPLITUDE_N = {
    "roll": 0.10,
    "pitch": 0.10,
    "yaw": 1.00,
}
DEFAULT_AMPLITUDE_SCALE = 1.0  # multiplier on top of the base amplitude above


def chirp(duration_s: float, sample_rate_hz: float, f0_hz: float, f1_hz: float,
          amplitude: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear-sweep chirp from f0_hz to f1_hz over duration_s."""
    t = np.arange(0.0, duration_s, 1.0 / sample_rate_hz)
    k = (f1_hz - f0_hz) / duration_s
    phase = 2 * np.pi * (f0_hz * t + 0.5 * k * t**2)
    signal = amplitude * np.sin(phase)
    return t, signal


def prbs(duration_s: float, sample_rate_hz: float, min_period_s: float, max_period_s: float,
          amplitude: float, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Pseudo-random binary sequence with hold time drawn uniformly in
    [min_period_s, max_period_s], switching between +-amplitude."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, 1.0 / sample_rate_hz)
    signal = np.zeros_like(t)
    idx = 0
    level = amplitude
    while idx < len(t):
        hold_s = rng.uniform(min_period_s, max_period_s)
        hold_n = max(1, int(round(hold_s * sample_rate_hz)))
        signal[idx: idx + hold_n] = level
        level = -level
        idx += hold_n
    return t, signal


def make_trial_signal(signal_type: str, duration_s: float, sample_rate_hz: float,
                       amplitude: float, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Scalar excitation signal e(t) for one trial, per docs/TDD.md section 3.

    'prbs_fast' is 'prbs' with a much shorter maximum hold (0.5s vs 5s).
    A long-hold PRBS applies near-constant torque for seconds at a time,
    which integrates into large attitude excursions (~35 deg even at 2%
    amplitude) - fine for the full-envelope SINDy dataset, but it makes a
    near-hover regime unreachable by amplitude reduction alone. Shortening
    the hold bounds the integration instead. See docs/TDD.md section 5.
    """
    if signal_type == "chirp":
        return chirp(duration_s, sample_rate_hz, f0_hz=0.1, f1_hz=15.0, amplitude=amplitude)
    if signal_type == "prbs":
        return prbs(duration_s, sample_rate_hz, min_period_s=0.05, max_period_s=5.0,
                    amplitude=amplitude, seed=seed)
    if signal_type == "prbs_fast":
        return prbs(duration_s, sample_rate_hz, min_period_s=0.05, max_period_s=0.5,
                    amplitude=amplitude, seed=seed)
    raise ValueError(f"unknown signal_type: {signal_type!r}")


def make_rotor_offsets(axis: str, signal_type: str, duration_s: float, sample_rate_hz: float,
                        amplitude_scale: float = DEFAULT_AMPLITUDE_SCALE,
                        seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-rotor differential thrust offsets (N) for a given axis/signal_type trial.

    axis: 'roll', 'pitch', 'yaw', or 'combined' (independent signal per axis, summed).
    amplitude_scale: multiplier applied on top of each axis's calibrated
    base amplitude (AXIS_BASE_AMPLITUDE_N) - use this to vary trial
    intensity, not an absolute Newton value, so roll/pitch/yaw stay
    correctly relatively-calibrated regardless of scale.
    """
    if axis == "combined":
        n = round(duration_s * sample_rate_hz)
        offsets = np.zeros((n, 4))
        t = None
        for i, ax in enumerate(("roll", "pitch", "yaw")):
            axis_seed = None if seed is None else seed + i + 1
            amp = AXIS_BASE_AMPLITUDE_N[ax] * amplitude_scale
            t, e = make_trial_signal(signal_type, duration_s, sample_rate_hz, amplitude=amp, seed=axis_seed)
            offsets += np.outer(e, AXIS_MIX[ax])
    else:
        amp = AXIS_BASE_AMPLITUDE_N[axis] * amplitude_scale
        t, e = make_trial_signal(signal_type, duration_s, sample_rate_hz, amplitude=amp, seed=seed)
        offsets = np.outer(e, AXIS_MIX[axis])
    return t, offsets
