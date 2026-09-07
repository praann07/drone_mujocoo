"""Open-loop rollout integration for Stage C validation.

Given a fitted model and a held-out segment's recorded control sequence,
integrate the model FORWARD from the segment's initial state - never
re-anchoring to ground truth mid-rollout - and compare against the
segment's own (Savitzky-Golay smoothed) trajectory. This is what actually
tests whether the identified model behaves like the system over time,
as opposed to one-step-ahead error which only tests local slope accuracy.

SINDy is integrated with RK4 (matching the RK4 integrator MuJoCo itself
uses for ground truth in 01_simulation/models/quad.xml, so
rollout error reflects model mismatch rather than a cruder discretization
scheme than the ground truth was generated with). DMDc is already a
discrete-time map, so its rollout is direct repeated application.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATE_COLS = ["theta_x", "theta_y", "theta_z", "omega_x", "omega_y", "omega_z"]
CONTROL_COLS = ["u1", "u2", "u3", "u4"]

# A rollout is called "diverged" once combined state error crosses this
# bound. theta error contributes in radians, omega error in rad/s -
# summed as an unweighted Euclidean norm since both are O(1) quantities
# in this regime (theta bounded by the ~20-180deg excursions seen in
# Stage A, omega by the few-rad/s excitation levels in TDD.md section 3).
DIVERGENCE_BOUND = 0.5


def sindy_rk4_rollout(model, theta0_omega0: np.ndarray, controls: np.ndarray, dt: float) -> np.ndarray:
    """Integrate the SINDy ODE with RK4, holding control constant per step
    (zero-order hold - matches how Stage A applied ctrl in sim_driver.py)."""
    n = len(controls)
    traj = np.zeros((n, 6))
    traj[0] = theta0_omega0
    for k in range(n - 1):
        x, u = traj[k], controls[k]
        k1 = model.predict_derivative(x[None, :], u[None, :])[0]
        k2 = model.predict_derivative((x + 0.5 * dt * k1)[None, :], u[None, :])[0]
        k3 = model.predict_derivative((x + 0.5 * dt * k2)[None, :], u[None, :])[0]
        k4 = model.predict_derivative((x + dt * k3)[None, :], u[None, :])[0]
        traj[k + 1] = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return traj


def dmdc_rollout(model, theta0_omega0: np.ndarray, controls: np.ndarray) -> np.ndarray:
    n = len(controls)
    traj = np.zeros((n, 6))
    traj[0] = theta0_omega0
    for k in range(n - 1):
        traj[k + 1] = model.step(traj[k][None, :], controls[k][None, :])[0]
    return traj


def rollout_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-timestep combined state error norm (see DIVERGENCE_BOUND)."""
    return np.linalg.norm(pred - true, axis=1)


def divergence_time(error: np.ndarray, t: np.ndarray, bound: float = DIVERGENCE_BOUND) -> float | None:
    """First timestamp where error crosses bound, or None if it never does."""
    crossed = np.flatnonzero(error > bound)
    return float(t[crossed[0]]) if len(crossed) else None


def qualifying_segments(frames: dict[str, pd.DataFrame], min_length_s: float) -> list[tuple[str, pd.DataFrame]]:
    """(trial_id, segment_df) pairs long enough for the requested horizon."""
    out = []
    for tid, df in frames.items():
        for _, seg in df.groupby("segment_id", sort=True):
            duration = seg.t.iloc[-1] - seg.t.iloc[0]
            if duration >= min_length_s:
                out.append((tid, seg.reset_index(drop=True)))
    return out


def run_rollout(model, kind: str, seg: pd.DataFrame, horizon_s: float, dt: float) -> dict:
    """Roll one segment out to horizon_s (or its full length if shorter)."""
    n_steps = min(len(seg), int(round(horizon_s / dt)) + 1)
    state = seg[STATE_COLS].to_numpy()[:n_steps]
    control = seg[CONTROL_COLS].to_numpy()[:n_steps]
    t = seg["t"].to_numpy()[:n_steps] - seg["t"].to_numpy()[0]

    if kind == "sindy":
        pred = sindy_rk4_rollout(model, state[0], control, dt)
    elif kind == "dmdc":
        pred = dmdc_rollout(model, state[0], control)
    else:
        raise ValueError(kind)

    err = rollout_error(pred, state)
    return {
        "t": t, "true": state, "pred": pred, "error": err,
        "divergence_t": divergence_time(err, t),
        "final_error": float(err[-1]),
        "max_error": float(err.max()),
    }
