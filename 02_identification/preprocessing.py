"""Stage B preprocessing: quaternion -> constraint-free local rotation
vector, Savitzky-Golay smoothing, and derivative estimation.

Implements the quaternion-constraint strategy chosen in docs/TDD.md
section 4 (option b): attitude is expressed as a body-frame rotation
vector theta relative to a periodically-reset reference attitude, so the
identification target lives in an unconstrained R^6 (theta, omega) and no
unit-norm constraint can be silently violated by the fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# Reset the reference attitude before the rotation-vector map approaches
# its singularity at 180 deg. 150 deg leaves clear margin.
RESET_MARGIN_DEG = 150.0
# Savitzky-Golay parameters. At 500 Hz a 31-sample window spans 62 ms -
# short relative to the ~0.4-0.8s rotational time constant (so real
# dynamics are not smoothed away) but long enough to suppress
# differentiation noise. Recorded per docs/DATA_MODEL.md section 2.
SG_WINDOW = 31
SG_POLYORDER = 3
# Samples to drop at each end of a reset-delimited segment: Savitzky-Golay
# fits a one-sided polynomial near edges, which is markedly less accurate
# than its centred estimate. Dropping half a window removes that bias
# rather than letting it enter the fit as if it were signal.
EDGE_TRIM = SG_WINDOW // 2


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, scalar-first [w,x,y,z] (docs/TDD.md section 1)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_log_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Quaternion -> rotation vector (axis * angle, radians).

    The sign ambiguity of the double cover (q and -q are the same
    rotation) is resolved toward w >= 0, which selects the shorter of the
    two equivalent rotations - without this the rotation vector can jump
    by 2*pi between adjacent samples for a smoothly rotating body.
    """
    if q[0] < 0:
        q = -q
    v = q[1:]
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(v_norm, q[0])
    return (angle / v_norm) * v


def to_local_rotvec(quats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quaternion trajectory -> (theta, reset_flags) per docs/TDD.md section 4.

    theta[i] is the rotation from the current reference attitude to
    quats[i]. When ||theta|| exceeds RESET_MARGIN_DEG the reference is
    re-anchored to the current attitude, so theta stays well away from
    the rotation-vector singularity at 180 deg.
    """
    n = len(quats)
    theta = np.zeros((n, 3))
    resets = np.zeros(n, dtype=bool)
    q_ref = quats[0].copy()
    margin_rad = np.radians(RESET_MARGIN_DEG)

    for i in range(n):
        rel = quat_multiply(quat_conjugate(q_ref), quats[i])
        tv = quat_log_to_rotvec(rel)
        if np.linalg.norm(tv) > margin_rad:
            q_ref = quats[i].copy()
            tv = np.zeros(3)
            resets[i] = True
        theta[i] = tv
    return theta, resets


def segment_bounds(resets: np.ndarray, min_length: int) -> list[tuple[int, int]]:
    """Contiguous [start, end) ranges between reference-attitude resets.

    Derivatives must never be taken across a reset: theta is discontinuous
    there by construction (it jumps to zero), so a filter spanning the
    boundary would report an enormous fictitious theta_dot that is a
    coordinate artifact, not motion.
    """
    starts = [0] + list(np.flatnonzero(resets))
    ends = list(np.flatnonzero(resets)) + [len(resets)]
    return [(s, e) for s, e in zip(starts, ends) if e - s >= min_length]


def preprocess_trial(df: pd.DataFrame) -> pd.DataFrame:
    """Raw trial log -> smoothed (theta, omega) and their derivatives.

    Returns only samples with trustworthy derivative estimates: segments
    shorter than the Savitzky-Golay window are dropped entirely, and
    EDGE_TRIM samples are removed from each segment end.
    """
    quats = df[["qw", "qx", "qy", "qz"]].to_numpy()
    omega = df[["wx", "wy", "wz"]].to_numpy()
    u = df[["u1", "u2", "u3", "u4"]].to_numpy()
    t = df["t"].to_numpy()
    dt = float(np.median(np.diff(t)))

    theta, resets = to_local_rotvec(quats)

    pieces = []
    for seg_index, (start, end) in enumerate(
            segment_bounds(resets, min_length=SG_WINDOW + 2 * EDGE_TRIM)):
        sl = slice(start, end)
        th_s = savgol_filter(theta[sl], SG_WINDOW, SG_POLYORDER, axis=0)
        om_s = savgol_filter(omega[sl], SG_WINDOW, SG_POLYORDER, axis=0)
        th_d = savgol_filter(theta[sl], SG_WINDOW, SG_POLYORDER, deriv=1, delta=dt, axis=0)
        om_d = savgol_filter(omega[sl], SG_WINDOW, SG_POLYORDER, deriv=1, delta=dt, axis=0)

        keep = slice(EDGE_TRIM, (end - start) - EDGE_TRIM)
        piece = pd.DataFrame({
            # segment_id is load-bearing, not bookkeeping: theta is
            # discontinuous across a reference-attitude reset (it jumps
            # from ~150 deg back to 0). Any consecutive-sample pair
            # spanning two segments describes a coordinate jump of ~2.6
            # rad where real motion is ~0.002 rad per step - squared,
            # a single such fake pair outweighs a million genuine ones
            # and will dominate any sum-of-squares metric computed over
            # the trial. Downstream code MUST group by segment_id before
            # forming (k, k+1) pairs.
            "segment_id": seg_index,
            "t": t[sl][keep],
            "theta_x": th_s[keep, 0], "theta_y": th_s[keep, 1], "theta_z": th_s[keep, 2],
            "omega_x": om_s[keep, 0], "omega_y": om_s[keep, 1], "omega_z": om_s[keep, 2],
            "theta_dot_x": th_d[keep, 0], "theta_dot_y": th_d[keep, 1], "theta_dot_z": th_d[keep, 2],
            "omega_dot_x": om_d[keep, 0], "omega_dot_y": om_d[keep, 1], "omega_dot_z": om_d[keep, 2],
            "u1": u[sl][keep, 0], "u2": u[sl][keep, 1],
            "u3": u[sl][keep, 2], "u4": u[sl][keep, 3],
        })
        pieces.append(piece)

    if not pieces:
        return pd.DataFrame()

    out = pd.concat(pieces, ignore_index=True)
    out["trial_id"] = df["trial_id"].iloc[0]
    out["sg_window"] = SG_WINDOW
    out["sg_polyorder"] = SG_POLYORDER
    out.attrs["dt"] = dt
    out.attrs["n_resets"] = int(resets.sum())
    return out


STATE_COLS = ["theta_x", "theta_y", "theta_z", "omega_x", "omega_y", "omega_z"]
DERIV_COLS = ["theta_dot_x", "theta_dot_y", "theta_dot_z",
              "omega_dot_x", "omega_dot_y", "omega_dot_z"]
CONTROL_COLS = ["u1", "u2", "u3", "u4"]
