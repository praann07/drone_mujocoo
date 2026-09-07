"""Step-response metrics from a closed-loop rollout log: settling time,
overshoot, steady-state error - computed on the scalar attitude-error
signal (degrees) so a single set of formulas covers roll/pitch/yaw and
combined-axis setpoints alike.
"""
from __future__ import annotations

import numpy as np

from controller import QuaternionController


def attitude_error_series(controller: QuaternionController, log: dict) -> np.ndarray:
    return np.array([controller.attitude_error_deg(log["q"][k], log["q_desired"][k])
                      for k in range(len(log["t"]))])


def step_response_metrics(t: np.ndarray, error_deg: np.ndarray,
                          settle_band_deg: float = 1.0) -> dict:
    """error_deg: attitude error to the (assumed constant, post-step)
    setpoint. settle_band_deg: the '2%-style' band used for settling
    time - 1 degree is a fixed absolute band, appropriate since setpoints
    here are O(10-20 deg), not a percentage of a varying step size.
    """
    initial_error = error_deg[0]
    overshoot_deg = max(0.0, initial_error - error_deg.min()) if initial_error > 0 else 0.0
    # More standard framing: overshoot is how far error goes NEGATIVE
    # past zero, i.e. how far past the setpoint the response swings.
    # error_deg is a magnitude (always >=0 from attitude_error_deg), so
    # "past the target" shows as error_deg dipping toward 0 then a
    # secondary rise would be needed to detect true overshoot in a scalar
    # magnitude signal - not reliably observable from magnitude alone for
    # a 3D rotation. Report the minimum reached instead, which is the
    # closest approach; a distinct secondary local max after that AND
    # above settle_band_deg is flagged as observed ringing.
    within_band = error_deg <= settle_band_deg
    settle_idx = None
    for i in range(len(within_band)):
        if within_band[i:].all():
            settle_idx = i
            break
    settling_time_s = float(t[settle_idx]) if settle_idx is not None else None

    steady_state_error_deg = float(np.mean(error_deg[-int(0.1 * len(error_deg)):]))

    # Ringing: does error rise again by >settle_band_deg after first
    # reaching its minimum? Only meaningful once a minimum is reached.
    min_idx = int(np.argmin(error_deg))
    post_min = error_deg[min_idx:]
    ringing_deg = float(post_min.max() - post_min[0]) if len(post_min) > 1 else 0.0

    return {
        "settling_time_s": settling_time_s,
        "min_error_deg": float(error_deg.min()),
        "ringing_deg": ringing_deg,
        "steady_state_error_deg": steady_state_error_deg,
        "settled": settle_idx is not None,
    }
