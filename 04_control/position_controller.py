"""Outer position/velocity loop (docs/TDD.md section 7b) - STANDARD
physics, not identified. PD(+feedforward) on position/velocity error
produces a desired acceleration, converted to (q_desired, thrust_desired)
via the standard differential-flatness-style thrust-vector relation:
thrust must point along the desired net specific-force direction
(desired acceleration plus gravity compensation), and its magnitude is
that vector's norm times mass.

Yaw is never commanded (docs/TDD.md section 7b) - the tilt quaternion
that aligns body +Z with the desired thrust direction is used directly,
with no additional yaw correction, which is the natural/minimal-rotation
choice for a yaw-agnostic quadrotor.

Gains target a critically-damped position response assuming a PURE
double-integrator plant (p_ddot = a_desired exactly), with a settling
time deliberately several times slower than the inner loop's (~0.5-0.9s,
docs/CLAUDE.md Stage D notes) - required for the cascade assumption
(docs/TDD.md section 7b) to hold. In the actual cascaded closed loop this
assumption is not exact (the inner loop has its own lag, and translation
has drag), so the real response shows mild ringing (~0.07m, decaying by
~4s) after the trajectory ends rather than the theoretically-predicted
zero overshoot - reported honestly in the Stage D gate output rather than
re-described as "no overshoot" after the fact. Still well inside the
position tolerance and does not affect the gate.
"""
from __future__ import annotations

import numpy as np

MASS_KG = 1.0
GRAVITY = 9.81

# Target 2nd-order position response: critically damped (zeta=1, no
# overshoot - overshooting a commanded waypoint would look wrong in the
# live demo and risks re-triggering inner-loop saturation on the way
# back), settling time deliberately 2-6x the inner loop's ~0.5-0.9s.
POS_SETTLING_TIME_S = 3.0
POS_DAMPING_RATIO = 1.0
_wn = 4.0 / (POS_DAMPING_RATIO * POS_SETTLING_TIME_S)
KP_POS = _wn ** 2          # (m/s^2) per m of position error
KD_POS = 2 * POS_DAMPING_RATIO * _wn   # (m/s^2) per (m/s) of velocity error


def minimal_rotation_z_to(v: np.ndarray) -> np.ndarray:
    """Quaternion (Hamilton, scalar-first) of the minimal rotation taking
    world +Z to the unit vector v. Leaves yaw about v unconstrained -
    the natural choice since yaw is never commanded (see module
    docstring)."""
    z = np.array([0.0, 0.0, 1.0])
    v = v / np.linalg.norm(v)
    axis = np.cross(z, v)
    s = np.linalg.norm(axis)
    c = float(np.dot(z, v))
    if s < 1e-9:
        if c > 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        # 180 degrees - z and v are anti-parallel; any perpendicular axis works.
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = axis / s
    angle = np.arctan2(s, c)
    return np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])


class PositionController:
    def __init__(self, mass: float = MASS_KG, kp: float = KP_POS, kd: float = KD_POS):
        self.mass = mass
        self.kp = kp
        self.kd = kd

    def compute(self, p: np.ndarray, v: np.ndarray,
                p_desired: np.ndarray, v_desired: np.ndarray,
                a_feedforward: np.ndarray) -> tuple[np.ndarray, float]:
        """-> (q_desired, thrust_desired). p, v: world-frame position and
        velocity (matches docs/TDD.md section 1's v convention)."""
        e_p = p_desired - p
        e_v = v_desired - v
        a_desired = self.kp * e_p + self.kd * e_v + a_feedforward

        f_desired = a_desired + np.array([0.0, 0.0, GRAVITY])  # specific force, world frame
        thrust_mag = self.mass * float(np.linalg.norm(f_desired))
        direction = f_desired / np.linalg.norm(f_desired)
        q_desired = minimal_rotation_z_to(direction)
        return q_desired, thrust_mag
