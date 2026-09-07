"""Point-to-point trajectory planner: straight line, trapezoidal velocity
profile (docs/TDD.md section 7b). Never commands a position step directly
- a raw step would demand instantaneous velocity, which is physically
impossible and would just saturate the outer loop; the trapezoidal
profile instead ramps velocity smoothly so the attitude setpoints it
drives stay within what the inner loop was actually validated against.
"""
from __future__ import annotations

import numpy as np

# Stage D outer-loop parameters, locked here (docs/TDD.md section 9 log).
MAX_VELOCITY_MPS = 1.0
MAX_ACCEL_MPS2 = 1.0
WAYPOINT_OFFSET_M = 1.5   # left/right/forward/back move this far
POSITION_TOLERANCE_M = 0.15


def _trapezoidal_1d(distance: float, v_max: float, a_max: float):
    """Returns (t_total, pos_fn, vel_fn, accel_fn) for 1D motion 0->distance."""
    if distance < 1e-9:
        return 0.0, (lambda t: 0.0), (lambda t: 0.0), (lambda t: 0.0)

    t_accel = v_max / a_max
    d_accel = 0.5 * a_max * t_accel ** 2

    if 2 * d_accel >= distance:
        # Triangular profile: never reaches v_max.
        t_accel = np.sqrt(distance / a_max)
        v_peak = a_max * t_accel
        t_total = 2 * t_accel
        t_cruise_end = t_accel
    else:
        d_cruise = distance - 2 * d_accel
        t_cruise = d_cruise / v_max
        v_peak = v_max
        t_total = 2 * t_accel + t_cruise
        t_cruise_end = t_accel + t_cruise

    def pos(t):
        if t <= 0:
            return 0.0
        if t < t_accel:
            return 0.5 * a_max * t ** 2
        if t < t_cruise_end:
            return 0.5 * a_max * t_accel ** 2 + v_peak * (t - t_accel)
        if t < t_total:
            td = t_total - t
            return distance - 0.5 * a_max * td ** 2
        return distance

    def vel(t):
        if t <= 0 or t >= t_total:
            return 0.0
        if t < t_accel:
            return a_max * t
        if t < t_cruise_end:
            return v_peak
        return a_max * (t_total - t)

    def accel(t):
        if t <= 0 or t >= t_total:
            return 0.0
        if t < t_accel:
            return a_max
        if t < t_cruise_end:
            return 0.0
        return -a_max

    return t_total, pos, vel, accel


class PointToPointTrajectory:
    """Straight-line trapezoidal-velocity trajectory from p0 to p1 (world
    frame, per docs/TDD.md section 7b - waypoint directions are WORLD
    frame, not body-relative, so 'forward' means +world-X regardless of
    current yaw - yaw is never commanded by any of the 6 voice commands,
    so this avoids needing to track it for direction semantics."""

    def __init__(self, p0: np.ndarray, p1: np.ndarray,
                 v_max: float = MAX_VELOCITY_MPS, a_max: float = MAX_ACCEL_MPS2):
        self.p0 = p0
        delta = p1 - p0
        self.distance = float(np.linalg.norm(delta))
        self.direction = delta / self.distance if self.distance > 1e-9 else np.zeros(3)
        self.t_total, self._pos, self._vel, self._accel = _trapezoidal_1d(self.distance, v_max, a_max)

    def position(self, t: float) -> np.ndarray:
        return self.p0 + self.direction * self._pos(min(max(t, 0.0), self.t_total))

    def velocity(self, t: float) -> np.ndarray:
        return self.direction * self._vel(min(max(t, 0.0), self.t_total))

    def acceleration(self, t: float) -> np.ndarray:
        return self.direction * self._accel(min(max(t, 0.0), self.t_total))
