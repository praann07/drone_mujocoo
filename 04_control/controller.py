"""Singularity-free quaternion attitude controller, Fresk-style P^2 law
(docs/TDD.md section 7): proportional feedback on the quaternion error
vector part plus proportional feedback on angular-velocity error.

    tau = -Kq * sign(qe_w) * qe_vec  -  Komega * (omega - omega_desired)

The sign(qe_w) term resolves the quaternion double-cover ambiguity (q and
-q represent the same rotation) so the controller always takes the SHORT
way around - without it, a large-angle error could command a rotation the
long way past 180 degrees. This is the same ambiguity-resolution already
used for the rotation-vector coordinate in 02_identification/
preprocessing.py::quat_log_to_rotvec, so the convention is consistent
across the project.

Gains are DERIVED from the Stage C-selected identified model (SINDy),
not hand-tuned constants - see gains_from_identified_model(). This is
what makes this "a controller designed against the identified model"
rather than a generic PD controller that happens to run on this system.
"""
from __future__ import annotations

import numpy as np

from mixer import HOVER_THRUST_PER_ROTOR, inverse_mix

# Target closed-loop second-order response, per axis. These are the
# ONLY hand-picked numbers in this file; everything else (Kp, Kd) is
# derived from them plus the identified model's inertia/drag.
SETTLING_TIME_S = 1.5   # 2% settling time
DAMPING_RATIO = 0.7     # underdamped, small overshoot, no ringing


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def gains_from_identified_model(A_sindy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis (Kq, Komega) from the SINDy hover-linearization.

    A_sindy is the 6x6 Jacobian from
    03_validation.run_validation.linearize_sindy_at_hover: its diagonal
    blocks give, per axis i, omega_dot_i = ... - (c_rot/I_i)*omega_i, so
    the identified drag-over-inertia ratio is read directly off
    A_sindy[3+i, 3+i] - no separate access to the simulator's ground-truth
    I/c_rot is used here, keeping this an honestly *identified-model-based*
    design rather than one that secretly consults ground truth.

    Standard 2nd-order design: for i_dot_dot = -c/I * i_dot + tau/I with
    tau = -Kp*e - Kd*e_dot, closed loop is
        e_ddot + ((Kd+c)/I) e_dot + (Kp/I) e = 0
    so Kp/I = wn^2, (Kd+c)/I = 2*zeta*wn. Converting to the quaternion
    error convention (qe_vec ~ e/2 for small angles): Kq = 2*Kp.
    """
    wn = 4.0 / (DAMPING_RATIO * SETTLING_TIME_S)   # 2% settling-time rule of thumb
    Kq = np.zeros(3)
    Komega = np.zeros(3)
    for i in range(3):
        drag_over_I = -A_sindy[3 + i, 3 + i]   # positive: identified c_rot/I_i
        Kp_over_I = wn ** 2
        total_damping_over_I = 2 * DAMPING_RATIO * wn
        Kd_over_I = total_damping_over_I - drag_over_I
        # Kp/I and Kd/I are what the closed-loop actually sees (tau/I on
        # the RHS) - no separate inertia value needed since the SINDy
        # model already expresses omega_dot directly, not I*omega_dot.
        Kq[i] = 2 * Kp_over_I
        Komega[i] = max(Kd_over_I, 0.0)  # never negative damping
    return Kq, Komega


class QuaternionController:
    """Inner-loop attitude controller. Outputs 4 rotor thrust commands."""

    def __init__(self, Kq: np.ndarray, Komega: np.ndarray):
        self.Kq = Kq
        self.Komega = Komega

    def compute(self, q: np.ndarray, omega: np.ndarray,
                q_desired: np.ndarray, omega_desired: np.ndarray,
                thrust_desired: float) -> tuple[np.ndarray, bool]:
        """One control step. q/q_desired: Hamilton scalar-first. omega in
        body frame (matches docs/TDD.md section 1 throughout).
        Returns (rotor_commands, saturated)."""
        qe = quat_multiply(quat_conjugate(q_desired), q)
        if qe[0] < 0:
            qe = -qe  # short-way-around, see module docstring
        qe_vec = qe[1:]
        omega_e = omega - omega_desired

        tau = -self.Kq * qe_vec - self.Komega * omega_e
        return inverse_mix(thrust_desired, tau)

    def attitude_error_deg(self, q: np.ndarray, q_desired: np.ndarray) -> float:
        qe = quat_multiply(quat_conjugate(q_desired), q)
        w = np.clip(abs(qe[0]), -1.0, 1.0)
        return float(np.degrees(2 * np.arccos(w)))
