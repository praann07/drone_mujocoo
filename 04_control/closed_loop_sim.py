"""Closed-loop simulation: QuaternionController driving either the
SINDy-identified model or ground-truth MuJoCo physics.

Attitude is propagated with EXACT quaternion kinematics
(dq/dt = 0.5 * q (x) [0, omega]), not SINDy's approximate local-coordinate
theta_dot - that approximation was a Stage B/C identification-coordinate
convenience, and 02_identification.run_stage_b confirmed (fitted
model coefficients) that the identified omega_dot has ZERO dependence on
theta, so SINDy is used only for what it actually models: omega_dot(omega,
u). This sidesteps the periodic reference-attitude reset machinery
entirely for closed-loop control, where step responses run at most a few
seconds and exact kinematics has no discontinuities to reset around.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_simulation"))
from sim_driver import C_ROT, C_TRANS, MODEL_PATH  # noqa: E402

DT = 0.002  # s, docs/TDD.md section 1


def quat_kinematics_rk4(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """One RK4 step of dq/dt = 0.5 * q (x) [0, omega], omega constant
    over the step (matches how MuJoCo/Stage A held control constant)."""
    def dq(qq):
        w, x, y, z = qq
        ox, oy, oz = omega
        return 0.5 * np.array([
            -x * ox - y * oy - z * oz,
             w * ox + y * oz - z * oy,
             w * oy - x * oz + z * ox,
             w * oz + x * oy - y * ox,
        ])
    k1 = dq(q)
    k2 = dq(q + 0.5 * dt * k1)
    k3 = dq(q + 0.5 * dt * k2)
    k4 = dq(q + dt * k3)
    q_next = q + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return q_next / np.linalg.norm(q_next)


def rollout_on_sindy(controller, sindy_model, q0: np.ndarray, omega0: np.ndarray,
                      q_desired_fn, thrust_desired: float, duration_s: float, dt: float = DT):
    """Closed loop: controller output -> mixer -> SINDy omega_dot(omega,u)
    -> exact quaternion kinematics. q_desired_fn(t) allows a time-varying
    setpoint (needed once the outer loop is wired in Stage D part 2)."""
    n = int(round(duration_s / dt))
    log = {"t": np.zeros(n), "q": np.zeros((n, 4)), "omega": np.zeros((n, 3)),
           "q_desired": np.zeros((n, 4)), "saturated": np.zeros(n, dtype=bool)}
    q, omega = q0.copy(), omega0.copy()
    theta_dummy = np.zeros(3)  # unused: omega_dot has zero theta-dependence (verified)

    for k in range(n):
        t = k * dt
        q_d = q_desired_fn(t)
        u, sat = controller.compute(q, omega, q_d, np.zeros(3), thrust_desired)

        log["t"][k] = t
        log["q"][k] = q
        log["omega"][k] = omega
        log["q_desired"][k] = q_d
        log["saturated"][k] = sat

        state = np.concatenate([theta_dummy, omega])[None, :]
        deriv = sindy_model.predict_derivative(state, u[None, :])[0]
        omega_dot = deriv[3:6]
        omega = omega + dt * omega_dot
        q = quat_kinematics_rk4(q, omega, dt)

    return log


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v (world<-body per q) - Hamilton scalar-first."""
    qv = np.array([0.0, *v])

    def mul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    return mul(mul(q, qv), q_conj)[1:]


def rollout_cascade_on_sindy(inner, outer, sindy_model, mass, c_trans,
                              p0, v0, q0, omega0, traj, duration_s, dt=DT):
    """Full cascade: outer loop (standard translational physics, computed
    here) -> inner loop -> SINDy omega_dot(omega,u) -> exact quaternion
    kinematics. Translation always uses standard physics per docs/TDD.md
    section 7b - only rotation differs between this and the MuJoCo cascade."""
    n = int(round(duration_s / dt))
    log = {"t": np.zeros(n), "p": np.zeros((n, 3)), "v": np.zeros((n, 3)),
           "q": np.zeros((n, 4)), "omega": np.zeros((n, 3)),
           "p_desired": np.zeros((n, 3)), "q_desired": np.zeros((n, 4))}
    p, v, q, omega = p0.copy(), v0.copy(), q0.copy(), omega0.copy()
    theta_dummy = np.zeros(3)

    for k in range(n):
        t = k * dt
        p_d, v_d, a_ff = traj.position(t), traj.velocity(t), traj.acceleration(t)
        q_d, thrust_d = outer.compute(p, v, p_d, v_d, a_ff)
        u, _ = inner.compute(q, omega, q_d, np.zeros(3), thrust_d)

        log["t"][k] = t
        log["p"][k], log["v"][k], log["q"][k], log["omega"][k] = p, v, q, omega
        log["p_desired"][k], log["q_desired"][k] = p_d, q_d

        state = np.concatenate([theta_dummy, omega])[None, :]
        omega_dot = sindy_model.predict_derivative(state, u[None, :])[0][3:6]
        omega = omega + dt * omega_dot
        q = quat_kinematics_rk4(q, omega, dt)

        actual_thrust = float(np.sum(u))
        body_z = quat_rotate_vector(q, np.array([0.0, 0.0, 1.0]))
        v_dot = (actual_thrust / mass) * body_z - np.array([0.0, 0.0, 9.81]) - (c_trans / mass) * v
        v = v + dt * v_dot
        p = p + dt * v

    return log


def rollout_cascade_on_mujoco(inner, outer, p0, v0, q0, omega0, traj, duration_s, dt=DT):
    """Full cascade driving ACTUAL ground-truth MuJoCo physics."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = p0
    data.qpos[3:7] = q0
    data.qvel[0:3] = v0
    data.qvel[3:6] = omega0
    mujoco.mj_forward(model, data)

    n = int(round(duration_s / dt))
    log = {"t": np.zeros(n), "p": np.zeros((n, 3)), "v": np.zeros((n, 3)),
           "q": np.zeros((n, 4)), "omega": np.zeros((n, 3)),
           "p_desired": np.zeros((n, 3)), "q_desired": np.zeros((n, 4))}

    for k in range(n):
        t = k * dt
        p = data.qpos[0:3].copy()
        v = data.qvel[0:3].copy()
        q = data.qpos[3:7].copy()
        omega = data.qvel[3:6].copy()
        p_d, v_d, a_ff = traj.position(t), traj.velocity(t), traj.acceleration(t)
        q_d, thrust_d = outer.compute(p, v, p_d, v_d, a_ff)
        u, _ = inner.compute(q, omega, q_d, np.zeros(3), thrust_d)

        log["t"][k] = t
        log["p"][k], log["v"][k], log["q"][k], log["omega"][k] = p, v, q, omega
        log["p_desired"][k], log["q_desired"][k] = p_d, q_d

        data.ctrl[:] = u
        data.qfrc_applied[0:3] = -C_TRANS * data.qvel[0:3]
        data.qfrc_applied[3:6] = -C_ROT * data.qvel[3:6]
        mujoco.mj_step(model, data)

    return log


def rollout_on_mujoco(controller, q0: np.ndarray, omega0: np.ndarray,
                       q_desired_fn, thrust_desired: float, duration_s: float, dt: float = DT):
    """Same closed loop, driving ACTUAL ground-truth MuJoCo physics
    (including the C_ROT/C_TRANS drag applied in Stage A) - the sim-to-sim
    gap check target."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[3:7] = q0
    data.qvel[3:6] = omega0
    mujoco.mj_forward(model, data)

    n = int(round(duration_s / dt))
    log = {"t": np.zeros(n), "q": np.zeros((n, 4)), "omega": np.zeros((n, 3)),
           "q_desired": np.zeros((n, 4)), "saturated": np.zeros(n, dtype=bool)}

    for k in range(n):
        t = k * dt
        q = data.qpos[3:7].copy()
        omega = data.qvel[3:6].copy()
        q_d = q_desired_fn(t)
        u, sat = controller.compute(q, omega, q_d, np.zeros(3), thrust_desired)

        log["t"][k] = t
        log["q"][k] = q
        log["omega"][k] = omega
        log["q_desired"][k] = q_d
        log["saturated"][k] = sat

        data.ctrl[:] = u
        data.qfrc_applied[0:3] = -C_TRANS * data.qvel[0:3]
        data.qfrc_applied[3:6] = -C_ROT * data.qvel[3:6]
        mujoco.mj_step(model, data)

    return log
