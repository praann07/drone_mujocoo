"""Verify the quad.xml rotor sites produce the mixer matrix stated in
docs/TDD.md section 2, and that hover thrust holds the vehicle roughly
level. This is what makes the mixer derivation in TDD.md load-bearing
rather than just a comment.
"""
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

MODEL_PATH = Path(__file__).resolve().parent.parent / "01_simulation" / "models" / "quad.xml"


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def test_mass_and_inertia_match_tdd(model):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quad")
    assert model.body_mass[body_id] == pytest.approx(1.0)
    inertia = model.body_inertia[body_id]
    assert inertia[0] == pytest.approx(0.003375, rel=1e-3)
    assert inertia[1] == pytest.approx(0.003375, rel=1e-3)
    assert inertia[2] == pytest.approx(0.006750, rel=1e-3)


def test_hover_thrust_holds_altitude(model):
    data = mujoco.MjData(model)
    hover_per_rotor = 1.0 * 9.81 / 4.0
    data.ctrl[:] = hover_per_rotor
    z0 = data.qpos[2]
    for _ in range(500):  # 1s at dt=0.002
        mujoco.mj_step(model, data)
    # Should stay near-level and near-altitude under pure hover thrust,
    # not tumble or free-fall - confirms sign/placement correctness.
    assert abs(data.qpos[2] - z0) < 0.05
    qw = data.qpos[3]
    assert qw == pytest.approx(1.0, abs=0.05)  # near-identity quaternion = level


@pytest.mark.parametrize(
    "ctrl,expected_sign",
    [
        # (u1,u2,u3,u4) differential patterns and the expected sign of the
        # resulting angular acceleration on the corresponding axis, per
        # the mixer derivation in TDD.md section 2:
        #   tau_x = (l/sqrt2)( u1 - u2 - u3 + u4)
        #   tau_y = (l/sqrt2)(-u1 + u2 - u3 + u4)
        #   tau_z = k_m * ( u1 + u2 - u3 - u4)
        ([1.0, 0.0, 0.0, 1.0], "x+"),  # u1,u4 up -> tau_x > 0
        ([0.0, 1.0, 1.0, 0.0], "x-"),  # u2,u3 up -> tau_x < 0
        ([0.0, 1.0, 0.0, 1.0], "y+"),  # u2,u4 up -> tau_y > 0
        ([1.0, 0.0, 1.0, 0.0], "y-"),  # u1,u3 up -> tau_y < 0
        ([1.0, 1.0, 0.0, 0.0], "z+"),  # u1,u2 up -> tau_z > 0
        ([0.0, 0.0, 1.0, 1.0], "z-"),  # u3,u4 up -> tau_z < 0
    ],
)
def test_mixer_signs_match_tdd(model, ctrl, expected_sign):
    data = mujoco.MjData(model)
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)  # compute accelerations without integrating
    ang_acc = data.qacc[3:6]  # body-frame angular acceleration (free joint)
    axis = "xyz".index(expected_sign[0])
    sign = 1 if expected_sign[1] == "+" else -1
    assert ang_acc[axis] * sign > 0, f"expected {expected_sign}, got qacc[3:6]={ang_acc}"
