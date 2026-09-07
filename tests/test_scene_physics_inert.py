"""Guard: scene.xml's visual additions must NOT change the plant.

Stage E's city environment (the `<include file="city.xml"/>` objects, the
map-textured floor, lights, and cameras in 01_simulation/models/scene.xml)
is physics-inert by construction: every city geom is contype=0
conaffinity=0, zero-mass, with no joints or actuators (see city.py and
make_city.py). Identification data comes from quad.xml (never scene.xml) so
this must stay true permanently - a city geom accidentally given mass or
contact, or a stray joint/actuator/DOF, would silently alter control or
identification behavior while looking like a harmless visual tweak.

This test drives quad.xml and scene.xml open-loop with an identical control
sequence and asserts the full state trajectory (qpos/qvel) matches
bit-for-bit. It is the standing, automated form of the project rule
"reconfirm physics-identical via the test suite after any scene change"
(CLAUDE.md Stage E history).
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "01_simulation" / "models"

STEPS = 2000  # 4 s at the 500 Hz control rate


def _load(name: str):
    m = mujoco.MjModel.from_xml_path(str(MODELS / name))
    return m, mujoco.MjData(m)


def test_scene_has_no_extra_dofs_or_actuators():
    m_q, _ = _load("quad.xml")
    m_s, _ = _load("scene.xml")
    assert (m_s.nq, m_s.nv, m_s.nu) == (m_q.nq, m_q.nv, m_q.nu)
    assert (m_q.njnt, m_q.nbody, m_q.nactuator) == (m_s.njnt, m_s.nbody, m_s.nactuator)


def test_scene_vs_quad_bit_for_bit_open_loop():
    m_q, d_q = _load("quad.xml")
    m_s, d_s = _load("scene.xml")
    assert m_q.opt.timestep == m_s.opt.timestep

    rng = np.random.default_rng(42)
    for i in range(STEPS):
        u = rng.uniform(5.0, 12.0, size=m_q.nu)
        d_q.ctrl[:] = u
        d_s.ctrl[:] = u
        mujoco.mj_step(m_q, d_q)
        mujoco.mj_step(m_s, d_s)
        if not (np.array_equal(d_q.qpos, d_s.qpos) and np.array_equal(d_q.qvel, d_s.qvel)):
            pytest.fail(
                f"scene.xml diverged from quad.xml at step {i}: "
                f"|dqpos|max={np.max(np.abs(d_q.qpos - d_s.qpos)):.3e}, "
                f"|dqvel|max={np.max(np.abs(d_q.qvel - d_s.qvel)):.3e}. "
                "A scene.xml edit added physics (contact/mass/joint/actuator); "
                "city additions must stay visual-only."
            )