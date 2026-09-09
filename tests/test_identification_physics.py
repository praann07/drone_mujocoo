"""The identified SINDy model must recover the KNOWN rigid-body physics.

This is the strongest available evidence that Stage B identified real
dynamics rather than an opaque curve fit that merely scores well: the
simulator's true parameters are known exactly (docs/TDD.md section 2), so
the recovered coefficients can be checked against them directly.

A model can post a good R^2 while being structurally wrong. It cannot
recover control effectiveness, drag, and the gyroscopic cross-term to
within a few percent by accident.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "02_identification"))

from run_identification import load_processed, load_split, stack  # noqa: E402
from sindy_fit import SINDyModel  # noqa: E402

IXX = IYY = 0.003375
IZZ = 0.006750
C_ROT = 0.008
ARM = 0.106066   # l/sqrt(2)
K_M = 0.02
SELECTED_THRESHOLD = 0.2  # chosen by the parsimony rule in run_identification.py
RTOL = 0.06               # 6% - covers Savitzky-Golay smoothing bias


@pytest.fixture(scope="module")
def model():
    split = load_split()
    fit_ids = split[(split.regime == "full_envelope") & (split.split == "fit")].trial_id
    frames = load_processed(fit_ids)
    if not frames:
        pytest.skip("Stage A data not generated - run run_excitation.py first")
    X, U, D = stack(frames)
    return SINDyModel(threshold=SELECTED_THRESHOLD).fit(X, U, D)


def coef(model, target_index: int, term: str) -> float:
    return dict(model.active_terms(target_index)).get(term, 0.0)


# Target row order: theta_dot_{x,y,z}=0,1,2 ; omega_dot_{x,y,z}=3,4,5
def test_kinematics_theta_dot_equals_omega(model):
    """theta_dot = omega + O(theta x omega): the omega coefficient is exactly 1."""
    for i, axis in enumerate("xyz"):
        assert coef(model, i, f"omega_{axis}") == pytest.approx(1.0, rel=0.02)


def test_control_effectiveness_matches_mixer(model):
    """omega_dot_x gains on u1..u4 must equal +-(l/sqrt2)/Ixx with the
    mixer's sign pattern (+ - - +) from docs/TDD.md section 2."""
    expected = ARM / IXX
    for term, sign in [("u1", 1), ("u2", -1), ("u3", -1), ("u4", 1)]:
        assert coef(model, 3, term) == pytest.approx(sign * expected, rel=RTOL)


def test_yaw_control_effectiveness_matches_reaction_torque(model):
    """omega_dot_z gains come from the rotor reaction torque k_m/Izz,
    with the (+ + - -) spin pattern - a much weaker path than roll/pitch."""
    expected = K_M / IZZ
    for term, sign in [("u1", 1), ("u2", 1), ("u3", -1), ("u4", -1)]:
        assert coef(model, 5, term) == pytest.approx(sign * expected, rel=RTOL)


def test_rotational_drag_recovered(model):
    """The linear -c_rot*omega drag the simulator applies must appear."""
    assert coef(model, 3, "omega_x") == pytest.approx(-C_ROT / IXX, rel=RTOL)
    assert coef(model, 5, "omega_z") == pytest.approx(-C_ROT / IZZ, rel=RTOL)


def test_gyroscopic_cross_term_recovered(model):
    """omega_dot_x carries -(Izz-Iyy)/Ixx * omega_y*omega_z."""
    assert coef(model, 3, "omega_y*omega_z") == pytest.approx(-(IZZ - IYY) / IXX, rel=RTOL)


def test_no_spurious_control_coupling_in_kinematics(model):
    """theta_dot is pure kinematics - thrust cannot enter it directly.
    Any large u term here would mean the fit is laundering dynamics into
    the kinematic rows, which rollout would expose as drift."""
    for i in range(3):
        for u in ("u1", "u2", "u3", "u4"):
            assert abs(coef(model, i, u)) < 1.0
