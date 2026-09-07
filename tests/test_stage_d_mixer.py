"""inverse_mix must be the exact inverse of the forward mixer already
verified against quad.xml in tests/test_mixer.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "04_control"))
from mixer import U_MAX, inverse_mix, mixer_matrix  # noqa: E402


@pytest.mark.parametrize("T,tau", [
    (9.81, np.array([0.0, 0.0, 0.0])),
    (9.81, np.array([0.05, -0.03, 0.01])),
    (6.0, np.array([-0.02, 0.02, -0.005])),
])
def test_round_trip(T, tau):
    u, saturated = inverse_mix(T, tau, clip=False)
    assert not saturated
    T_back, *tau_back = mixer_matrix() @ u
    assert T_back == pytest.approx(T, rel=1e-9)
    assert np.allclose(tau_back, tau, atol=1e-9)


def test_saturation_flagged_not_hidden():
    u, saturated = inverse_mix(100.0, np.array([0.0, 0.0, 0.0]))
    assert saturated
    assert np.all(u <= U_MAX)
