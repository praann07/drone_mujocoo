"""Mixer / inverse-mixer between (thrust, torque) and individual rotor
commands. Matches the rotor geometry in 01_simulation/models/quad.xml
and the mixer derivation in docs/TDD.md section 2 exactly - mixer_matrix()
is checked against that XML's actual rotor site positions and spin signs
in tests/test_mixer.py, not just asserted here.
"""
from __future__ import annotations

import numpy as np

ARM = 0.106066   # l/sqrt(2), m - docs/TDD.md section 2
K_M = 0.02       # rotor reaction-torque coefficient, m
U_MAX = 6.0      # N per rotor
HOVER_THRUST_PER_ROTOR = 1.0 * 9.81 / 4.0


def mixer_matrix() -> np.ndarray:
    """Rows: [T, tau_x, tau_y, tau_z]. Columns: [u1, u2, u3, u4].
    Rotor positions/spin signs per docs/TDD.md section 2 table."""
    return np.array([
        [1.0,    1.0,    1.0,    1.0],
        [ARM,   -ARM,   -ARM,    ARM],
        [-ARM,   ARM,   -ARM,    ARM],
        [K_M,    K_M,   -K_M,   -K_M],
    ])


_M = mixer_matrix()
_M_INV = np.linalg.inv(_M)


def inverse_mix(T: float, tau: np.ndarray, clip: bool = True) -> tuple[np.ndarray, bool]:
    """(thrust, torque) -> 4 rotor commands. Returns (u, saturated) - the
    caller must check `saturated` rather than silently accepting a
    clipped, physically-unachievable command as if it were exact."""
    u = _M_INV @ np.concatenate([[T], tau])
    saturated = bool(np.any(u < 0.0) or np.any(u > U_MAX))
    if clip:
        u = np.clip(u, 0.0, U_MAX)
    return u, saturated
