"""SINDy fit with a physically-motivated library (docs/TDD.md section 4).

The library is built explicitly rather than as a blind degree-N
polynomial. Every included term corresponds to a mechanism actually
present in rigid-body attitude dynamics:

    theta_dot = omega + (1/2) theta x omega + O(||theta||^2)
    omega_dot = I^-1 ( M u  -  omega x (I omega)  -  c_rot omega )

  constant       - allows a bias term; should be driven to zero by
                   sparsity if the fit is unbiased (a useful check).
  theta, omega   - linear kinematic and linear (drag) damping terms. The
                   damping term is REQUIRED: the simulator applies
                   -c_rot*omega drag (docs/TDD.md section 2), so omitting
                   it would guarantee a missing-term bias that no amount
                   of sparsity tuning could fix.
  u              - control effectiveness (I^-1 M u), linear in thrust.
  omega_i*omega_j - the gyroscopic omega x (I omega) term.
  theta_i*omega_j - the bilinear kinematic correction.

Deliberately EXCLUDED, because no mechanism in this system produces
them: theta*theta, u*u, theta*u, omega*u. A blind degree-2 polynomial
over all 10 inputs would carry 66 terms including all of those; this
library carries 26. Fewer spurious candidates means sparsity selection is
doing physics rather than fighting noise.
"""
from __future__ import annotations

import numpy as np
from pysindy.optimizers import STLSQ

# Default sparsity threshold. Swept in Stage C (sparsity-vs-error
# ablation) so the final value is a documented choice, not a default.
DEFAULT_THRESHOLD = 0.01


def library_feature_names() -> list[str]:
    names = ["1"]
    names += ["theta_x", "theta_y", "theta_z"]
    names += ["omega_x", "omega_y", "omega_z"]
    names += ["u1", "u2", "u3", "u4"]
    axes = ["x", "y", "z"]
    for i in range(3):
        for j in range(i, 3):
            names.append(f"omega_{axes[i]}*omega_{axes[j]}")
    for i in range(3):
        for j in range(3):
            names.append(f"theta_{axes[i]}*omega_{axes[j]}")
    return names


def build_library(state: np.ndarray, control: np.ndarray) -> np.ndarray:
    """(theta, omega) and u -> library matrix Theta. See module docstring.

    state: (n, 6) [theta_xyz, omega_xyz]; control: (n, 4) rotor thrusts.
    """
    theta, omega = state[:, 0:3], state[:, 3:6]
    n = len(state)
    cols = [np.ones(n)]
    cols += [theta[:, i] for i in range(3)]
    cols += [omega[:, i] for i in range(3)]
    cols += [control[:, i] for i in range(4)]
    for i in range(3):
        for j in range(i, 3):
            cols.append(omega[:, i] * omega[:, j])
    for i in range(3):
        for j in range(3):
            cols.append(theta[:, i] * omega[:, j])
    return np.column_stack(cols)


class SINDyModel:
    """Sparse identified model: d/dt [theta; omega] = Theta(state, u) @ Xi."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.coef_: np.ndarray | None = None   # (n_targets=6, n_features)
        self.feature_names = library_feature_names()

    def fit(self, state: np.ndarray, control: np.ndarray, deriv: np.ndarray) -> "SINDyModel":
        theta_lib = build_library(state, control)
        opt = STLSQ(threshold=self.threshold, alpha=0.0)
        opt.fit(theta_lib, deriv)
        self.coef_ = np.atleast_2d(opt.coef_)
        return self

    def predict_derivative(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model not fitted")
        return build_library(state, control) @ self.coef_.T

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """One explicit-Euler step, used for one-step-ahead evaluation."""
        return state + dt * self.predict_derivative(state, control)

    def active_terms(self, target_index: int) -> list[tuple[str, float]]:
        return [(n, c) for n, c in zip(self.feature_names, self.coef_[target_index]) if c != 0.0]

    def n_active(self) -> int:
        return int(np.count_nonzero(self.coef_))
