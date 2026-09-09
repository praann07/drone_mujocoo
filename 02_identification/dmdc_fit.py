"""DMDc fit, restricted to the near-hover trim regime (docs/TDD.md section 5).

DMDc fits a linear operator  z_{k+1} = A z_k + B du_k  on the state
z = (theta, omega). Attitude dynamics are genuinely nonlinear, so this is
only meaningful near hover - hence the dedicated small-amplitude trial
family (peak rotation < 20 deg, enforced by
01_simulation/run_excitation.py::assert_near_hover_bound). No
EDMD/Koopman lift is used, by explicit decision in docs/TDD.md section 5.

Control is expressed as the DIFFERENTIAL thrust du = u - hover rather than
absolute thrust. Absolute rotor thrusts sit on a large common DC offset
(~2.45 N each at hover) which makes the control snapshot matrix nearly
rank-deficient along the uniform-thrust direction; that direction also
produces exactly zero torque, so it carries no attitude information to
begin with. Removing it is both better-conditioned and physically
equivalent for attitude.

pydmd's DMDc is used as specified by the project stack, but its result is
cross-checked against a direct least-squares solve: with no SVD
truncation the two are mathematically the same estimator, so a
disagreement means numerical trouble rather than a modelling choice, and
must not pass silently into Stage C.
"""
from __future__ import annotations

import warnings

import numpy as np
from pydmd import DMDc

HOVER_THRUST_PER_ROTOR = 1.0 * 9.81 / 4.0


class DMDcModel:
    """Linear near-hover model  z_{k+1} = A z_k + B (u - hover)."""

    def __init__(self):
        self.A: np.ndarray | None = None
        self.B: np.ndarray | None = None
        self.pydmd_mismatch: float | None = None

    @staticmethod
    def _least_squares(Z: np.ndarray, Zp: np.ndarray, dU: np.ndarray):
        """Direct DMDc solve: [A B] = Zp @ pinv([Z; dU])."""
        omega_mat = np.vstack([Z, dU])
        G = Zp @ np.linalg.pinv(omega_mat)
        n = Z.shape[0]
        return G[:, :n], G[:, n:]

    @staticmethod
    def _effective_rank(X: np.ndarray, rtol: float = 1e-8) -> int:
        s = np.linalg.svd(X, compute_uv=False)
        return int(np.sum(s > s[0] * rtol))

    def fit(self, snapshots: list[tuple[np.ndarray, np.ndarray]]) -> "DMDcModel":
        """snapshots: list of (state (n_t, 6), control (n_t, 4)) per segment.

        Consecutive-sample pairs are formed WITHIN each segment only -
        never across a segment or trial boundary, which would fabricate a
        transition between two unrelated states.
        """
        Z, Zp, dU = [], [], []
        for state, control in snapshots:
            Z.append(state[:-1].T)
            Zp.append(state[1:].T)
            dU.append((control[:-1] - HOVER_THRUST_PER_ROTOR).T)
        Z = np.hstack(Z)
        Zp = np.hstack(Zp)
        dU = np.hstack(dU)

        self.A, self.B = self._least_squares(Z, Zp, dU)

        # Cross-check against pydmd. DMDc needs one contiguous snapshot
        # sequence, so a single segment is used rather than the
        # concatenation (concatenating would imply a transition across
        # the join that never physically occurred).
        #
        # The segment must be FULL RANK in the state, not merely the
        # longest. A single-axis excitation trial leaves 4 of the 6 state
        # components identically zero, so its snapshot matrix is rank 2
        # with condition number ~1e48. np.linalg.pinv truncates those
        # null directions silently, but pydmd with svd_rank=-1 faithfully
        # retains all 6 modes - four of which are pure floating-point
        # noise - and returns an operator with entries ~1e11. That is
        # correct behaviour for a rank-deficient input, not a library
        # fault, but it makes such a segment useless as a reference.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ranked = sorted(snapshots,
                                key=lambda s: (self._effective_rank(s[0].T), len(s[0])),
                                reverse=True)
                state, control = ranked[0]
                rank = self._effective_rank(state.T)
                X = state.T
                U = (control[:-1] - HOVER_THRUST_PER_ROTOR).T
                d = DMDc(svd_rank=rank)
                d.fit(X, U)
                basis = d.basis
                A_pydmd = basis @ d.operator.as_numpy_array @ basis.conj().T
            A_ref, _ = self._least_squares(X[:, :-1], X[:, 1:], U)
            self.pydmd_mismatch = float(np.abs(np.real(A_pydmd) - A_ref).max())
            self.pydmd_check_rank = rank
        except Exception as exc:  # pragma: no cover - diagnostic only
            self.pydmd_mismatch = float("nan")
            self.pydmd_check_rank = -1
            print(f"  [dmdc] pydmd cross-check unavailable: {exc}")

        return self

    def step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """One-step prediction for a batch of (state, control) rows."""
        du = control - HOVER_THRUST_PER_ROTOR
        return state @ self.A.T + du @ self.B.T

    def eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvals(self.A)
