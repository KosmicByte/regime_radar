"""Exact Dynamic Mode Decomposition (DMD).

This is the baseline that EDMD generalises. Given snapshot pairs (X, Y) where
Y = A X (under the assumption of locally-linear dynamics), we estimate the
operator's eigendecomposition without ever forming A explicitly.

Algorithm (Tu et al. 2014, exact DMD):
    1. SVD: X = U Σ V*
    2. Truncate to rank r (energy- or rank-based)
    3. Ã = U* Y V Σ⁻¹                       (r×r projection of A)
    4. Ã W = W Λ                            (eigendecomp on the small matrix)
    5. Φ = Y V Σ⁻¹ W                        (DMD modes in original space)
    6. Continuous-time growth = log(λ)/Δt
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DMDResult:
    """Output of a DMD fit."""

    eigenvalues: np.ndarray  # discrete-time λ_i, shape (r,) complex
    modes: np.ndarray  # Φ, shape (n_features, r) complex
    amplitudes: np.ndarray  # b_i, shape (r,) complex — initial amplitudes
    rank: int
    singular_values: np.ndarray  # all singular values of X (for energy diagnostics)

    @property
    def growth_rates(self) -> np.ndarray:
        """Real part of log(λ): >0 grows, <0 decays."""
        return np.real(np.log(self.eigenvalues + 1e-300))

    @property
    def frequencies(self) -> np.ndarray:
        """Imag part of log(λ)/(2π): oscillation rate in cycles per bar."""
        return np.imag(np.log(self.eigenvalues + 1e-300)) / (2 * np.pi)

    @property
    def mode_energies(self) -> np.ndarray:
        """Energy in each mode (|b_i| × ||Φ_i||), normalised to sum to 1."""
        e = np.abs(self.amplitudes) * np.linalg.norm(self.modes, axis=0)
        s = e.sum()
        return e / s if s > 0 else e

    @property
    def spectral_gap(self) -> float:
        """|λ_(1)| - |λ_(next distinct)|. For real data, eigenvalues come in conjugate
        pairs with the same magnitude, so we look past pairs to find a meaningful gap."""
        mags = np.sort(np.abs(self.eigenvalues))[::-1]
        if len(mags) < 2:
            return 0.0
        top = mags[0]
        for m in mags[1:]:
            if top - m > 1e-6:
                return float(top - m)
        return 0.0


def _choose_rank(s: np.ndarray, rank: int | None, energy: float) -> int:
    """Pick truncation rank either explicitly or by retained energy fraction."""
    if rank is not None:
        return min(rank, len(s))
    cum = np.cumsum(s**2) / (s**2).sum()
    return int(np.searchsorted(cum, energy) + 1)


def fit_dmd(
    snapshots: np.ndarray,
    rank: int | None = None,
    energy: float = 0.99,
) -> DMDResult:
    """Fit DMD on a snapshot matrix of shape (n_features, n_snapshots).

    Args:
        snapshots: data matrix where columns are time-ordered states.
        rank: explicit truncation rank; if None, energy-based.
        energy: target retained energy fraction when `rank` is None.

    Returns:
        DMDResult.
    """
    if snapshots.ndim != 2 or snapshots.shape[1] < 2:
        raise ValueError("snapshots must be 2-D with at least 2 columns")

    X = snapshots[:, :-1]
    Y = snapshots[:, 1:]

    # Step 1: SVD of X
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    r = _choose_rank(S, rank, energy)
    Ur, Sr, Vr = U[:, :r], S[:r], Vh[:r, :].conj().T

    # Step 2: projected operator Ã
    Sr_inv = np.diag(1.0 / Sr)
    A_tilde = Ur.conj().T @ Y @ Vr @ Sr_inv

    # Step 3: eigendecomposition of Ã
    eigvals, W = np.linalg.eig(A_tilde)

    # Step 4: DMD modes in original space (exact DMD; Tu et al.)
    modes = Y @ Vr @ Sr_inv @ W

    # Step 5: amplitudes — project initial snapshot onto modes
    x0 = X[:, 0]
    amplitudes, *_ = np.linalg.lstsq(modes, x0, rcond=None)

    return DMDResult(
        eigenvalues=eigvals,
        modes=modes,
        amplitudes=amplitudes,
        rank=r,
        singular_values=S,
    )
