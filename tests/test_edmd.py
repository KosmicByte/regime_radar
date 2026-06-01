"""Tests for DMD and EDMD — verify on a known linear system."""

from __future__ import annotations

import numpy as np

from regime_radar.core.dmd import fit_dmd
from regime_radar.core.edmd import fit_edmd


def test_dmd_recovers_eigenvalues_of_known_linear_system() -> None:
    """If x_{t+1} = A x_t for known A, DMD must recover A's eigenvalues."""
    rng = np.random.default_rng(42)
    A = np.array([[0.95, 0.1], [-0.1, 0.85]])
    true_eigs = np.sort(np.abs(np.linalg.eigvals(A)))

    x = rng.standard_normal((2, 1))
    snapshots = [x[:, 0]]
    for _ in range(99):
        x = A @ x
        snapshots.append(x[:, 0])
    snap = np.array(snapshots).T  # shape (2, 100)

    res = fit_dmd(snap, rank=2)
    recovered = np.sort(np.abs(res.eigenvalues))
    np.testing.assert_allclose(recovered, true_eigs, atol=1e-6)


def test_dmd_spectral_gap_nonnegative() -> None:
    rng = np.random.default_rng(0)
    snap = rng.standard_normal((5, 50))
    res = fit_dmd(snap, rank=3)
    assert res.spectral_gap >= 0


def test_dmd_mode_energies_sum_to_one() -> None:
    rng = np.random.default_rng(1)
    snap = rng.standard_normal((5, 50))
    res = fit_dmd(snap, rank=3)
    assert res.mode_energies.sum() == 0 or abs(res.mode_energies.sum() - 1.0) < 1e-9


def test_edmd_runs_on_gbm_and_orders_by_magnitude(gbm_up: np.ndarray) -> None:
    res = fit_edmd(gbm_up, rank=8)
    mags = np.abs(res.eigenvalues)
    # Sorted descending after our reorder step
    assert np.all(mags[:-1] >= mags[1:] - 1e-12)


def test_edmd_dominant_eigenvalue_near_unit_for_gbm(gbm_up: np.ndarray) -> None:
    """GBM is locally a random walk; the Koopman operator should have a dominant mode
    with magnitude near 1.0 (persistent)."""
    res = fit_edmd(gbm_up, rank=8)
    assert 0.7 < abs(res.eigenvalues[0]) < 1.15


def test_edmd_ou_has_strong_decay_mode(ou_series: np.ndarray) -> None:
    """OU is strongly mean-reverting; we expect at least one eigenvalue well below 1."""
    res = fit_edmd(ou_series, rank=8)
    mags = np.abs(res.eigenvalues)
    assert mags.min() < 0.85
