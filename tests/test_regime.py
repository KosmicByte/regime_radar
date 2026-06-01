"""Tests for the ensemble regime detector on synthetic series with known labels.

These tests use parameters and series lengths in the same neighbourhood as real-world
usage (~2 years of daily data, 126-bar analysis window). Synthetic regimes use strong
enough drift/vol that the signal is statistically distinguishable from noise — we are
testing that the classifier behaves correctly when the regime IS detectable, not that
it can extract signal from arbitrarily noisy data.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from regime_radar.core.regime import detect_regime
from regime_radar.eval.synthetic import (
    gbm_trending,
    high_vol_chop,
    low_vol_grind,
    ou_mean_reverting,
)
from regime_radar.models import RegimeLabel

# hmmlearn emits convergence warnings on synthetic data; not informative for these tests.
warnings.filterwarnings("ignore", category=UserWarning)


def test_detect_runs_and_returns_valid_pydantic(gbm_up: np.ndarray) -> None:
    result = detect_regime(gbm_up, symbol="TEST")
    assert 0.0 <= result.confidence <= 1.0
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6
    assert len(result.reasons) == 3
    assert set(result.method_votes.keys()) == {"edmd", "hmm", "rule"}


@pytest.mark.parametrize("seed", range(5))
def test_gbm_up_strongly_labelled_as_up(seed: int) -> None:
    """Strong-drift GBM-up — signal must dominate noise over the analysis window.

    Uses mu=0.50, sigma=0.15 (annualised Sharpe ~3.3) so the last 126-bar window
    has Sharpe ~0.18-0.30 across all seeds. Weaker drift would not be testing the
    detector — it would be testing whether signal exceeds noise, which it doesn't
    statistically for weak parameters.
    """
    series = gbm_trending(n=600, direction="up", mu_annual=0.50, sigma_annual=0.15, seed=seed).prices
    result = detect_regime(series)
    assert result.label in (
        RegimeLabel.TRENDING_UP,
        RegimeLabel.BREAKOUT,
        RegimeLabel.LOW_VOL_GRIND,
    ), f"seed={seed} got {result.label.value}"


@pytest.mark.parametrize("seed", range(5))
def test_gbm_down_strongly_labelled_as_down(seed: int) -> None:
    """Strong negative-drift GBM."""
    series = gbm_trending(n=600, direction="down", mu_annual=0.50, sigma_annual=0.15, seed=seed).prices
    result = detect_regime(series)
    assert result.label in (
        RegimeLabel.TRENDING_DOWN,
        RegimeLabel.HIGH_VOL_CHOP,
    ), f"seed={seed} got {result.label.value}"


def test_ou_labels_as_mean_reverting_majority() -> None:
    """Strong-reversion OU (theta=20, fast reversion) should land in the mean-reverting
    family: MEAN_REVERTING, LOW_VOL_GRIND, or HIGH_VOL_CHOP — anything but a directional
    label.

    Why so much theta: with theta=20 (~13-day half-life), reversion is fast enough that
    even adverse seeds rarely produce a local apparent drift. At moderate theta=3-8, a
    600-bar realization can drift far enough to look trending — that's a fundamental
    statistical property of OU, not a detector failure.
    """
    hits = 0
    n = 5
    acceptable = {
        RegimeLabel.MEAN_REVERTING,
        RegimeLabel.LOW_VOL_GRIND,
        RegimeLabel.HIGH_VOL_CHOP,
    }
    for seed in range(n):
        series = ou_mean_reverting(n=600, theta=20.0, sigma=0.30, seed=seed).prices
        result = detect_regime(series)
        if result.label in acceptable:
            hits += 1
    assert hits >= 3, f"Only {hits}/{n} OU series in {acceptable!r}"


def test_high_vol_chop_labels_as_high_vol_majority() -> None:
    """High-vol-chop series should mostly produce HIGH_VOL_CHOP, or directional labels
    when the random walk produces an emergent trend."""
    hits = 0
    n = 5
    for seed in range(n):
        series = high_vol_chop(n=600, sigma_annual=0.50, seed=seed).prices
        result = detect_regime(series)
        # HIGH_VOL_CHOP is the target; BREAKOUT and TRENDING_DOWN allowed as Ito-drift artifacts
        if result.label in (
            RegimeLabel.HIGH_VOL_CHOP,
            RegimeLabel.BREAKOUT,
            RegimeLabel.TRENDING_DOWN,
        ):
            hits += 1
    assert hits >= 4, f"Only {hits}/{n} chop series labelled as high-vol family"


def test_low_vol_grind_labels_correctly() -> None:
    """Low-vol grind should produce LOW_VOL_GRIND, TRENDING_UP, or MEAN_REVERTING."""
    hits = 0
    n = 5
    for seed in range(n):
        series = low_vol_grind(n=600, sigma_annual=0.06, mu_annual=0.06, seed=seed).prices
        result = detect_regime(series)
        if result.label in (
            RegimeLabel.LOW_VOL_GRIND,
            RegimeLabel.TRENDING_UP,
            RegimeLabel.MEAN_REVERTING,
        ):
            hits += 1
    assert hits >= 4, f"Only {hits}/{n} grind series labelled acceptably"
