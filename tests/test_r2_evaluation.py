"""Tests for R2 — economic validation, stability, and confidence intervals.

The anchor is :func:`test_switching_series_beats_null`: on a series whose regime genuinely
switches, the regime-driven strategy must beat the shuffled-regime null — and on a series with
nothing to time, it must *not* (:func:`test_pure_trend_has_no_timing_edge`). Together these
show the permutation test detects timing skill rather than rewarding any active strategy.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from regime_radar.eval.economic import (
    evaluate_economic,
    run_regime_backtest,
)
from regime_radar.eval.stability import stability
from regime_radar.eval.synthetic import (
    RegimeSpec,
    gbm_trending,
    high_vol_chop,
    regime_switching,
)
from regime_radar.eval.uncertainty import bootstrap_ci, wilson_interval

warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------- economic
def test_backtest_is_point_in_time_and_shaped() -> None:
    s = gbm_trending(n=600, direction="up", seed=1)
    bt = run_regime_backtest(s.prices, window=126, stride=5, edmd_rank=10)
    n = len(bt.returns)
    assert len(bt.buyhold_returns) == n
    assert len(bt.positions) == n
    assert len(bt.labels) == n
    assert len(bt.equity) == n
    assert np.all(np.isfinite(bt.returns))


def test_switching_series_beats_null() -> None:
    specs = [RegimeSpec("gbm_up", 300, 1), RegimeSpec("gbm_down", 300, 2), RegimeSpec("gbm_up", 300, 3)]
    s = regime_switching(specs)
    bt = run_regime_backtest(s.prices, window=126, stride=5, edmd_rank=10)
    summ = evaluate_economic(bt, n_permutations=500, seed=1)
    assert summ.sharpe > summ.buyhold_sharpe  # timing adds value when the regime switches
    assert summ.beats_null(0.10)  # significant edge over the shuffled-regime null


def test_pure_trend_has_no_timing_edge() -> None:
    # On a single-regime uptrend, holding long always ≈ buy-hold, so there is no *timing*
    # skill — the permutation test should NOT flag significance.
    s = gbm_trending(n=900, direction="up", mu_annual=0.30, sigma_annual=0.15, seed=3)
    bt = run_regime_backtest(s.prices, window=126, stride=5, edmd_rank=10)
    summ = evaluate_economic(bt, n_permutations=500, seed=1)
    assert not summ.beats_null(0.05)


def test_chop_keeps_strategy_flat() -> None:
    s = high_vol_chop(n=700, sigma_annual=0.50, seed=3)
    bt = run_regime_backtest(s.prices, window=126, stride=5, edmd_rank=10)
    summ = evaluate_economic(bt, n_permutations=200, seed=1)
    # Chop maps to a flat position, so turnover and Sharpe are ~0 (we correctly stay out).
    assert summ.turnover < 0.05
    assert abs(summ.sharpe) < 0.5


def test_summary_finite() -> None:
    s = gbm_trending(n=500, direction="down", seed=2)
    bt = run_regime_backtest(s.prices, window=126, stride=5, edmd_rank=10)
    summ = evaluate_economic(bt, n_permutations=200)
    for v in (summ.sharpe, summ.buyhold_sharpe, summ.ann_return, summ.max_drawdown, summ.turnover):
        assert np.isfinite(v)
    assert 0.0 <= summ.permutation_p_value <= 1.0


def test_gap_in_close_does_not_poison_stats() -> None:
    """A missing close (real-feed gap, e.g. yfinance NaN bar) must not NaN the whole summary."""
    s = gbm_trending(n=600, direction="up", seed=4)
    close = s.prices.copy()
    close[300] = np.nan  # simulate a single missing/suspended bar
    bt = run_regime_backtest(close, window=126, stride=5, edmd_rank=10)
    summ = evaluate_economic(bt, n_permutations=200)
    assert np.all(np.isfinite(bt.returns))
    for v in (summ.sharpe, summ.buyhold_sharpe, summ.ann_return, summ.max_drawdown):
        assert np.isfinite(v), f"non-finite summary value: {v}"


# --------------------------------------------------------------------- stability
def test_stability_counts_switches() -> None:
    labels = ["a", "a", "b", "b", "b", "a"]
    rep = stability(labels)
    assert rep.n_switches == 2
    assert rep.max_dwell == 3
    assert rep.whipsaw_rate == pytest.approx(2 / 5)


def test_stability_constant_sequence() -> None:
    rep = stability(["a"] * 10)
    assert rep.n_switches == 0
    assert rep.whipsaw_rate == 0.0
    assert rep.mean_dwell == 10.0


# --------------------------------------------------------------------- uncertainty
def test_wilson_interval_brackets_proportion() -> None:
    lo, hi = wilson_interval(38, 50)
    assert 0.0 <= lo < 38 / 50 < hi <= 1.0


def test_bootstrap_ci_brackets_mean() -> None:
    sample = np.array([0.7, 0.8, 0.6, 0.9, 0.75, 0.72, 0.81])
    point, lo, hi = bootstrap_ci(sample, n_boot=1000)
    assert lo <= point <= hi
    assert abs(point - sample.mean()) < 1e-9
