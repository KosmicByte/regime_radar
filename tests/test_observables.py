"""Tests for core/observables.py — pure math, must be exactly right."""

from __future__ import annotations

import numpy as np
import pytest

from regime_radar.core.observables import (
    ObservableConfig,
    build_observables,
    drawdown,
    log_returns,
    rolling_mean,
    rolling_std,
    standardise,
)


def test_log_returns_basic() -> None:
    close = np.array([100.0, 110.0, 99.0])
    r = log_returns(close)
    assert r.shape == (2,)
    np.testing.assert_allclose(r, [np.log(1.1), np.log(99 / 110)])


def test_log_returns_constant_price_is_zero() -> None:
    close = np.full(10, 100.0)
    r = log_returns(close)
    np.testing.assert_allclose(r, np.zeros(9), atol=1e-12)


def test_rolling_std_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50)
    rs = rolling_std(x, window=10)
    for i in range(9, 50):
        assert rs[i] == pytest.approx(np.std(x[i - 9 : i + 1]))


def test_rolling_mean_matches_numpy() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(40)
    rm = rolling_mean(x, window=5)
    for i in range(4, 40):
        assert rm[i] == pytest.approx(np.mean(x[i - 4 : i + 1]))


def test_drawdown_is_zero_on_monotonic_up() -> None:
    close = np.linspace(100, 200, 50)
    dd = drawdown(close)
    np.testing.assert_allclose(dd, np.zeros(50))


def test_drawdown_captures_peak_to_trough() -> None:
    close = np.array([100.0, 120.0, 90.0])
    dd = drawdown(close)
    assert dd[-1] == pytest.approx(90 / 120 - 1.0)


def test_build_observables_drops_nan_warmup() -> None:
    rng = np.random.default_rng(2)
    close = 100.0 * np.exp(np.cumsum(rng.standard_normal(200) * 0.01))
    psi, names = build_observables(close, ObservableConfig())
    assert not np.isnan(psi).any()
    assert len(names) == psi.shape[1]


def test_standardise_zero_mean_unit_var() -> None:
    rng = np.random.default_rng(3)
    psi = rng.standard_normal((100, 4)) * 5 + 3
    out, mean, std = standardise(psi)
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-12)
