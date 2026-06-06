"""Tests for the R2 remainder — benchmark confidence intervals, stability, macro-F1, and the
HMM-dampening A/B harness.

Kept to a small battery so they run fast; the statistical properties (CI brackets the point
estimate, A/B produces a paired delta with a CI and a retire/keep verdict) are what matter.
"""

from __future__ import annotations

import warnings

import pytest

from regime_radar.eval.metrics import ab_dampening, benchmark
from regime_radar.eval.synthetic import (
    gbm_trending,
    high_vol_chop,
    low_vol_grind,
    ou_mean_reverting,
)

warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture(scope="module")
def small_battery() -> list:
    battery = []
    for s in range(2):
        battery += [
            gbm_trending(n=1008, direction="up", mu_annual=0.30, sigma_annual=0.15, seed=s),
            gbm_trending(n=1008, direction="down", mu_annual=0.30, sigma_annual=0.15, seed=s),
            ou_mean_reverting(n=1008, theta=15.0, sigma=0.25, seed=s),
            high_vol_chop(n=1008, sigma_annual=0.50, seed=s),
            low_vol_grind(n=1008, sigma_annual=0.06, mu_annual=0.06, seed=s),
        ]
    return battery


@pytest.fixture(scope="module")
def result(small_battery):
    return benchmark(small_battery, window=252, step=21, grouped=True)


def test_accuracy_ci_brackets_point(result) -> None:
    acc, lo, hi = result.accuracy_ci(n_boot=300)
    assert 0.0 <= lo <= acc <= hi <= 1.0
    assert abs(acc - result.overall_accuracy) < 1e-9


def test_macro_f1_in_range(result) -> None:
    macro, per = result.macro_f1()
    assert 0.0 <= macro <= 1.0
    assert all(0.0 <= v <= 1.0 for v in per.values())


def test_stability_summary(result) -> None:
    stab = result.stability_summary()
    assert 0.0 <= stab.whipsaw_rate <= 1.0
    assert stab.mean_dwell >= 1.0
    assert stab.n_switches >= 0


def test_ab_dampening_shapes(small_battery) -> None:
    ab = ab_dampening(small_battery, window=252, step=21, grouped=True)
    s = ab.summary(n_boot=300)
    for key in ("on_accuracy", "off_accuracy", "delta_mean", "delta_ci", "can_retire",
                "label_divergence", "exercised"):
        assert key in s
    d_lo, d_hi = s["delta_ci"]
    assert d_lo <= s["delta_mean"] <= d_hi
    assert isinstance(s["can_retire"], bool)
    assert isinstance(s["exercised"], bool)
    assert 0.0 <= s["label_divergence"] <= 1.0


def test_ab_uses_same_battery_keys(small_battery) -> None:
    # Both arms must score the identical scenarios for the paired test to be valid.
    ab = ab_dampening(small_battery, window=252, step=21, grouped=True)
    assert set(ab.on.per_scenario) == set(ab.off.per_scenario)
