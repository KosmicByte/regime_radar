"""Synthetic demo: detect regimes on series with known ground truth, no internet needed.

Walks through each synthetic generator, runs the detector, and shows whether it
matches the true label. Useful as a sanity check and as a template for plugging in
your own labelled data.

Run with:
    uv run python examples/synthetic_demo.py
"""

from __future__ import annotations

import warnings

from regime_radar.core.regime import detect_regime
from regime_radar.eval.synthetic import (
    breakout_jump,
    gbm_trending,
    high_vol_chop,
    low_vol_grind,
    ou_mean_reverting,
    regime_switching,
    RegimeSpec,
)
from regime_radar.models import RegimeLabel

warnings.filterwarnings("ignore", category=UserWarning)  # silence hmmlearn convergence noise


def run(name: str, prices, expected_family: set[RegimeLabel]) -> None:
    result = detect_regime(prices, symbol=name)
    match = "✓" if result.label in expected_family else "✗"
    print(
        f"{match} {name:25s} → {result.label.value:18s} "
        f"(conf {result.confidence:.0%}, ann vol {result.realized_vol:.0%}, "
        f"Sharpe {result.trend_strength:+.2f})"
    )


def main() -> None:
    print("Synthetic regime detection demo\n" + "=" * 50)

    s = gbm_trending(n=600, direction="up", mu_annual=0.40, sigma_annual=0.15, seed=0).prices
    run("GBM up (strong)", s, {RegimeLabel.TRENDING_UP, RegimeLabel.BREAKOUT})

    s = gbm_trending(n=600, direction="down", mu_annual=0.40, sigma_annual=0.15, seed=1).prices
    run("GBM down (strong)", s, {RegimeLabel.TRENDING_DOWN, RegimeLabel.HIGH_VOL_CHOP})

    s = ou_mean_reverting(n=600, theta=15.0, sigma=0.25, seed=2).prices
    run("OU mean-reverting", s, {RegimeLabel.MEAN_REVERTING, RegimeLabel.HIGH_VOL_CHOP})

    s = high_vol_chop(n=600, sigma_annual=0.50, seed=3).prices
    run("High-vol chop", s, {RegimeLabel.HIGH_VOL_CHOP, RegimeLabel.TRENDING_DOWN, RegimeLabel.BREAKOUT})

    s = low_vol_grind(n=600, sigma_annual=0.06, mu_annual=0.06, seed=4).prices
    run("Low-vol grind", s, {RegimeLabel.LOW_VOL_GRIND, RegimeLabel.MEAN_REVERTING, RegimeLabel.TRENDING_UP})

    s = breakout_jump(n=600, jump_at=400, jump_size=0.20, seed=5).prices
    run("Breakout (jump)", s, {RegimeLabel.BREAKOUT, RegimeLabel.TRENDING_UP, RegimeLabel.HIGH_VOL_CHOP})

    s = regime_switching(
        [
            RegimeSpec(generator="gbm_up", n=200, seed=6),
            RegimeSpec(generator="chop", n=200, seed=7),
            RegimeSpec(generator="ou", n=200, seed=8),
        ]
    ).prices
    print(f"  Regime-switching: last segment is OU, so expect mean-reverting family")
    run("Regime-switching", s, {RegimeLabel.MEAN_REVERTING, RegimeLabel.HIGH_VOL_CHOP, RegimeLabel.LOW_VOL_GRIND})


if __name__ == "__main__":
    main()
