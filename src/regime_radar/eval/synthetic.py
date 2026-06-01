"""Synthetic data generators with KNOWN regime labels.

These are the ground truth against which we measure accuracy. Every generator returns
both the price series and the per-bar true label, so any downstream metric (overall
accuracy, per-regime F1, transition lag) is well-defined.

Generators:
    - gbm_trending(direction, ...)              → TRENDING_UP or TRENDING_DOWN
    - ou_mean_reverting(...)                    → MEAN_REVERTING
    - high_vol_chop(...)                        → HIGH_VOL_CHOP
    - low_vol_grind(...)                        → LOW_VOL_GRIND
    - breakout_jump(...)                        → BREAKOUT
    - regime_switching(specs, ...)              → stitched series with regime change points

All are reproducible via `seed`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from regime_radar.models import RegimeLabel


@dataclass(frozen=True)
class SyntheticSeries:
    """A synthetic price series with per-bar ground-truth labels."""

    prices: np.ndarray  # length n+1 (one more than labels — initial price)
    labels: np.ndarray  # length n, dtype=object (RegimeLabel)
    name: str

    @property
    def n(self) -> int:
        return len(self.labels)


def gbm_trending(
    n: int = 504,
    mu_annual: float = 0.20,
    sigma_annual: float = 0.15,
    s0: float = 100.0,
    direction: str = "up",
    seed: int = 0,
) -> SyntheticSeries:
    """Geometric Brownian Motion with positive or negative drift."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    if direction not in ("up", "down"):
        raise ValueError("direction must be 'up' or 'down'")
    mu = mu_annual if direction == "up" else -mu_annual
    z = rng.standard_normal(n)
    r = (mu - 0.5 * sigma_annual**2) * dt + sigma_annual * np.sqrt(dt) * z
    prices = s0 * np.exp(np.cumsum(np.concatenate([[0.0], r])))
    label = RegimeLabel.TRENDING_UP if direction == "up" else RegimeLabel.TRENDING_DOWN
    labels = np.array([label] * n, dtype=object)
    return SyntheticSeries(prices=prices, labels=labels, name=f"gbm_{direction}")


def ou_mean_reverting(
    n: int = 504,
    mu: float = 100.0,
    theta: float = 1.5,
    sigma: float = 0.10,
    s0: float = 100.0,
    seed: int = 0,
) -> SyntheticSeries:
    """Ornstein-Uhlenbeck on log-price → mean-reverting prices."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    prices = np.empty(n + 1)
    prices[0] = s0
    log_mu = np.log(mu)
    x = np.log(s0)
    for t in range(n):
        x = x + theta * (log_mu - x) * dt + sigma * np.sqrt(dt) * rng.standard_normal()
        prices[t + 1] = np.exp(x)
    labels = np.array([RegimeLabel.MEAN_REVERTING] * n, dtype=object)
    return SyntheticSeries(prices=prices, labels=labels, name="ou_meanrev")


def high_vol_chop(
    n: int = 504,
    sigma_annual: float = 0.45,
    s0: float = 100.0,
    seed: int = 0,
) -> SyntheticSeries:
    """High-vol zero-drift random walk — the 'choppy' regime."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    z = rng.standard_normal(n)
    r = -0.5 * sigma_annual**2 * dt + sigma_annual * np.sqrt(dt) * z
    prices = s0 * np.exp(np.cumsum(np.concatenate([[0.0], r])))
    labels = np.array([RegimeLabel.HIGH_VOL_CHOP] * n, dtype=object)
    return SyntheticSeries(prices=prices, labels=labels, name="high_vol_chop")


def low_vol_grind(
    n: int = 504,
    mu_annual: float = 0.05,
    sigma_annual: float = 0.06,
    s0: float = 100.0,
    seed: int = 0,
) -> SyntheticSeries:
    """Low-vol, weak-drift regime — calm grind."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    z = rng.standard_normal(n)
    r = (mu_annual - 0.5 * sigma_annual**2) * dt + sigma_annual * np.sqrt(dt) * z
    prices = s0 * np.exp(np.cumsum(np.concatenate([[0.0], r])))
    labels = np.array([RegimeLabel.LOW_VOL_GRIND] * n, dtype=object)
    return SyntheticSeries(prices=prices, labels=labels, name="low_vol_grind")


def breakout_jump(
    n: int = 504,
    sigma_annual: float = 0.20,
    jump_at: int = 252,
    jump_size: float = 0.15,
    s0: float = 100.0,
    seed: int = 0,
) -> SyntheticSeries:
    """Calm regime followed by a large positive jump and sustained drift."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    prices = np.empty(n + 1)
    prices[0] = s0
    labels = np.empty(n, dtype=object)
    for t in range(n):
        if t == jump_at:
            prices[t + 1] = prices[t] * (1 + jump_size)
            labels[t] = RegimeLabel.BREAKOUT
        else:
            z = rng.standard_normal()
            mu_eff = 0.0 if t < jump_at else 0.30
            r = (mu_eff - 0.5 * sigma_annual**2) * dt + sigma_annual * np.sqrt(dt) * z
            prices[t + 1] = prices[t] * np.exp(r)
            labels[t] = RegimeLabel.LOW_VOL_GRIND if t < jump_at else RegimeLabel.BREAKOUT
    return SyntheticSeries(prices=prices, labels=labels, name="breakout_jump")


@dataclass(frozen=True)
class RegimeSpec:
    """One segment of a regime-switching series."""

    generator: str  # one of: 'gbm_up', 'gbm_down', 'ou', 'chop', 'grind'
    n: int
    seed: int = 0


def regime_switching(specs: list[RegimeSpec], s0: float = 100.0) -> SyntheticSeries:
    """Stitch multiple segments into one series, preserving ground-truth labels.

    Each new segment continues from the last price of the previous segment, so the price
    series is continuous. Labels are the concatenation of each segment's labels.
    """
    parts: list[SyntheticSeries] = []
    cur_price = s0
    for spec in specs:
        if spec.generator == "gbm_up":
            s = gbm_trending(n=spec.n, direction="up", s0=cur_price, seed=spec.seed)
        elif spec.generator == "gbm_down":
            s = gbm_trending(n=spec.n, direction="down", s0=cur_price, seed=spec.seed)
        elif spec.generator == "ou":
            s = ou_mean_reverting(n=spec.n, mu=cur_price, s0=cur_price, seed=spec.seed)
        elif spec.generator == "chop":
            s = high_vol_chop(n=spec.n, s0=cur_price, seed=spec.seed)
        elif spec.generator == "grind":
            s = low_vol_grind(n=spec.n, s0=cur_price, seed=spec.seed)
        else:
            raise ValueError(f"Unknown generator: {spec.generator}")
        parts.append(s)
        cur_price = float(s.prices[-1])

    # Stitch: keep first segment's full price, subsequent segments drop the duplicate first price.
    prices_concat = [parts[0].prices]
    labels_concat = [parts[0].labels]
    for p in parts[1:]:
        prices_concat.append(p.prices[1:])
        labels_concat.append(p.labels)
    return SyntheticSeries(
        prices=np.concatenate(prices_concat),
        labels=np.concatenate(labels_concat),
        name="regime_switching",
    )
