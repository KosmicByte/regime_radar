"""Confidence intervals on evaluation metrics.

A headline accuracy of "76%" is a point estimate with no honesty about sampling noise. "76%
[72, 80]" tells you how much to trust it. This module provides a Wilson interval for
proportions (accuracy, coverage) and a generic bootstrap for any statistic over a sample
(e.g. per-scenario accuracy, Sharpe).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%).

    More accurate than the normal approximation for small n or proportions near 0/1, and never
    produces bounds outside [0, 1].
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(
    sample: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for ``statistic`` over ``sample``.

    Returns ``(point_estimate, lower, upper)`` at the ``1 - alpha`` level. Resamples the
    sample with replacement ``n_boot`` times. Use for per-scenario accuracy, Sharpe, etc.,
    where the sampling distribution is not a simple proportion.
    """
    sample = np.asarray(sample, dtype=float)
    point = float(statistic(sample))
    if len(sample) < 2:
        return (point, point, point)
    rng = np.random.default_rng(seed)
    n = len(sample)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = statistic(sample[rng.integers(0, n, n)])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (point, lo, hi)
