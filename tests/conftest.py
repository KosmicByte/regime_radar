"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from regime_radar.eval.synthetic import (
    gbm_trending,
    high_vol_chop,
    low_vol_grind,
    ou_mean_reverting,
)


@pytest.fixture
def gbm_up() -> np.ndarray:
    return gbm_trending(n=300, direction="up", seed=1).prices


@pytest.fixture
def gbm_down() -> np.ndarray:
    return gbm_trending(n=300, direction="down", seed=2).prices


@pytest.fixture
def ou_series() -> np.ndarray:
    return ou_mean_reverting(n=300, seed=3).prices


@pytest.fixture
def chop_series() -> np.ndarray:
    return high_vol_chop(n=300, seed=4).prices


@pytest.fixture
def grind_series() -> np.ndarray:
    return low_vol_grind(n=300, seed=5).prices
