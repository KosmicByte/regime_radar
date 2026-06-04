"""Tests for R1 — calibration, conformal prediction sets, and OOD scoring.

Two properties anchor this suite: temperature scaling must never change the label
(:func:`test_temperature_preserves_label`), and conformal sets must hit their coverage target
on held-out data (:func:`test_conformal_coverage_on_holdout`).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from regime_radar.calibration.artifact import CalibrationArtifact
from regime_radar.calibration.calibrator import apply_temperature, fit_temperature
from regime_radar.calibration.conformal import fit_threshold, prediction_set
from regime_radar.calibration.fit import fit_calibration
from regime_radar.calibration.ood import fit_reference, ood_score
from regime_radar.eval.calibration_metrics import (
    brier_score,
    empirical_coverage,
    expected_calibration_error,
)
from regime_radar.eval.synthetic import (
    gbm_trending,
    high_vol_chop,
    low_vol_grind,
    ou_mean_reverting,
)

warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------- pure math
@pytest.fixture
def overconfident() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    k, n = 7, 1500
    true = rng.integers(0, k, n)
    probs = np.full((n, k), 0.01)
    for i in range(n):
        target = true[i] if rng.random() < 0.65 else rng.integers(0, k)
        probs[i, target] = 0.9
    probs /= probs.sum(axis=1, keepdims=True)
    return probs, true


def test_temperature_preserves_label(overconfident) -> None:
    probs, _ = overconfident
    cal = apply_temperature(probs, 2.5)
    assert np.array_equal(probs.argmax(1), cal.argmax(1))


def test_temperature_reduces_ece(overconfident) -> None:
    probs, true = overconfident
    t = fit_temperature(probs, true)
    cal = apply_temperature(probs, t)
    assert expected_calibration_error(cal, true) < expected_calibration_error(probs, true)
    assert brier_score(cal, true) <= brier_score(probs, true) + 1e-9


def test_conformal_coverage_on_holdout(overconfident) -> None:
    probs, true = overconfident
    t = fit_temperature(probs[:750], true[:750])
    cal_fit = apply_temperature(probs[:750], t)
    cal_test = apply_temperature(probs[750:], t)
    q = fit_threshold(cal_fit, true[:750], alpha=0.1)
    assert empirical_coverage(cal_test, true[750:], q) >= 0.85


def test_prediction_set_never_empty() -> None:
    probs = np.array([0.05, 0.05, 0.9])
    labels = ["a", "b", "c"]
    assert prediction_set(probs, threshold=0.0, labels=labels) == ["c"]


def test_ood_separates_in_and_out() -> None:
    rng = np.random.default_rng(1)
    mean, std, inv = fit_reference(rng.normal(0, 1, (400, 5)))
    assert ood_score(rng.normal(0, 1, 5), mean, std, inv) < 0.95
    assert ood_score(np.full(5, 8.0), mean, std, inv) > 0.99


def test_ood_score_robust_to_nonfinite_features() -> None:
    """A non-finite feature (degenerate real-market window) must never yield a NaN score."""
    rng = np.random.default_rng(2)
    mean, std, inv = fit_reference(rng.normal(0, 1, (400, 7)))
    for bad_val in (np.nan, np.inf, -np.inf):
        x = rng.normal(0, 1, 7)
        x[2] = bad_val
        s = ood_score(x, mean, std, inv)
        assert np.isfinite(s) and 0.0 <= s <= 1.0
    # Imputation is neutral: a NaN feature scores as if that dim were at the reference mean.
    x_nan = rng.normal(0, 1, 7)
    x_nan[3] = np.nan
    x_mean = x_nan.copy()
    x_mean[3] = mean[3]
    assert abs(ood_score(x_nan, mean, std, inv) - ood_score(x_mean, mean, std, inv)) < 1e-9


# --------------------------------------------------------------------- artifact
def test_artifact_roundtrip(tmp_path) -> None:
    art = CalibrationArtifact(
        labels=["a", "b"],
        temperature=1.5,
        conformal_threshold=0.3,
        coverage_level=0.9,
        ood_mean=[0.0, 0.0],
        ood_std=[1.0, 1.0],
        ood_inv_cov=[[1.0, 0.0], [0.0, 1.0]],
        ood_dim=2,
        ood_threshold=0.99,
        fit_source="synthetic",
        model_version="1.1.0",
    )
    path = tmp_path / "calibration.json"
    art.save(path)
    back = CalibrationArtifact.load(path)
    assert back.temperature == 1.5 and back.is_synthetic_fit()
    assert CalibrationArtifact.try_load(tmp_path / "missing.json") is None


# --------------------------------------------------------------------- detection wiring
@pytest.fixture(scope="module")
def fitted_artifact(tmp_path_factory) -> str:
    battery = []
    for s in range(2):
        battery += [
            gbm_trending(n=1008, direction="up", mu_annual=0.30, sigma_annual=0.15, seed=s),
            gbm_trending(n=1008, direction="down", mu_annual=0.30, sigma_annual=0.15, seed=s),
            ou_mean_reverting(n=1008, theta=15.0, sigma=0.25, seed=s),
            high_vol_chop(n=1008, sigma_annual=0.50, seed=s),
            low_vol_grind(n=1008, sigma_annual=0.06, mu_annual=0.06, seed=s),
        ]
    artifact, _ = fit_calibration(battery, window=252, step=21, alpha=0.1)
    path = tmp_path_factory.mktemp("calib") / "calibration.json"
    artifact.save(path)
    return str(path)


def _fresh_settings(monkeypatch, artifact_path: str) -> None:
    monkeypatch.setenv("REGIME_CALIBRATION_ARTIFACT", artifact_path)
    import regime_radar.config as cfg
    import regime_radar.core.regime as reg

    cfg._settings = None  # type: ignore[attr-defined]
    reg._ARTIFACT_CACHE.clear()


def test_detect_without_artifact_is_uncalibrated(monkeypatch, tmp_path) -> None:
    _fresh_settings(monkeypatch, str(tmp_path / "nope.json"))
    from regime_radar.core.regime import detect_regime

    close = gbm_trending(n=300, direction="up", seed=7).prices
    r = detect_regime(close=close, edmd_rank=10)
    assert r.calibrated is False
    assert r.calibrator_version is None


def test_detect_with_artifact_is_calibrated(monkeypatch, fitted_artifact) -> None:
    _fresh_settings(monkeypatch, fitted_artifact)
    from regime_radar.core.regime import detect_regime

    close = gbm_trending(n=300, direction="up", seed=8).prices
    ts = pd.Series(pd.date_range("2024-01-01", periods=len(close), freq="D"))
    cal = detect_regime(close=close, timestamps=ts, edmd_rank=10, calibrate=True)
    assert cal.calibrated is True
    assert cal.calibrator_version.startswith("synthetic@")
    assert abs(sum(cal.probabilities.values()) - 1.0) < 1e-6
    assert abs(cal.confidence - cal.probabilities[cal.label]) < 1e-9
    assert cal.label in cal.prediction_set
    assert 0.0 <= cal.ood_score <= 1.0
    assert cal.coverage_level == 0.9


def test_calibration_preserves_label(monkeypatch, fitted_artifact) -> None:
    _fresh_settings(monkeypatch, fitted_artifact)
    from regime_radar.core.regime import detect_regime

    for seed in (8, 11, 21):
        close = gbm_trending(n=300, direction="up", seed=seed).prices
        raw = detect_regime(close=close, edmd_rank=10, calibrate=False)
        cal = detect_regime(close=close, edmd_rank=10, calibrate=True)
        assert raw.label == cal.label  # temperature scaling cannot change the decision
