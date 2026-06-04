"""Calibration-quality metrics: ECE, Brier, reliability curve, and conformal coverage.

These answer "are the probabilities honest?" rather than "is the label right?". A detector can
be accurate yet badly calibrated (systematically over-confident), and these metrics expose
that. Used by ``regime calibrate`` to report before/after quality and by the R2 evaluation
work later.
"""

from __future__ import annotations

import numpy as np


def expected_calibration_error(probs: np.ndarray, true_idx: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error of the top-label confidence.

    Bins predictions by their max probability, then averages |confidence − accuracy| across
    bins weighted by bin population. 0 = perfectly calibrated.
    """
    probs = np.atleast_2d(np.asarray(probs, dtype=float))
    true_idx = np.asarray(true_idx, dtype=int)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == true_idx).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        in_bin = (conf > lo) & (conf <= hi)
        if not np.any(in_bin):
            continue
        ece += (in_bin.sum() / n) * abs(conf[in_bin].mean() - correct[in_bin].mean())
    return float(ece)


def brier_score(probs: np.ndarray, true_idx: np.ndarray) -> float:
    """Multiclass Brier score — mean squared error against the one-hot truth. Lower is better."""
    probs = np.atleast_2d(np.asarray(probs, dtype=float))
    true_idx = np.asarray(true_idx, dtype=int)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(probs)), true_idx] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def reliability_curve(
    probs: np.ndarray, true_idx: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(bin_confidence, bin_accuracy, bin_count)`` for a reliability diagram."""
    probs = np.atleast_2d(np.asarray(probs, dtype=float))
    true_idx = np.asarray(true_idx, dtype=int)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == true_idx).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bc, ba, cnt = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        in_bin = (conf > lo) & (conf <= hi)
        if np.any(in_bin):
            bc.append(float(conf[in_bin].mean()))
            ba.append(float(correct[in_bin].mean()))
            cnt.append(int(in_bin.sum()))
    return np.array(bc), np.array(ba), np.array(cnt)


def empirical_coverage(probs: np.ndarray, true_idx: np.ndarray, threshold: float) -> float:
    """Fraction of examples whose true label lands in the conformal set ``{p >= 1 - threshold}``.

    Should be at least the target ``1 - alpha`` on held-out data when the threshold is fit
    correctly.
    """
    probs = np.atleast_2d(np.asarray(probs, dtype=float))
    true_idx = np.asarray(true_idx, dtype=int)
    cutoff = 1.0 - threshold
    in_set = probs[np.arange(len(probs)), true_idx] >= (cutoff - 1e-12)
    return float(np.mean(in_set))
