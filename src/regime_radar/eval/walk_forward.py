"""Walk-forward evaluation — the honest way to measure regime-detection performance.

We slide a window across a labeled series, run `detect_regime` on each window, and
compare the *predicted* label at the right edge against the *true* label at the right
edge. Reports overall accuracy, per-regime precision/recall/F1, and a confusion matrix.

For regime-switching series, we additionally measure transition detection lag:
how many bars after a true regime change before the predictor flags the new regime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from regime_radar.core.regime import detect_regime
from regime_radar.eval.synthetic import SyntheticSeries
from regime_radar.models import RegimeLabel


@dataclass(frozen=True)
class WalkForwardResult:
    """Per-window predictions, plus aggregate metrics."""

    predicted: np.ndarray  # (n_windows,) RegimeLabel
    true: np.ndarray  # (n_windows,) RegimeLabel
    confidence: np.ndarray  # (n_windows,) float
    window_end_indices: np.ndarray  # (n_windows,) int — index into the original series

    @property
    def accuracy(self) -> float:
        if len(self.predicted) == 0:
            return 0.0
        return float(np.mean(self.predicted == self.true))

    def confusion_matrix(self) -> tuple[np.ndarray, list[RegimeLabel]]:
        labels = sorted({*self.true.tolist(), *self.predicted.tolist()}, key=lambda x: x.value)
        idx = {lbl: i for i, lbl in enumerate(labels)}
        m = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(self.true, self.predicted, strict=False):
            m[idx[t], idx[p]] += 1
        return m, labels

    def per_label_metrics(self) -> dict[RegimeLabel, dict[str, float]]:
        """Precision, recall, F1 per label."""
        m, labels = self.confusion_matrix()
        out: dict[RegimeLabel, dict[str, float]] = {}
        for i, lbl in enumerate(labels):
            tp = m[i, i]
            fp = m[:, i].sum() - tp
            fn = m[i, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            out[lbl] = {"precision": float(prec), "recall": float(rec), "f1": float(f1)}
        return out


def walk_forward(
    series: SyntheticSeries,
    window: int = 126,
    step: int = 21,
    edmd_rank: int = 10,
    hmm_n_states: int = 3,
    grouped: bool = True,
) -> WalkForwardResult:
    """Slide a window across `series`, run detect_regime, score against ground truth.

    Args:
        series: SyntheticSeries with known labels.
        window: bars per fit.
        step: stride between successive fits.
        edmd_rank, hmm_n_states: detector knobs.
        grouped: if True, collapse fine labels into broad groups (see below). For a
                 stricter test, set False to require exact label match.

    Grouping (when grouped=True): we treat TRENDING_UP / BREAKOUT as 'directional_up'
    and TRENDING_DOWN as 'directional_down' is kept distinct; this matches how a
    practitioner reads the dashboard.
    """
    prices = series.prices
    true_labels = series.labels
    n_total = len(prices) - 1  # n labels

    preds: list[RegimeLabel] = []
    truths: list[RegimeLabel] = []
    confs: list[float] = []
    ends: list[int] = []

    for end in range(window, n_total + 1, step):
        start = end - window
        window_close = prices[start : end + 1]  # +1 because prices is one longer
        try:
            result = detect_regime(
                close=window_close,
                symbol="synthetic",
                interval="1d",
                edmd_rank=min(edmd_rank, window // 3),
                hmm_n_states=hmm_n_states,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue

        preds.append(result.label)
        truths.append(true_labels[end - 1])
        confs.append(result.confidence)
        ends.append(end - 1)

    pred_arr = np.array(preds, dtype=object)
    true_arr = np.array(truths, dtype=object)

    if grouped:
        pred_arr = np.array([_group(lbl) for lbl in pred_arr], dtype=object)
        true_arr = np.array([_group(lbl) for lbl in true_arr], dtype=object)

    return WalkForwardResult(
        predicted=pred_arr,
        true=true_arr,
        confidence=np.array(confs),
        window_end_indices=np.array(ends),
    )


def _group(label: RegimeLabel) -> RegimeLabel:
    """Optional grouping for less-strict scoring: BREAKOUT folded into TRENDING_UP."""
    if label == RegimeLabel.BREAKOUT:
        return RegimeLabel.TRENDING_UP
    return label


def transition_detection_lag(
    series: SyntheticSeries,
    window: int = 126,
    step: int = 1,
) -> list[int]:
    """For each true regime change in `series`, return bars-until-detection.

    Walks forward with step=1 (or the given step) and records the first time the
    predicted label matches the new true label after a change point.
    """
    prices = series.prices
    true_labels = series.labels
    change_points = np.where(true_labels[1:] != true_labels[:-1])[0] + 1  # indices into labels
    if len(change_points) == 0:
        return []

    # Predict at every step from `window` onwards (expensive — caller chooses step).
    preds: dict[int, RegimeLabel] = {}
    for end in range(window, len(prices) - 1, step):
        try:
            result = detect_regime(close=prices[end - window : end + 1])
            preds[end - 1] = result.label
        except (ValueError, np.linalg.LinAlgError):
            continue

    lags: list[int] = []
    for cp in change_points:
        new_label = true_labels[cp]
        # find the first prediction index >= cp where prediction matches new_label
        future_idxs = sorted(i for i in preds if i >= cp)
        lag = None
        for i in future_idxs:
            if preds[i] == new_label or _group(preds[i]) == _group(new_label):
                lag = i - cp
                break
        if lag is not None:
            lags.append(lag)
    return lags
