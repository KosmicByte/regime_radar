"""Aggregate benchmark across a battery of synthetic regimes.

Run `benchmark()` to get the headline 'accuracy on synthetic ground truth' number that
backs (or refutes) the 70% target. Re-run after every change to detection logic — this
is the regression test that matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from regime_radar.eval.stability import StabilityReport, stability
from regime_radar.eval.synthetic import (
    RegimeSpec,
    SyntheticSeries,
    gbm_trending,
    high_vol_chop,
    low_vol_grind,
    ou_mean_reverting,
    regime_switching,
)
from regime_radar.eval.walk_forward import WalkForwardResult, walk_forward
from regime_radar.models import RegimeLabel


@dataclass
class BenchmarkResult:
    """Per-scenario results + overall accuracy."""

    per_scenario: dict[str, WalkForwardResult] = field(default_factory=dict)

    @property
    def overall_accuracy(self) -> float:
        all_pred = np.concatenate([r.predicted for r in self.per_scenario.values()])
        all_true = np.concatenate([r.true for r in self.per_scenario.values()])
        if len(all_pred) == 0:
            return 0.0
        return float(np.mean(all_pred == all_true))

    @property
    def per_scenario_accuracy(self) -> dict[str, float]:
        return {name: r.accuracy for name, r in self.per_scenario.items()}

    def confusion(self) -> tuple[np.ndarray, list[RegimeLabel]]:
        """Overall confusion matrix across all scenarios."""
        all_pred = np.concatenate([r.predicted for r in self.per_scenario.values()])
        all_true = np.concatenate([r.true for r in self.per_scenario.values()])
        labels = sorted({*all_pred.tolist(), *all_true.tolist()}, key=lambda x: x.value)
        idx = {lbl: i for i, lbl in enumerate(labels)}
        m = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(all_true, all_pred, strict=False):
            m[idx[t], idx[p]] += 1
        return m, labels

    def accuracy_ci(
        self, *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
    ) -> tuple[float, float, float]:
        """Bootstrap CI on overall accuracy, resampling *scenarios* (not windows).

        Windows within a scenario are autocorrelated, so resampling them understates the
        interval. Resampling whole scenarios (the independent unit) gives an honest spread:
        the headline becomes "76% [70, 82]" rather than a bare point estimate.
        """
        scenarios = list(self.per_scenario.values())
        if len(scenarios) < 2:
            acc = self.overall_accuracy
            return (acc, acc, acc)
        rng = np.random.default_rng(seed)
        k = len(scenarios)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.integers(0, k, k)
            pred = np.concatenate([scenarios[i].predicted for i in pick])
            true = np.concatenate([scenarios[i].true for i in pick])
            boots[b] = np.mean(pred == true) if len(pred) else 0.0
        lo = float(np.percentile(boots, 100 * alpha / 2))
        hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
        return (self.overall_accuracy, lo, hi)

    def stability_summary(self) -> StabilityReport:
        """Aggregate label-stability across every scenario's predicted sequence.

        Whipsaw and switch counts are summed over scenarios; dwell is averaged. Tells you how
        steady the detector's output is — a jumpy detector is expensive to trade regardless of
        accuracy.
        """
        total_n = 0
        total_switches = 0
        dwell_means: list[float] = []
        max_dwell = 0
        for r in self.per_scenario.values():
            rep = stability(list(r.predicted))
            total_n += rep.n
            total_switches += rep.n_switches
            dwell_means.append(rep.mean_dwell)
            max_dwell = max(max_dwell, rep.max_dwell)
        whipsaw = total_switches / max(total_n - len(self.per_scenario), 1)
        return StabilityReport(
            n=total_n,
            whipsaw_rate=float(whipsaw),
            n_switches=int(total_switches),
            mean_dwell=float(np.mean(dwell_means)) if dwell_means else 0.0,
            max_dwell=int(max_dwell),
        )

    def macro_f1(self) -> tuple[float, dict[str, float]]:
        """Macro-averaged F1 across regimes, plus per-regime F1, from the pooled confusion.

        Accuracy can flatter a detector that nails the common regime and fails the rare one.
        Macro-F1 weights every regime equally, exposing that.
        """
        m, labels = self.confusion()
        per: dict[str, float] = {}
        for i, lbl in enumerate(labels):
            tp = m[i, i]
            fp = m[:, i].sum() - tp
            fn = m[i, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per[lbl.value] = float(f1)
        macro = float(np.mean(list(per.values()))) if per else 0.0
        return macro, per


def default_battery(n_per_scenario: int = 1008, seeds_per_scenario: int = 5) -> list[SyntheticSeries]:
    """The default mix of synthetic scenarios used by the benchmark.

    Parameters are chosen so each regime is statistically distinguishable from a random
    walk over the analysis window. Weaker parameters would not be testing the detector
    — they would be testing whether noise dominates signal, which it always does for
    weak regimes regardless of method. The whole point of regime detection is to
    classify regimes that ARE detectable; we measure that ability here.

    Default: 1008 bars (~4 years) per scenario × 5 seeds = 30 series, ~1000 walk-forward
    windows total. The longer series give ~36 walk-forward windows each (at window=252,
    step=21), so per-seed noise averages out. With 5 seeds per scenario, individual
    unlucky realisations do not dominate the headline number.
    """
    out: list[SyntheticSeries] = []
    for s in range(seeds_per_scenario):
        # Strong trends: Sharpe ~2 annual = clearly distinguishable from random walk
        out.append(gbm_trending(n=n_per_scenario, direction="up", mu_annual=0.30, sigma_annual=0.15, seed=s))
        out.append(gbm_trending(n=n_per_scenario, direction="down", mu_annual=0.30, sigma_annual=0.15, seed=s))
        # Strong mean reversion: theta=15 -> ~17-day half-life
        out.append(ou_mean_reverting(n=n_per_scenario, theta=15.0, sigma=0.25, seed=s))
        # High-vol regime: 50% annualised
        out.append(high_vol_chop(n=n_per_scenario, sigma_annual=0.50, seed=s))
        # Low-vol grind: 6% annualised with modest drift
        out.append(low_vol_grind(n=n_per_scenario, sigma_annual=0.06, mu_annual=0.06, seed=s))
        # One stitched regime-switching scenario per seed (intentionally hard)
        out.append(
            regime_switching(
                [
                    RegimeSpec(generator="gbm_up", n=n_per_scenario // 3, seed=s),
                    RegimeSpec(generator="chop", n=n_per_scenario // 3, seed=s + 100),
                    RegimeSpec(generator="ou", n=n_per_scenario // 3, seed=s + 200),
                ]
            )
        )
    return out


def benchmark(
    series: list[SyntheticSeries] | None = None,
    window: int = 126,
    step: int = 21,
    edmd_rank: int = 10,
    hmm_n_states: int = 3,
    grouped: bool = True,
    raw_hmm_dampening: bool | None = None,
) -> BenchmarkResult:
    """Run walk-forward across the battery and aggregate."""
    series = series or default_battery()
    result = BenchmarkResult()
    for i, s in enumerate(series):
        try:
            wf = walk_forward(
                s,
                window=window,
                step=step,
                edmd_rank=edmd_rank,
                hmm_n_states=hmm_n_states,
                grouped=grouped,
                raw_hmm_dampening=raw_hmm_dampening,
            )
            # Key the dict to keep multiple seeds distinguishable
            result.per_scenario[f"{s.name}_{i}"] = wf
        except Exception:  # noqa: BLE001
            continue
    return result


@dataclass
class ABResult:
    """A/B comparison of the detector with the HMM-dampening ladder on vs off."""

    on: BenchmarkResult
    off: BenchmarkResult

    def summary(self, *, margin: float = 0.01, n_boot: int = 2000, seed: int = 0) -> dict:
        """Paired comparison of ON vs OFF on the shared battery.

        Because both arms run on the *same* scenarios, the right test is paired: bootstrap the
        per-scenario accuracy delta (OFF − ON). The ladder can be retired when OFF is
        non-inferior — its delta CI lower bound is no worse than ``-margin`` (default 1pp).
        """
        keys = [k for k in self.on.per_scenario if k in self.off.per_scenario]
        on_acc = np.array([self.on.per_scenario[k].accuracy for k in keys])
        off_acc = np.array([self.off.per_scenario[k].accuracy for k in keys])
        deltas = off_acc - on_acc
        point_delta = float(deltas.mean()) if len(deltas) else 0.0

        # How often do the two configs actually disagree on the label? If this is ~0 the ladder
        # never changed an outcome on this battery, so the A/B cannot speak to retiring it —
        # a zero accuracy delta then means "untested", not "safe to remove".
        mismatch = 0
        total = 0
        for k in keys:
            op = self.on.per_scenario[k].predicted
            fp = self.off.per_scenario[k].predicted
            m = min(len(op), len(fp))
            if m:
                mismatch += int(np.sum(op[:m] != fp[:m]))
                total += m
        label_divergence = float(mismatch / total) if total else 0.0

        rng = np.random.default_rng(seed)
        k = len(deltas)
        if k > 1:
            boots = np.array([deltas[rng.integers(0, k, k)].mean() for _ in range(n_boot)])
            d_lo, d_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        else:
            d_lo = d_hi = point_delta

        on_overall, on_lo, on_hi = self.on.accuracy_ci(seed=seed)
        off_overall, off_lo, off_hi = self.off.accuracy_ci(seed=seed)
        on_macro, _ = self.on.macro_f1()
        off_macro, _ = self.off.macro_f1()
        return {
            "on_accuracy": on_overall,
            "on_ci": (on_lo, on_hi),
            "off_accuracy": off_overall,
            "off_ci": (off_lo, off_hi),
            "delta_mean": point_delta,
            "delta_ci": (d_lo, d_hi),
            "on_macro_f1": on_macro,
            "off_macro_f1": off_macro,
            "can_retire": bool(d_lo >= -margin),
            "margin": margin,
            "label_divergence": label_divergence,
            # The ladder is only "exercised" if it changes some labels; otherwise the A/B is
            # blind to it and a zero delta means "untested", not "safe to retire".
            "exercised": bool(label_divergence > 1e-6),
        }


def ab_dampening(
    series: list[SyntheticSeries] | None = None,
    window: int = 126,
    step: int = 21,
    edmd_rank: int = 10,
    hmm_n_states: int = 3,
    grouped: bool = True,
) -> ABResult:
    """Run the benchmark twice on the SAME battery: HMM dampening ladder on vs off.

    This is the evidence for whether the hand-tuned ``hmm_conf *=`` heuristics still earn their
    place now that principled calibration sits on top. The same series are used for both arms so
    the comparison is apples-to-apples.
    """
    battery = series or default_battery()
    on = benchmark(battery, window, step, edmd_rank, hmm_n_states, grouped, raw_hmm_dampening=True)
    off = benchmark(battery, window, step, edmd_rank, hmm_n_states, grouped, raw_hmm_dampening=False)
    return ABResult(on=on, off=off)
