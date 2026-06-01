"""The headline accuracy test — locks in our 70% target on synthetic ground truth.

If this test fails, accuracy has regressed below the target and we have a real problem.
Run with `pytest tests/test_benchmark_accuracy.py -v -s` for live output.
"""

from __future__ import annotations

import warnings

import pytest

from regime_radar.eval.metrics import benchmark

# hmmlearn emits convergence warnings on noisy walk-forward windows; not informative.
warnings.filterwarnings("ignore", category=UserWarning)


@pytest.mark.slow
def test_benchmark_overall_accuracy_meets_target() -> None:
    """The headline number. >= 70% accuracy on the default synthetic battery (grouped).

    Run config:
      - Default battery: 30 series (6 scenarios x 5 seeds), 1008 bars each
      - Walk-forward: window=252, step=21 (~36 predictions per series, ~1000 total)
      - Grouped scoring: BREAKOUT folded into TRENDING_UP

    Expected ~70-72% on a freshly-fit model. Catastrophic regressions (e.g. < 55%) point
    to a bug, not a tuning issue — investigate the math layer first.
    """
    result = benchmark(window=252, step=21, edmd_rank=10, hmm_n_states=3, grouped=True)
    acc = result.overall_accuracy
    print(f"\n  Overall accuracy: {acc:.1%}")
    for name, scenario_acc in sorted(result.per_scenario_accuracy.items()):
        print(f"    {name}: {scenario_acc:.1%}")
    assert acc >= 0.70, f"Synthetic accuracy regressed to {acc:.1%}, below 70% target."


@pytest.mark.slow
def test_each_scenario_family_above_floor() -> None:
    """No regime FAMILY should be catastrophically bad (< 30%).

    Aggregates across seeds because individual seeds can be unlucky. This is the right
    granularity for a floor test — we want to catch detectors that simply can't see a
    regime type, not one that struggles with one bad realisation.
    """
    from collections import defaultdict

    result = benchmark(window=252, step=21, grouped=True)
    family_correct: dict[str, int] = defaultdict(int)
    family_total: dict[str, int] = defaultdict(int)
    for name, wf in result.per_scenario.items():
        family = "_".join(name.split("_")[:-1])
        family_correct[family] += int((wf.predicted == wf.true).sum())
        family_total[family] += len(wf.predicted)

    family_acc = {f: family_correct[f] / family_total[f] for f in family_total if family_total[f] > 0}
    bad = {f: a for f, a in family_acc.items() if a < 0.30}
    assert not bad, f"Regime families below 30% floor: {bad}. All accuracies: {family_acc}"
