# Evaluation

## Headline result

**~76% accuracy on the default synthetic battery under grouped scoring.**

| Qualifier | Detail |
|---|---|
| Range | Individual runs yield 73–78% (6 scenario families × 5 seeds). 70% is the regression floor. |
| Synthetic | Ground truth is per-bar labels from generative processes. Not a real-market figure. |
| Grouped scoring | `BREAKOUT` is folded into `TRENDING_UP`. Strict scoring is ~5–7 percentage points lower. |

## Synthetic battery

Six families, 5 seeds each, 1008 bars per series (~4 years daily), ~1110 walk-forward windows in total.

| Family | Generator | Parameters | Label |
|---|---|---|---|
| `gbm_up` | GBM, positive drift | μ=0.30, σ=0.15 | TRENDING_UP |
| `gbm_down` | GBM, negative drift | μ=−0.30, σ=0.15 | TRENDING_DOWN |
| `ou_meanrev` | Ornstein-Uhlenbeck on log-price | θ=15, σ=0.25 | MEAN_REVERTING |
| `high_vol_chop` | Zero-drift random walk | σ=0.50 | HIGH_VOL_CHOP |
| `low_vol_grind` | Low-drift GBM | μ=0.06, σ=0.06 | LOW_VOL_GRIND |
| `regime_switching` | gbm_up + chop + ou, stitched | mixed | per bar |

Parameters are set so each regime is statistically distinguishable from a random walk over the analysis window.

## Walk-forward protocol

1. Start at bar `window = 252`.
2. Run `detect_regime` on bars `[end − window, end]`.
3. Compare the predicted label with the true label at bar `end`.
4. Advance `end` by `step = 21` and repeat.

Yields ~36 predictions per series.

## Per-family accuracy (representative run)

```
gbm_up               66-78%
gbm_down             77%
high_vol_chop        96%
low_vol_grind        83%
ou_meanrev           80%
regime_switching     53%
```

- `high_vol_chop`: highest accuracy; vol far exceeds the historical norm.
- `regime_switching`: lowest by design. A 252-bar window straddles a regime boundary in roughly one third of windows.
- `gbm_up` vs `gbm_down`: at σ=0.15, positive-drift GBM produces near-zero-Sharpe local windows more often than negative-drift GBM, due to compounding asymmetry.

## Real-market evaluation

No real-market accuracy figure is published. Real-market ground truth requires a labelling rule, and any such rule is itself a modelling choice. Candidate rules:

- **Forward-vol percentile** — percentile of forward 21-day realised vol; top 20% = `HIGH_VOL_CHOP`.
- **Drawdown depth** — drawdown from rolling 252-day peak.
- **Trend strength** — Sharpe of forward 63 bars.

`walk_forward()` accepts any labelling rule and any dataset.

## Running the benchmark

```bash
# Regression test (~80 s)
uv run pytest tests/test_benchmark_accuracy.py -v -s

# CLI
uv run regime eval
uv run regime eval --strict
uv run regime eval --window 252 --step 21 --edmd-rank 10 --hmm-states 3
uv run regime eval --plot      # confusion-matrix PNG to artifacts/
```

## Reproducibility

Generators use `numpy.random.default_rng(seed)` with seeds `0..4` per scenario. Results reproduce to within a few decimal places; residual variance comes from HMM EM initialisation.

## Regression diagnosis

The benchmark test fails below 70%. Common causes:

- Changes to observable defaults affecting EDMD spectrum sensitivity.
- Changes to HMM covariance handling affecting state labelling.
- Changes to rule thresholds not validated on the full battery.

Investigation order:

1. `pytest tests/test_regime.py` — isolate a failing scenario.
2. `pytest tests/test_observables.py tests/test_edmd.py` — check the math layer.
3. `regime eval --strict` — separate grouped-scoring effects from core logic.
4. Inspect the confusion matrix for concentrated errors.

## Reporting results

Report accuracy with its qualifiers, e.g.:

> RegimeRadar attains 75–78% accuracy on a synthetic battery of GBM, OU, high-vol-chop, low-vol-grind, and regime-switching scenarios under walk-forward evaluation with grouped scoring.
