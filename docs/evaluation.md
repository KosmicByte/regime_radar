# Evaluation

How to honestly measure regime-detection performance.

## The headline number — and why it's qualified

**RegimeRadar achieves ~76% accuracy on the default synthetic battery under grouped scoring.**

That sentence is loaded. Each clause matters:

- **~76%** — not a single number. The benchmark uses 5 seeds per scenario across 6 scenario families, so individual runs may produce 73-78%. The 70% target is set as a regression test floor, not a precision claim.
- **synthetic battery** — generated processes with **known per-bar labels**. This is not the same as real-market accuracy; see "Real markets" below.
- **grouped scoring** — `BREAKOUT` is folded into `TRENDING_UP` for the headline. A high-vol upward jump is functionally the same regime as a strong uptrend from a portfolio-management perspective. Strict scoring (no grouping) lands around 5-7 percentage points lower.

## What's in the synthetic battery

Six families, 5 seeds each, 1008 bars each (~4 years daily), totalling ~1110 walk-forward windows:

| Family | Generator | Parameters | Expected label |
|---|---|---|---|
| `gbm_up` | GBM with positive drift | μ=0.30, σ=0.15 | TRENDING_UP |
| `gbm_down` | GBM with negative drift | μ=-0.30, σ=0.15 | TRENDING_DOWN |
| `ou_meanrev` | Ornstein-Uhlenbeck on log-price | θ=15, σ=0.25 | MEAN_REVERTING |
| `high_vol_chop` | High-vol zero-drift RW | σ=0.50 | HIGH_VOL_CHOP |
| `low_vol_grind` | Low-vol low-drift GBM | μ=0.06, σ=0.06 | LOW_VOL_GRIND |
| `regime_switching` | Stitched: gbm_up + chop + ou | mixed | varies per bar |

The parameters are chosen so each regime is **statistically distinguishable** from a random walk over the analysis window. Weaker parameters would not be testing the detector — they would be testing whether signal exceeds noise, which it doesn't statistically for weak regimes regardless of method.

## Walk-forward protocol

For each series:

1. Start at bar `window = 252`.
2. Run `detect_regime` on bars `[end - window, end]`.
3. Compare the predicted label at the right edge against the **true** label at that bar.
4. Advance `end` by `step = 21` (one month) and repeat.

This gives ~36 predictions per 1008-bar series, ~1000+ total windows in the battery.

## Per-family accuracy (representative run)

```
gbm_up               66-78%
gbm_down             77%
high_vol_chop        96%
low_vol_grind        83%
ou_meanrev           80%
regime_switching     53%
```

Notes:

- `high_vol_chop` is the easiest (annualised vol >> historical norm is unmissable).
- `regime_switching` is intentionally hard. The stitched series transitions between three regimes; a 252-bar window straddles a regime boundary about a third of the time, and ground truth at the right edge changes faster than our window can update.
- `gbm_up` is weaker than `gbm_down` because positive drift with σ=0.15 produces local windows with Sharpe near zero more often than negative-drift windows (due to compounding asymmetry of geometric Brownian motion).

## Real markets — what to measure

There is **no canonical accuracy number for real markets**, because ground truth requires a labelling rule, and any such rule is itself a modelling choice. Reasonable rules include:

- **Forward-vol percentile** — label each bar by the percentile of its forward-21-day realized vol relative to history; "HIGH_VOL_CHOP" = top 20%.
- **Drawdown depth** — label by current drawdown from rolling-252-day peak.
- **Trend strength** — label by Sharpe of forward 63 bars.

The package exposes `walk_forward()` so you can plug in any labelling rule and measure honestly on your own data. The shipped synthetic harness is a controlled environment; the real-world story is whatever you measure with your rule.

What we **deliberately don't do**: publish a single real-markets accuracy number. Doing so would require defending a particular labelling rule as canonical, and any such defence would be unsupportable.

## Running the benchmark

```bash
# Headline test (~80 seconds)
uv run pytest tests/test_benchmark_accuracy.py -v -s

# Or via the CLI:
uv run regime eval

# Strict scoring (no BREAKOUT → TRENDING_UP folding)
uv run regime eval --strict

# Custom config
uv run regime eval --window 252 --step 21 --edmd-rank 10 --hmm-states 3
```

The `eval` CLI also writes a confusion-matrix PNG to `artifacts/` when given `--plot`.

## Reproducibility

Synthetic generators use `numpy.random.default_rng(seed)`. The default seeds are `0..4` per scenario. Re-running with the same seeds reproduces the headline number to within a few decimal places (HMM EM has minor non-determinism from internal initialisation).

## What a regression looks like

If the benchmark accuracy drops below 70%, the test fails. Common causes:

- A change to observable defaults that affects EDMD spectrum sensitivity.
- A change to HMM covariance handling that affects state labelling.
- A change to rule thresholds without verifying on the full battery.

Investigation order:

1. Run `pytest tests/test_regime.py` — does any single-scenario unit test fail? That points at a specific case.
2. Run `pytest tests/test_observables.py tests/test_edmd.py` — has the math layer regressed?
3. Run `regime eval --strict` to see if the issue is in grouped scoring vs. underlying logic.
4. Look at the confusion matrix — where is the error concentrated?

## Honesty in headlines

When you publish results, prefer:

> "RegimeRadar attains 75-78% accuracy on a synthetic battery of GBM, OU, high-vol-chop, low-vol-grind, and regime-switching scenarios under walk-forward evaluation with grouped scoring."

over:

> "RegimeRadar achieves 76% accuracy."

The first is a claim; the second is marketing. The package supports the first; it does not support the second.
