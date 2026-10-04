# Architecture

## Design principles

1. **Explainability** — every regime label is auditable and carries the factors that produced it, with their contributions.
2. **Independent voters** — the ensemble combines three methods with orthogonal statistical assumptions, so their errors are weakly correlated.
3. **Pure math core** — `core/` has no I/O, globals, or side effects. Input: a numpy array. Output: a Pydantic model. All math is testable on synthetic data with known ground truth.
4. **Reproducible evaluation** — the evaluation harness ships with the package. The 70% accuracy floor is enforced as a regression test.

## Dependency graph

```
            ┌──────────────┐
            │  marketlake  │  (optional — falls back to yfinance)
            └──────┬───────┘
                   │
              data.py (single market-data entry point)
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    core/      core/        core/
   regime  ←  edmd        rules
       ↑    ←  hmm           ↑
       │    ←  observables   │
       │    ←  transition    │
       └──────────┬──────────┘
                  │
            models.py (Pydantic v2)
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
       cli/      eval/    viz/
```

`core/` depends only on numpy, scipy, and hmmlearn. `cli/`, `eval/`, and `viz/` depend on `core/` and `models.py`, not on each other.

## Ensemble

### 1. EDMD (Extended Dynamic Mode Decomposition)

- **Input**: linear-operator approximation of market dynamics in a lifted observable space.
- **Strength**: captures persistent (trending) and decaying (mean-reverting) modes in one framework.
- **Limitation**: eigenstructure is sensitive to observable choice and window length; spectrum can be unstable on short windows.
- **Vote**: maps `|λ₁|`, `arg(λ₁)`, and spectral gap to a label via fixed thresholds.

### 2. Gaussian HMM

- **Input**: latent discrete states governing `(return, log-vol)` emissions.
- **Strength**: calibrated posterior probability per state at each time step.
- **Limitation**: over-confident on directional labels in low-information regimes.
- **Vote**: labels each fitted state by its `(mean return, mean vol)` signature; outputs the label of the Viterbi-decoded current state.

### 3. Rule-based classifier

- **Input**: annualised drift, annualised vol, daily Sharpe, lag-5 variance ratio, AR(1) half-life of log-price.
- **Strength**: transparent, deterministic, no training. The AR(1) half-life is a price-level test not replicable by returns-based methods.
- **Limitation**: hand-tuned thresholds; may be miscalibrated on atypical markets.
- **Vote**: decision tree over vol, Sharpe, variance ratio, and half-life thresholds, with priority ordering for edge cases (e.g. half-life override for OU mean reversion).

### Voting

Default weights: `{edmd: 0.40, hmm: 0.30, rule: 0.30}`.

Each voter assigns `weight × confidence` to its label and distributes `weight × (1 − confidence)` uniformly across the remaining labels. The final label is the argmax of the normalised distribution.

HMM confidence modulation, applied when HMM votes a directional label:

| Rule-classifier condition | HMM confidence multiplier |
|---|---|
| AR(1) half-life < 15 bars | 0.3 |
| Variance ratio indicates strong reversion | 0.4 |
| Sharpe near zero | 0.6 |

## Transition-risk model

Measures deviation of the current Koopman spectrum from a baseline of recent spectra. Three signals, each mapped through a sigmoid to approximately `[0, 1]`, are combined by weighted average:

1. **Eigenvalue drift** — L2 distance between the current top-k eigenvalues (by magnitude) and the mean over `baseline_windows` historical fits.
2. **Spectral-gap collapse** — reduction in `|λ₁| − |λ₂|` relative to the baseline mean. Indicates loss of a dominant mode.
3. **Vol acceleration** — z-score of `(short_vol − long_vol)` against its 60-bar history.

Scores above `transition_threshold` (default 0.6) are flagged.

## Data layer

`data.py` exposes a single function: `load(symbol, interval, start, end)`. MarketLake is used if importable; otherwise yfinance. No other module calls yfinance or Upstox directly.

Schema (bhavcopy-aligned): `symbol, interval, ts, open, high, low, close, volume`. Timestamps are tz-aware (IST).

## Out of scope

- **Backtesting** — RegimeRadar outputs labels and risk scores only.
- **Forecasting** — a regime is a present-tense classification. Forward claims require a logged evaluation transcript.
- **Risk management** — labels are decision inputs, not decisions.
