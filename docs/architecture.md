# Architecture

## Design principles

1. **Explainability** — every regime label is auditable and carries the factors that produced it, with their contributions.
2. **Independent voters** — the ensemble combines three methods with orthogonal statistical assumptions, so their errors are weakly correlated.
3. **Pure math core** — `core/` has no I/O, globals, or side effects. Input: a numpy array. Output: a Pydantic model. All math is testable on synthetic data with known ground truth.
4. **Reproducible evaluation** — the evaluation harness ships with the package. The 70% accuracy floor is enforced as a regression test.
5. **Point-in-time by construction** — detection cannot access future data, and every result is reproducible. Enforced structurally (R0).

## Point-in-time contract and reproducibility (R0)

Every detection consumes a `PointInTimeFrame` (`core/contract.py`). The frame wraps a price series and an `as_of` instant and truncates to `as_of` at construction; no method returns a bar dated after it. `detect_regime` accepts a frame directly or builds one from a raw `close` array.

The walk-forward harness wraps each window in a frame with `as_of` set to the window's last bar. Appending future bars leaves a past-dated result byte-identical (`test_future_bars_do_not_leak`).

Fields stamped on every `RegimeResult`:

| Field | Content |
|---|---|
| `model_version` | Detection-logic semver (`MODEL_VERSION` in `version.py`); bumped on any change that can alter output. Independent of the package version. |
| `code_version` | Git SHA (with `+dirty` marker), or package version as fallback |
| `input_hash` | 16-char SHA-256 of visible inputs; deterministic |

With `provenance_enabled`, each detection appends an immutable `InferenceRecord` (one JSON line) to `provenance_dir`. Writing is stdlib-only, append-only, and isolated so it cannot fail a detection.

`with_exogenous` is reserved (currently raises `NotImplementedError`) for macro and cross-asset features with `known_at` publication-lag semantics.

## Calibration and uncertainty (R1)

Driven by a single JSON artifact (`calibration/artifact.py`) produced by `regime calibrate`:

1. **Calibrated confidence** — temperature scaling (`calibration/calibrator.py`): one scalar fitted to minimise NLL on a harvested calibration set. `softmax(log p / T)` preserves the argmax, so the label is never changed and accuracy cannot regress.
2. **Conformal prediction sets** — split conformal (`calibration/conformal.py`) with nonconformity `1 − p(true)`. Distribution-free coverage at `1 − alpha`. Never empty; the argmax is always included.
3. **Out-of-distribution score** — Mahalanobis distance over features derived from the `RegimeResult` (`calibration/ood.py`), mapped through the chi-square CDF to `[0, 1]`. Higher values indicate a market state unlike the fit data.

`detect_regime` applies the artifact when present and `calibration_enabled` is set. `calibrate=False` returns raw output (used during calibrator fitting and A/B tests). Without an artifact, output is identical to the uncalibrated detector.

The artifact records `fit_source`, currently `"synthetic"`. Re-fitting on real data requires one `regime calibrate` run.

The HMM confidence-dampening ladder is gated behind `raw_hmm_dampening` (default on). The R2 A/B found no label change on synthetic data; the ladder is retained pending real-data evaluation.

## Dependency graph

```
            ┌──────────────┐
            │  marketlake  │  (optional — falls back to yfinance)
            └──────┬───────┘
                   │
              data.py (single market-data entry point)
                   │
            core/contract.py (point-in-time frame — truncates to as_of)
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
            version.py · provenance.py (R0: versioning + inference log)
            calibration/ (R1: temperature · conformal · OOD)
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
       cli/      eval/    viz/
```

- `core/` depends only on numpy, scipy, and hmmlearn. R0 additions use stdlib `hashlib` and `json`; R1 uses scipy.
- `cli/`, `eval/`, and `viz/` depend on `core/` and `models.py`, not on each other.
- `core/regime.py` imports the `calibration/` leaf modules to apply an artifact. The fit path (`calibration/fit.py`) is invoked only by `regime calibrate`, so no import cycle exists.

## Ensemble

### 1. EDMD (Extended Dynamic Mode Decomposition)

- **Input**: linear-operator approximation of market dynamics in a lifted observable space.
- **Strength**: captures persistent (trending) and decaying (mean-reverting) modes in one framework.
- **Limitation**: eigenstructure is sensitive to observable choice and window length; spectrum can be unstable on short windows.
- **Vote**: maps `|λ₁|`, `arg(λ₁)`, and spectral gap to a label via fixed thresholds.

### 2. Gaussian HMM

- **Input**: latent discrete states governing `(return, log-vol)` emissions.
- **Strength**: posterior probability per state at each time step.
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

HMM confidence dampening (`raw_hmm_dampening`), applied when HMM votes a directional label:

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

- **Forecasting** — a regime is a present-tense classification. Forward claims require a logged evaluation transcript.
- **Trading signals** — `regime backtest` is an economic validation of regime timing, not a strategy or signal generator.
- **Risk management** — labels are decision inputs, not decisions.
