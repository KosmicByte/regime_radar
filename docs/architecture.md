# Architecture

## Goals

1. **Explainable** — every regime label must be auditable. The user can ask "why?" and get a list of factors with their contributions.
2. **Independent voters** — ensemble three methods whose statistical assumptions are orthogonal, so their errors are uncorrelated and aggregate accuracy beats any single method.
3. **Pure math core** — the `core/` package has no I/O, no globals, no side effects. Inject a numpy array, get back a Pydantic result. This makes the math fully testable on synthetic data with known ground truth.
4. **Honest measurement** — ship the evaluation harness so accuracy claims are reproducible. The 70% headline is enforced as a regression test, not a marketing number.
5. **Point-in-time by construction** — detection cannot see the future and every result is reproducible. This is structural, not a convention to remember (see below).

## Point-in-time contract & reproducibility (R0)

Every detection consumes a `PointInTimeFrame` (`core/contract.py`). The frame wraps a price series plus an `as_of` instant and **truncates to that instant at construction** — there is no method that returns a bar dated after `as_of`. `detect_regime` accepts a frame directly, or builds one internally from a raw `close` array, so both call paths get the same guarantee. The walk-forward harness wraps each window in a frame whose `as_of` is the window's last bar, turning "no look-ahead" from a hope into an enforced invariant: appending future bars to a series leaves a past-dated decision byte-identical (this is a test, `test_future_bars_do_not_leak`).

Reproducibility rides on three fields stamped onto every `RegimeResult`:

- `model_version` — the detection-logic semver from `version.py` (`MODEL_VERSION`), bumped on any change that can alter outputs. Distinct from the package version.
- `code_version` — git SHA (with a `+dirty` marker) or, failing that, the package version.
- `input_hash` — a stable 16-char SHA-256 of the visible inputs; identical inputs always produce an identical hash.

When `provenance_enabled` is set, each detection also appends an immutable `InferenceRecord` (one JSON line) under `provenance_dir`. Records are stdlib-only and append-only, and writing is wrapped so provenance can never break a detection.

The contract reserves a `with_exogenous` hook (currently a documented `NotImplementedError`) for macro / cross-asset features that will carry their own `known_at` publication lag — so those features can be added later without a breaking change.

## Calibration & uncertainty (R1)

A point label with a raw probability is not enough to act on safely. R1 adds three things, all driven by a single fitted artifact (`calibration/artifact.py`, plain JSON), produced by `regime calibrate`:

1. **Calibrated confidence** — temperature scaling (`calibration/calibrator.py`) corrects the sharpness of the soft-vote distribution with one learned scalar, fit to minimise NLL on a harvested calibration set. Because `softmax(log p / T)` preserves the per-row argmax, **calibration never changes the label** — it only makes `confidence` honest. This is why R1 cannot regress accuracy.
2. **Conformal prediction sets** — split conformal (`calibration/conformal.py`) with nonconformity `1 - p(true)` yields a label set with a distribution-free coverage guarantee at the target `1 - alpha`. The set surfaces plausible alternatives and is never empty (the argmax is always retained).
3. **Out-of-distribution score** — a Mahalanobis distance in a small feature space derived entirely from the `RegimeResult` (`calibration/ood.py`), mapped through the chi-square CDF to [0, 1]. High means the current market state is unlike the fit data, so the label should be trusted less.

`detect_regime` applies the artifact when one is present and `calibration_enabled` is set; with `calibrate=False` (used while *fitting* the calibrator, and for A/B tests) it returns raw output. When no artifact exists, behaviour is identical to the uncalibrated detector — calibration is purely additive.

The artifact records its `fit_source`. Today that is `"synthetic"`: the calibrator is fit on the synthetic battery because real NSE data is not yet wired in, and the artifact says so explicitly. Re-fitting on real data later is a single `regime calibrate` run.

R1 also gates the hand-tuned HMM-confidence dampening ladder in `regime.py` behind `raw_hmm_dampening` (default on). Calibration now sits on top of it; the R2 evaluation work will A/B the ladder off against calibration to decide whether to retire it.

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

`core/` has zero external coupling beyond numpy/scipy/hmmlearn (the R0 additions use only stdlib `hashlib`/`json`; the R1 calibration layer uses scipy, already a dependency). `cli/`, `eval/`, and `viz/` all depend on `core/` and `models.py` but never on each other. `core/regime.py` consumes the `calibration/` leaf modules to apply an artifact, but the fit path (`calibration/fit.py`) lives outside the hot path and is only invoked by `regime calibrate`, so there is no import cycle.

## The ensemble

Three voters, each operating on different statistical assumptions:

### 1. EDMD (Extended Dynamic Mode Decomposition)
- **What it sees**: linear-operator approximation of market dynamics in a lifted observable space.
- **Strength**: captures persistent modes (trending) and decay modes (mean reversion) in one framework.
- **Weakness**: the eigenstructure is sensitive to which observables are chosen and to the window length. Spectrum can be unstable on short windows.
- **Vote**: maps `|λ₁|`, `arg(λ₁)`, and spectral gap to a regime label via documented thresholds.

### 2. Gaussian HMM
- **What it sees**: latent discrete states governing (return, log-vol) emissions.
- **Strength**: provides a calibrated posterior probability for each state at each time step.
- **Weakness**: tends to be over-confident on directional labels in low-info regimes (random walks with apparent local drifts).
- **Vote**: labels each fitted state by its `(mean return, mean vol)` signature and outputs the Viterbi-decoded current state's label.

### 3. Rule-based vol + trend + reversion
- **What it sees**: annualised drift, annualised vol, daily Sharpe, lag-5 variance ratio, AR(1) half-life of log-price.
- **Strength**: fully transparent, fast, no learning. Uses orthogonal statistics — the AR(1) half-life is a price-level test that returns-based methods cannot replicate.
- **Weakness**: hand-tuned thresholds; can be slightly miscalibrated on highly atypical markets.
- **Vote**: a decision tree of vol/Sharpe/VR/half-life thresholds with priorities chosen to handle edge cases (e.g., AR(1) half-life override for OU mean reversion).

### Voting

The default weights are `{edmd: 0.40, hmm: 0.30, rule: 0.30}`. Each voter contributes `weight × confidence` to its chosen label; the remaining `weight × (1 - confidence)` is spread uniformly across other labels (mild smoothing so we never zero out alternatives). The final label is the argmax of the normalised distribution.

Two confidence-modulation rules tune HMM's over-confidence:

- When the rule classifier identifies strong mean reversion (half-life < 15 bars) and HMM votes directional, HMM's confidence is multiplied by 0.3.
- When the rule's variance ratio strongly indicates reversion and HMM votes directional, HMM's confidence is multiplied by 0.4.
- When the rule's Sharpe is near zero and HMM votes directional, HMM's confidence is multiplied by 0.6.

## The transition-risk model

A separate path that asks: given the current Koopman spectrum, how different is it from recent baseline spectra?

Three signals, each mapped through a sigmoid to roughly `[0, 1]`, then weighted-averaged:

1. **Eigenvalue drift**: L2 distance between the current top-k eigenvalues (by magnitude) and the average of `baseline_windows` historical fits.
2. **Spectral-gap collapse**: drop in `|λ₁| - |λ₂|` versus the baseline mean. A collapsing gap means no clear dominant mode — regime in flux.
3. **Vol acceleration**: z-score of `(short_vol - long_vol)` versus its 60-bar history. Vol regime shifts often lead structural shifts.

Risk above `transition_threshold` (default 0.6) is flagged for the user.

## Data layer

`data.py` exposes one function: `load(symbol, interval, start, end)`. It tries to import MarketLake first; if absent, falls back to yfinance. Downstream code never calls yfinance/Upstox directly — when MarketLake exists, this becomes a thin pass-through.

The schema is bhavcopy-aligned: `symbol, interval, ts, open, high, low, close, volume`. Times are tz-aware (IST).

## What this is NOT

- Not a backtester. RegimeRadar produces labels and risk scores; backtesting is a separate job.
- Not a forecaster. "Regime" is a present-tense classification, not a forward prediction. Any forward claim must be backed by a logged evaluation transcript.
- Not a substitute for risk management. A label is an input to your decision process, not the decision.
