# Changelog

Entries are keyed to the detection-logic version (`regime_radar.version.MODEL_VERSION`), which is bumped when a change can alter the label for an identical input. This is independent of the package version in `pyproject.toml`.

Format based on [Keep a Changelog](https://keepachangelog.com/).

---

## [Unreleased] — branch `regime-radar-hardening`

Hardening track: reliability, reproducibility, and measurement before new modelling capability.

### R2 (part 2) — Benchmark confidence intervals and HMM-dampening A/B

No `MODEL_VERSION` change (measurement only). Completes R2.

**Added**
- `eval/metrics.py` — `BenchmarkResult.accuracy_ci()` (bootstrap over scenarios, as windows are autocorrelated), `stability_summary()` (whipsaw, dwell), `macro_f1()` (equal-weighted, plus per-regime F1).
- `eval/metrics.py` — `ab_dampening()` and `ABResult`: benchmark run on the same battery with the HMM-dampening ladder on and off; paired bootstrap on the per-scenario delta.
- `core/regime.py`, `eval/walk_forward.py`, `eval/metrics.py` — `raw_hmm_dampening` override, allowing the A/B to toggle the ladder without changing global settings.
- `cli/eval.py` — headline output includes 95% CI, macro-F1, and label stability. New `--ab-dampening` flag reports the A/B with a retire/keep verdict.
- `tests/test_r2_eval_rigor.py` — 5 tests.

**Finding**
- On the synthetic battery, the ladder alters HMM confidence (~15 of 51 windows on a switching series) but never the final ensemble label. Accuracy delta: 0.0%.
- Decision: `raw_hmm_dampening=True` retained. The ladder targets real-data disagreement cases not reproduced by the synthetic battery. Retirement decision deferred to real labelled data.

**Changed**
- `--ab-dampening` verdict reports `label_divergence` (fraction of windows where ON/OFF labels differ). When no labels change, the verdict is "untested" rather than "retire".

**Removed**
- Unused import `breakout_jump` from `eval/metrics.py`.

### R2 (part 1) — Economic validation, stability, and confidence intervals

No `MODEL_VERSION` change (measurement only; `detect_regime` output unaffected). Tests whether trading the regime has an edge.

**Added**
- `eval/economic.py` — walk-forward, point-in-time regime backtest with a fixed regime→position map and transaction costs. `evaluate_economic` reports Sharpe vs buy-and-hold, max drawdown, turnover, and a shuffled-regime permutation test (circular-shift null) isolating timing skill from position mix. Validation: significant edge on a switching series (p≈0.03); none on pure trend or chop.
- `eval/uncertainty.py` — Wilson interval and percentile bootstrap for headline metrics (e.g. "76% [72, 80]").
- `eval/stability.py` — whipsaw rate, mean/max dwell, switch count.
- `cli/backtest.py` — `regime backtest --symbol <ticker>`: edge, permutation p-value, and stability on a real symbol, with a not-financial-advice notice. Registered in `__main__.py`.
- `tests/test_r2_evaluation.py` — 9 tests, including null-beating on timing-sensitive series and `test_gap_in_close_does_not_poison_stats`.

**Fixed**
- `eval/economic.py` — a single missing close (observed on GAIL.NS via yfinance) produced NaN across all performance metrics. Closes are now forward-filled (point-in-time safe), returns are sanitised to finite values, and `_sharpe` guards non-finite results.

### MODEL_VERSION 1.1.1 — R1 fix: OOD scoring

**Fixed**
- `calibration/ood.py` — `ood_score` returned `NaN` when the feature vector contained a non-finite value (e.g. zero-variance windows on real data). Non-finite features are now imputed to the reference mean, the squared Mahalanobis distance is clamped non-negative, and the score falls back to 0.0 if still non-finite. Regression tests added.

### MODEL_VERSION 1.1.0 — R1: Calibration and uncertainty

**Added**
- `calibration/` package:
  - `artifact.py` — `CalibrationArtifact`: versioned JSON bundle (temperature, conformal threshold, OOD reference) recording `fit_source` and `model_version`.
  - `calibrator.py` — temperature scaling; preserves the argmax.
  - `conformal.py` — split-conformal prediction sets with coverage at `1 − alpha`; never empty.
  - `ood.py` — Mahalanobis OOD score in `[0, 1]` from `RegimeResult` features.
  - `fit.py` — harvests detections across the synthetic battery (calibration off) and fits the artifact; returns before/after diagnostics.
- `eval/calibration_metrics.py` — ECE, Brier, reliability curve, empirical coverage.
- `cli/calibrate.py` — `regime calibrate`: fit, persist, and report calibration quality. Registered in `__main__.py`.
- `tests/test_r1_calibration.py` — 9 tests, including label preservation and held-out conformal coverage.

**Changed**
- `models.py` — `RegimeResult` adds `calibrated`, `calibrator_version`, `prediction_set`, `coverage_level`, `ood_score`, `in_distribution` (all defaulted). `confidence` docstring corrected.
- `core/regime.py` — `detect_regime` adds a `calibrate` flag and applies the artifact via a cached loader. HMM dampening ladder gated behind `raw_hmm_dampening` (default on).
- `config.py` — adds `calibration_enabled`, `calibration_artifact`, `raw_hmm_dampening`.

**Notes**
- Additive: without an artifact, output is identical to R0. Label accuracy is unchanged.
- Synthetic harvest: ECE ~0.21 → ~0.10–0.14; conformal coverage at the 90% target.
- Artifact is fit on synthetic data (`fit_source="synthetic"`); coverage and OOD thresholds are indicative until re-fit on real data.
- Minor version bump: additive, label-preserving output changes.

### MODEL_VERSION 1.0.0 — R0: Point-in-time contract and reproducible inference

**Added**
- `core/contract.py` — `PointInTimeFrame`: truncates inputs to `as_of` at construction. Constructors `from_dataframe`, `from_arrays`; helpers `tail(n)`, `input_hash()`. Reserved `with_exogenous` hook for features with `known_at` semantics.
- `version.py` — `MODEL_VERSION` and `code_version()` (git SHA with `+dirty`, falling back to package version).
- `provenance.py` — `InferenceRecord` and append-only JSONL `write_record` / `read_records`. Opt-in via `provenance_enabled`; failures never break detection.
- `tests/test_r0_contract.py` — 14 tests, including `test_future_bars_do_not_leak`.

**Changed**
- `models.py` — `RegimeResult` adds `model_version`, `code_version`, `input_hash` (default `""`).
- `core/regime.py` — `detect_regime` resolves a `PointInTimeFrame` (`frame=` or built from `close=`); `as_of` and `input_hash` taken from the frame; results stamped with version fields; emits `InferenceRecord` when enabled. Legacy `close=`/`timestamps=` signature retained.
- `eval/walk_forward.py` — each window wrapped in a `PointInTimeFrame`, with an assertion that no bar beyond the window end is visible.
- `config.py` — adds `provenance_enabled`, `provenance_dir` (`REGIME_` prefix).

**Notes**
- No detection-logic change; synthetic accuracy unchanged (~75–78%, walk-forward, grouped).
- `MODEL_VERSION` 1.0.0 marks the first versioned, reproducible baseline.
- Backward compatible: 44/44 tests pass.

---

## Planned

- **R2.5 — Exogenous features**: macro and cross-asset features via `with_exogenous`, with `known_at` point-in-time discipline.

Capability work (HSMM, regime forecasting, BOCPD, learned weights, governance/API) continues on branch `regime-radar-forecasting`. HMM-dampening retirement is deferred until the A/B runs on real labelled data.
