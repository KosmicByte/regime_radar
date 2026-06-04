# Changelog

All notable changes to RegimeRadar are recorded here. Entries are keyed to the
**detection-logic version** (`regime_radar.version.MODEL_VERSION`), which is bumped whenever
a change can alter the label produced for an identical input. This is deliberately separate
from the package/distribution version in `pyproject.toml`.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/).

---

## [Unreleased] — branch `regime-radar-hardening`

The enterprise-hardening track. Goal: make RegimeRadar trustworthy enough to be load-bearing
for the wider ecosystem before adding new modelling capability.

### R2 (part 1) — Economic validation, stability & confidence intervals

No `MODEL_VERSION` change: R2 is pure measurement and adds no detection-logic change — it does
not alter `detect_regime` output. It answers a different question from accuracy: *does trading
the regime have an edge?*

**Added**
- `eval/economic.py` — walk-forward, point-in-time regime backtest with a transparent
  regime→position map and transaction costs, plus `evaluate_economic`: Sharpe vs buy-and-hold,
  max drawdown, turnover, and a **shuffled-regime permutation test** (circular-shift null) that
  isolates timing skill from the base position mix. Validated: detects significant edge on a
  switching series (p≈0.03), correctly finds none on a pure trend (≈ buy-hold) or on chop
  (stays flat).
- `eval/uncertainty.py` — Wilson interval for proportions and a percentile bootstrap, so
  headline metrics ship as "76% [72, 80]" rather than a bare point estimate.
- `eval/stability.py` — whipsaw rate, mean/max dwell, switch count: the steadiness of the
  label sequence, since every flip costs money under the cost model.
- `cli/backtest.py` — `regime backtest --symbol GAIL.NS`: runs the economic validation on a
  real symbol and reports edge, the permutation p-value, and stability — decision *evidence*,
  explicitly not a recommendation or signal, with a not-financial-advice caveat. Registered in
  `__main__.py`.
- `tests/test_r2_evaluation.py` — 9 tests, anchored by the "beats null when timing matters /
  doesn't when it doesn't" pair.
- `tests/test_r2_evaluation.py` also asserts a missing close (real-feed gap) cannot NaN the
  summary (`test_gap_in_close_does_not_poison_stats`).

**Fixed**
- `eval/economic.py` — a single NaN/missing close (observed on GAIL.NS via yfinance) poisoned
  `mean()`/`std()` and turned the whole performance table NaN. Closes are now forward-filled
  (point-in-time safe) before any computation, returns are sanitised to finite, and `_sharpe`
  guards non-finite results.

**Still to come in R2**
- Fold the bootstrap CIs and stability metrics into the `regime eval` benchmark output.
- The `raw_hmm_dampening` on/off A/B against calibration, to decide whether to retire the
  hand-tuned HMM ladder.

### MODEL_VERSION 1.1.1 — R1 fix: robust OOD scoring

**Fixed**
- `calibration/ood.py` — `ood_score` could return `NaN` when a detection's feature vector
  contained a non-finite value (observed on a real-market symbol whose window had a flat,
  zero-variance stretch, making a Sharpe-/gap-style feature divide by zero). The score now
  imputes any non-finite feature to the reference mean (neutral contribution), clamps the
  squared Mahalanobis distance non-negative, and falls back to 0.0 if anything is still
  non-finite — so a detection can never emit a `NaN` OOD score. Added regression tests.

### MODEL_VERSION 1.1.0 — R1: calibration & uncertainty

**Added**
- `calibration/` package:
  - `artifact.py` — `CalibrationArtifact`, a versioned JSON bundle (temperature, conformal
    threshold, OOD reference) that records its `fit_source` and `model_version`.
  - `calibrator.py` — temperature scaling. `softmax(log p / T)` preserves the argmax, so
    calibration corrects confidence without ever changing the label.
  - `conformal.py` — split-conformal prediction sets with a distribution-free coverage
    guarantee at `1 - alpha`; never returns an empty set.
  - `ood.py` — Mahalanobis out-of-distribution score in [0, 1], from a feature vector derived
    entirely from the `RegimeResult` (no recomputation).
  - `fit.py` — harvests detections across the synthetic battery (with calibration off) and
    fits the full artifact; returns before/after diagnostics.
- `eval/calibration_metrics.py` — ECE, Brier, reliability curve, empirical coverage.
- `cli/calibrate.py` — `regime calibrate`: fit + persist the artifact and print calibration
  quality. Registered in `__main__.py`.
- `tests/test_r1_calibration.py` — 9 tests, including label-preservation and held-out
  conformal coverage.

**Changed**
- `models.py` — `RegimeResult` gains `calibrated`, `calibrator_version`, `prediction_set`,
  `coverage_level`, `ood_score`, `in_distribution` (all defaulted so behaviour is unchanged
  when no artifact is present). The `confidence` docstring is now accurate.
- `core/regime.py` — `detect_regime` gains a `calibrate` flag; applies the artifact (when
  present and enabled) via a cached loader; the hand-tuned HMM dampening ladder is now gated
  behind the `raw_hmm_dampening` setting (default on).
- `config.py` — adds `calibration_enabled`, `calibration_artifact`, `raw_hmm_dampening`.

**Notes**
- Calibration is additive: with no artifact, output is identical to R0. Temperature scaling
  preserves the argmax, so **label accuracy is unchanged** — only confidence quality improves.
  On the synthetic harvest, ECE dropped ~0.21 → ~0.10–0.14 and conformal coverage hit the 90%
  target.
- The artifact is **fit on synthetic data** and marked `fit_source="synthetic"`; coverage and
  OOD thresholds are indicative until re-fit on real data (a one-command refresh).
- `MODEL_VERSION` → 1.1.0: a minor bump for additive, label-preserving output changes.

### MODEL_VERSION 1.0.0 — R0: point-in-time contract & reproducible inference

**Added**
- `core/contract.py` — `PointInTimeFrame`, the sanctioned input to detection. Truncates
  every input to an `as_of` instant at construction; no API can return a bar dated after it.
  Constructors `from_dataframe` (consumes `data.load` output) and `from_arrays`. Helpers
  `tail(n)` (point-in-time-safe windowing) and `input_hash()` (stable SHA-256 of inputs).
  Reserves a `with_exogenous` hook for future macro / cross-asset features with `known_at`
  publication-lag semantics.
- `version.py` — `MODEL_VERSION` (detection-logic semver) and `code_version()` (git SHA with
  `+dirty` marker, falling back to package version).
- `provenance.py` — `InferenceRecord` plus append-only JSONL `write_record` / `read_records`.
  Opt-in via `provenance_enabled`; best-effort so it can never break a detection.
- `tests/test_r0_contract.py` — 14 tests including `test_future_bars_do_not_leak`, the
  guarantee that appending future bars cannot change a past-dated decision.

**Changed**
- `models.py` — `RegimeResult` gains `model_version`, `code_version`, and `input_hash`
  (all default to `""` so existing construction and deserialisation keep working).
- `core/regime.py` — `detect_regime` now resolves a `PointInTimeFrame` (accepts `frame=`
  directly, or builds one from `close=`); `as_of` and `input_hash` come from the frame;
  results are stamped with the version fields; emits an `InferenceRecord` when enabled.
  The legacy `close=`/`timestamps=` signature still works.
- `eval/walk_forward.py` — each window is wrapped in a `PointInTimeFrame` and passed via
  `frame=`, with an assertion that no bar beyond the window end is visible.
- `config.py` — adds `provenance_enabled` and `provenance_dir` settings (env-prefixed
  `REGIME_`).

**Notes**
- No change to detection logic, so synthetic-benchmark accuracy is unchanged (≈75–78%,
  walk-forward, grouped). Despite that, `MODEL_VERSION` starts at 1.0.0 to mark the first
  versioned, reproducible baseline; future detection changes increment from here.
- Backward compatible: all prior call sites and the full pre-existing test suite pass
  unchanged (44/44 total with the new R0 tests).

---

## Upcoming (planned on this branch)

- **R2 (remainder):** fold bootstrap CIs + stability into `regime eval`; run the
  `raw_hmm_dampening` A/B against calibration to decide whether to retire the HMM ladder.
- **R2.5 — Exogenous features:** macro / cross-asset conditioning features via the
  `with_exogenous` hook, with strict `known_at` point-in-time discipline.

Capability work (HSMM, regime forecasting, BOCPD, learned weights, governance/API) continues
on the `regime-radar-forecasting` branch.
