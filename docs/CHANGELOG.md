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

- **R2 — Evaluation rigor:** confidence intervals on the headline metric, label-stability
  (whipsaw) metrics, an economic-value check, and the A/B that decides whether to retire the
  HMM dampening ladder now that calibration sits on top of it.
- **R2.5 — Exogenous features:** macro / cross-asset conditioning features via the
  `with_exogenous` hook, with strict `known_at` point-in-time discipline.

Capability work (HSMM, regime forecasting, BOCPD, learned weights, governance/API) continues
on the `regime-radar-forecasting` branch.
