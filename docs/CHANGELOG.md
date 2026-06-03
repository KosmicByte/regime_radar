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

- **R1 — Calibration & uncertainty:** make `RegimeResult.confidence` genuinely calibrated;
  add conformal prediction sets and an out-of-distribution score.
- **R2 — Evaluation rigor:** confidence intervals on the headline metric, label-stability
  (whipsaw) metrics, and an economic-value check.
- **R2.5 — Exogenous features:** macro / cross-asset conditioning features via the
  `with_exogenous` hook, with strict `known_at` point-in-time discipline.

Capability work (HSMM, regime forecasting, BOCPD, learned weights, governance/API) continues
on the `regime-radar-forecasting` branch.
