# RegimeRadar

> Explainable market-regime detection via Koopman/DMD, EDMD, and HMM ensembles.

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-54%20passed-brightgreen)](#testing)

RegimeRadar classifies the current state of a market — *trending up/down, mean-reverting, high-vol chop, breakout, or low-vol grind* — and produces an auditable explanation for every label. Built for Indian equity markets with native support for indices and F&O-eligible stocks.

Every detection runs through a **point-in-time contract** that makes look-ahead bias structurally impossible, is stamped with a model version, code version, and input hash so any past result is reproducible and auditable, and — once a calibration artifact is fit — reports **calibrated confidence**, a **conformal prediction set**, and an **out-of-distribution score** so you know when to trust the label and when the market looks unlike anything the detector has seen.

## Quickstart

```bash
git clone https://github.com/KosmicByte/regime_radar.git
cd regime_radar
uv sync
uv run regime detect --symbol ^NSEI
```

Full installation, configuration, and worked examples: [`docs/quickstart.md`](docs/quickstart.md).

## Documentation

| Document | Purpose |
|---|---|
| [`docs/quickstart.md`](docs/quickstart.md) | Installation, configuration, worked examples |
| [`docs/cli.md`](docs/cli.md) | CLI reference — every command, flag, and watchlist |
| [`docs/architecture.md`](docs/architecture.md) | System design, dependency graph, ensemble logic |
| [`docs/methods.md`](docs/methods.md) | Mathematical foundations — Koopman, EDMD, HMM, VR, AR(1) |
| [`docs/evaluation.md`](docs/evaluation.md) | Benchmark methodology and accuracy claims |
| [`docs/features.md`](docs/features.md) | Roadmap and competitive feature analysis |
| [`CHANGELOG.md`](CHANGELOG.md) | Versioned change history, keyed to the detection-logic `MODEL_VERSION` |

## Accuracy

RegimeRadar achieves **75–78% accuracy** on the synthetic ground-truth battery under walk-forward evaluation with grouped scoring. Reproducible via `regime eval` and enforced as a CI floor at 70%. A real-markets accuracy figure is deliberately not published — see [`docs/evaluation.md`](docs/evaluation.md) for the framing.

## Reproducibility & point-in-time integrity

Detection consumes a `PointInTimeFrame` — a contract that truncates every input to an `as_of` instant and exposes no API to see beyond it. Appending future bars to a series cannot change a past-dated decision, so backtests and the walk-forward harness are honest by construction.

Each `RegimeResult` carries `model_version` (the detection-logic semver, bumped whenever outputs can change), `code_version` (git SHA or package version), and `input_hash` (a stable hash of the inputs). With `REGIME_PROVENANCE_ENABLED=true`, every detection also appends an immutable JSONL inference record under `artifacts/provenance/`, turning "why did the regime flip on that date?" into a lookup rather than a guess.

## Calibrated confidence & uncertainty

Run `regime calibrate` once to fit a calibration artifact. After that, every detection reports:

- **Calibrated confidence** — temperature-scaled so "70%" means right ~70% of the time. Because temperature scaling preserves the argmax, calibration never changes the label, only its honesty.
- **A conformal prediction set** — a set of labels guaranteed to cover the true regime at the target rate (default 90%), so you see the plausible alternatives, not just the point pick.
- **An out-of-distribution score** — a 0–1 novelty signal; when it's high, the current market looks unlike the fit data and the label should be trusted less.

The calibrator is currently fit on the synthetic battery and is marked as such; re-running `regime calibrate` on real data later refreshes it with no code change. See [`docs/cli.md`](docs/cli.md#regime-calibrate).

## Testing

```bash
uv run pytest                      # unit tests (~3 seconds)
uv run pytest -m slow              # add synthetic benchmark suite (~80 seconds)
```

Current status: **54/54 tests passing** — 14 math-layer, 14 ensemble on synthetic regimes, 14 point-in-time/provenance (R0), 10 calibration/conformal/OOD (R1), 2 benchmark gates.

## License

MIT — see [`LICENSE`](LICENSE).
