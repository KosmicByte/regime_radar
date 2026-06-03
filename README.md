# RegimeRadar

> Explainable market-regime detection via Koopman/DMD, EDMD, and HMM ensembles.

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-44%20passed-brightgreen)](#testing)

RegimeRadar classifies the current state of a market — *trending up/down, mean-reverting, high-vol chop, breakout, or low-vol grind* — and produces an auditable explanation for every label. Built for Indian equity markets with native support for indices and F&O-eligible stocks.

Every detection runs through a **point-in-time contract** that makes look-ahead bias structurally impossible, and is stamped with a model version, code version, and input hash so any past result is reproducible and auditable.

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

## Testing

```bash
uv run pytest                      # unit tests (~3 seconds)
uv run pytest -m slow              # add synthetic benchmark suite (~80 seconds)
```

Current status: **44/44 tests passing** — 14 math-layer, 14 ensemble on synthetic regimes, 14 point-in-time/provenance (R0), 2 benchmark gates.

## License

MIT — see [`LICENSE`](LICENSE).
