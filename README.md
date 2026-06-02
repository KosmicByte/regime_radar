# RegimeRadar

> Explainable market-regime detection via Koopman/DMD, EDMD, and HMM ensembles.

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-30%20passed-brightgreen)](#testing)

RegimeRadar classifies the current state of a market — *trending up/down, mean-reverting, high-vol chop, breakout, or low-vol grind* — and produces an auditable explanation for every label. Built for Indian equity markets with native support for indices and F&O-eligible stocks.

## Quickstart

```bash
git clone https://github.com/<your-username>/regime_radar.git
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

## Accuracy

RegimeRadar achieves **75–78% accuracy** on the synthetic ground-truth battery under walk-forward evaluation with grouped scoring. Reproducible via `regime eval` and enforced as a CI floor at 70%. A real-markets accuracy figure is deliberately not published — see [`docs/evaluation.md`](docs/evaluation.md) for the framing.

## Testing

```bash
uv run pytest                      # unit tests (~3 seconds)
uv run pytest -m slow              # add synthetic benchmark suite (~80 seconds)
```

Current status: **30/30 tests passing** — 14 math-layer, 14 ensemble on synthetic regimes, 2 benchmark gates.

## License

MIT — see [`LICENSE`](LICENSE).
