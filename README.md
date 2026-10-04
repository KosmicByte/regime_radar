# RegimeRadar

Explainable market-regime detection using an EDMD (Koopman), HMM, and rule-based ensemble.

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-30%20passed-brightgreen)](#testing)

Classifies market state as `TRENDING_UP`, `TRENDING_DOWN`, `MEAN_REVERTING`, `HIGH_VOL_CHOP`, `BREAKOUT`, or `LOW_VOL_GRIND`, with an auditable explanation for each label. Targets Indian equity markets: NSE indices and F&O-eligible stocks.

## Quickstart

```bash
git clone https://github.com/KosmicByte/regime_radar.git
cd regime_radar
uv sync
uv run regime detect --symbol ^NSEI
```

Installation, configuration, and examples: [`docs/quickstart.md`](docs/quickstart.md).

## Documentation

| Document | Contents |
|---|---|
| [`docs/quickstart.md`](docs/quickstart.md) | Installation, configuration, examples |
| [`docs/cli.md`](docs/cli.md) | Commands, flags, watchlists |
| [`docs/architecture.md`](docs/architecture.md) | System design, dependency graph, ensemble |
| [`docs/methods.md`](docs/methods.md) | Koopman, EDMD, HMM, variance ratio, AR(1) half-life |
| [`docs/evaluation.md`](docs/evaluation.md) | Benchmark methodology and results |
| [`docs/features.md`](docs/features.md) | Roadmap |

## Accuracy

75–78% on the synthetic ground-truth battery under walk-forward evaluation with grouped scoring. Reproducible via `regime eval`; enforced in CI with a 70% floor. No real-market accuracy figure is published — see [`docs/evaluation.md`](docs/evaluation.md).

## Testing

```bash
uv run pytest            # unit tests (~3 s)
uv run pytest -m slow    # includes synthetic benchmark (~80 s)
```

30 tests: 14 math layer, 14 ensemble on synthetic regimes, 2 benchmark gates.

## License

MIT — see [`LICENSE`](LICENSE).
