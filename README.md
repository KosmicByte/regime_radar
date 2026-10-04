# RegimeRadar

Explainable market-regime detection using an EDMD (Koopman), HMM, and rule-based ensemble.

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-69%20passed-brightgreen)](#testing)

Classifies market state as `TRENDING_UP`, `TRENDING_DOWN`, `MEAN_REVERTING`, `HIGH_VOL_CHOP`, `BREAKOUT`, or `LOW_VOL_GRIND`, with an auditable explanation for each label. Targets Indian equity markets: NSE indices and F&O-eligible stocks.

- **Point-in-time contract** — look-ahead bias is structurally excluded.
- **Reproducibility** — every result is stamped with model version, code version, and input hash.
- **Uncertainty** — with a calibration artifact: calibrated confidence, conformal prediction set, and out-of-distribution score.

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
| [`docs/architecture.md`](docs/architecture.md) | System design, point-in-time contract, calibration, ensemble |
| [`docs/methods.md`](docs/methods.md) | Koopman, EDMD, HMM, variance ratio, AR(1) half-life |
| [`docs/evaluation.md`](docs/evaluation.md) | Benchmark methodology and results |
| [`docs/features.md`](docs/features.md) | Roadmap |
| [`CHANGELOG.md`](CHANGELOG.md) | Change history keyed to `MODEL_VERSION` |

## Accuracy

75–78% on the synthetic ground-truth battery under walk-forward evaluation with grouped scoring. Reproducible via `regime eval`; enforced in CI with a 70% floor. No real-market accuracy figure is published — see [`docs/evaluation.md`](docs/evaluation.md).

## Point-in-time integrity and reproducibility

Detection consumes a `PointInTimeFrame`, which truncates all inputs to an `as_of` instant and exposes no data beyond it. Appending future bars does not change a past-dated result.

| `RegimeResult` field | Content |
|---|---|
| `model_version` | Detection-logic semver; bumped when outputs can change |
| `code_version` | Git SHA or package version |
| `input_hash` | Deterministic hash of inputs |

With `REGIME_PROVENANCE_ENABLED=true`, each detection appends an immutable JSONL inference record to `artifacts/provenance/`.

## Calibration and uncertainty

`regime calibrate` fits a calibration artifact. Detections then report:

| Output | Description |
|---|---|
| Calibrated confidence | Temperature-scaled probability. Preserves the argmax; labels are unchanged. |
| Conformal prediction set | Label set covering the true regime at the target rate (default 90%). |
| Out-of-distribution score | `[0, 1]`; high values indicate a market state unlike the fit data. |

The current artifact is fit on the synthetic battery and marked as such. Re-running `regime calibrate` on real data refreshes it without code changes. See [`docs/cli.md`](docs/cli.md#regime-calibrate).

## Economic validation

`regime backtest` tests whether a symbol's regime sequence has tradeable timing. It walks the detector over history (point-in-time), applies a fixed regime→position map with transaction costs, and compares against a shuffled-regime null.

```bash
uv run regime backtest --symbol GAIL.NS
```

| Output | Description |
|---|---|
| Permutation p-value | Fraction of circularly shifted position sequences with Sharpe ≥ the real strategy. Low values indicate timing skill beyond position mix. |
| Performance | Sharpe vs buy-and-hold, max drawdown |
| Stability | Whipsaw rate, mean dwell |

Output is validation evidence, not a trading signal or recommendation, and not financial advice. See [`docs/cli.md`](docs/cli.md#regime-backtest).

## Testing

```bash
uv run pytest            # unit tests (~3 s)
uv run pytest -m slow    # includes synthetic benchmark (~80 s)
```

| Suite | Tests |
|---|---|
| Math layer | 14 |
| Ensemble on synthetic regimes | 14 |
| Point-in-time and provenance (R0) | 14 |
| Calibration, conformal, OOD (R1) | 10 |
| Economic, stability, uncertainty (R2) | 10 |
| Benchmark CIs and dampening A/B (R2) | 5 |
| Benchmark gates | 2 |
| **Total** | **69** |

## License

MIT — see [`LICENSE`](LICENSE).
