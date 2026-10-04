# Quickstart

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or pip
- Internet access (yfinance fallback) or a local MarketLake installation

## Installation

### uv

```bash
git clone https://github.com/KosmicByte/regime_radar.git
cd regime_radar
uv sync
uv run regime --help
```

### pip

```bash
git clone https://github.com/KosmicByte/regime_radar.git
cd regime_radar
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
regime --help
```

## Configuration

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
|---|---|---|
| `REGIME_DEFAULT_SYMBOL` | `^NSEI` | Default ticker |
| `REGIME_DEFAULT_INTERVAL` | `1d` | Bar interval (`1d`, `1h`, `5m`, `15m`) |
| `REGIME_WINDOW` | `126` | Analysis window in bars |
| `REGIME_ARTIFACTS_DIR` | `./artifacts` | Output directory for plots and reports |
| `MARKETLAKE_DATA_DIR` | — | Use MarketLake instead of yfinance when set and importable |

Full list: [`config.py`](../src/regime_radar/config.py).

## Single symbol

```bash
uv run regime detect --symbol RELIANCE.NS
```

```
╭────────── Regime ──────────╮
│ RELIANCE.NS @ 2026-05-29   │
│ TRENDING_UP                │
│ conf 73% | ann vol 22.1%   │
│ trend Sharpe +0.21         │
╰────────────────────────────╯
                Probability distribution
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Regime             ┃ Probability ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ trending_up        │         73% │
│ low_vol_grind      │         12% │
│ mean_reverting     │          9% │
└────────────────────┴─────────────┘
                       Why this label
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Factor          ┃ Note                                           ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ rule_vol_trend  │ Positive Sharpe +0.21, drift +25%              │
│ edmd_spectrum   │ |λ₁|=0.987 ≈ 1 — persistent mode               │
│ hmm_state       │ HMM Viterbi state with posterior 0.68          │
└─────────────────┴────────────────────────────────────────────────┘
```

## Watchlist scan

```bash
uv run regime scan                                             # default watchlist
uv run regime scan --watchlist banks-private                   # sector
uv run regime scan --watchlist tata                            # group
uv run regime scan --watchlist broad --delay-ms 500            # ~70 symbols
uv run regime scan --list                                      # all watchlists
uv run regime scan --symbol HDFCBANK.NS --symbol ICICIBANK.NS  # ad-hoc
```

## Library

```python
from regime_radar.data import load
from regime_radar.core.regime import detect_regime
from regime_radar.core.transition import score_transition_risk

df = load(symbol="^NSEI", interval="1d", lookback_days=720)
close = df["close"].to_numpy()

result = detect_regime(close[-130:], symbol="^NSEI")
print(f"{result.label.value} — confidence {result.confidence:.0%}")
for reason in result.reasons:
    print(f"  [{reason.factor}] {reason.note}")

risk = score_transition_risk(close, symbol="^NSEI", window=126)
print(f"transition risk: {risk.risk_score:.0%} — {risk.note}")
```

Runnable scripts: [`examples/quickstart.py`](../examples/quickstart.py), [`examples/synthetic_demo.py`](../examples/synthetic_demo.py).

## Benchmark

```bash
uv run regime eval          # ~80 s; CI regression threshold: 70%
uv run regime eval --plot   # confusion-matrix PNG to artifacts/
```

## Ticker conventions

| Market | Format | Example |
|---|---|---|
| NSE equities | `.NS` suffix | `RELIANCE.NS`, `TCS.NS` |
| BSE equities | `.BO` suffix | `RELIANCE.BO` |
| NSE indices | `^` prefix | `^NSEI`, `^NSEBANK` |
| US equities | none | `AAPL`, `MSFT` |

F&O-eligible symbols are recommended. Illiquid small caps may produce gappy bars that register as false transitions.

## Troubleshooting

| Issue | Resolution |
|---|---|
| `KeyError: 'ts'` or empty data | Yahoo rate limit. Retry after 30–60 s or use `--delay-ms 500` |
| HMM convergence warnings in `eval` | Non-fatal; no effect on accuracy score |
| Limited intraday history | `5m` / `15m` capped at ~60 days regardless of `--lookback` |

## Further reading

- [`architecture.md`](architecture.md) — ensemble design, point-in-time contract, calibration
- [`methods.md`](methods.md) — mathematical foundations
- [`evaluation.md`](evaluation.md) — benchmark methodology
- [`features.md`](features.md) — roadmap
