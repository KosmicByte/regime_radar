# Quickstart

Installation, configuration, and worked examples for RegimeRadar.

## Requirements

- Python 3.12 or later
- [`uv`](https://docs.astral.sh/uv/) (recommended) or pip
- Internet access for the yfinance fallback, or a local MarketLake installation

## Installation

### With uv (recommended)

```bash
git clone https://github.com/<your-username>/regime_radar.git
cd regime_radar
uv sync                       # installs runtime + dev dependencies
uv run regime --help          # CLI is exposed as `regime`
```

### With pip

```bash
git clone https://github.com/<your-username>/regime_radar.git
cd regime_radar
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
regime --help
```

## Configuration

RegimeRadar reads configuration from environment variables, all namespaced `REGIME_*`. A template is provided at `.env.example`:

```bash
cp .env.example .env
# edit as needed
```

Key settings:

| Variable | Default | Purpose |
|---|---|---|
| `REGIME_DEFAULT_SYMBOL` | `^NSEI` | Default ticker for `regime detect` |
| `REGIME_DEFAULT_INTERVAL` | `1d` | Bar interval (`1d`, `1h`, `5m`, `15m`) |
| `REGIME_WINDOW` | `126` | Analysis window in bars (~6 months daily) |
| `REGIME_ARTIFACTS_DIR` | `./artifacts` | Where plots and reports are written |
| `MARKETLAKE_DATA_DIR` | — | If set and importable, MarketLake is used in preference to yfinance |

See [`config.py`](../src/regime_radar/config.py) for the full list.

## Worked example — single symbol

```bash
uv run regime detect --symbol RELIANCE.NS
```

Output:

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

## Worked example — watchlist scan

The shipped `scan` command supports 35 named watchlists across NSE sectors and group affiliations:

```bash
# Tight bellwether scan — fast, no rate limits
uv run regime scan

# By sector
uv run regime scan --watchlist banks-private
uv run regime scan --watchlist it
uv run regime scan --watchlist pharma

# By corporate group
uv run regime scan --watchlist tata
uv run regime scan --watchlist adani

# Broad market — ~70 names; add delay to avoid Yahoo rate-limiting
uv run regime scan --watchlist broad --delay-ms 500

# List all available watchlists
uv run regime scan --list

# Ad-hoc
uv run regime scan --symbol HDFCBANK.NS --symbol ICICIBANK.NS
```

## Worked example — library use

```python
from regime_radar.data import load
from regime_radar.core.regime import detect_regime
from regime_radar.core.transition import score_transition_risk

# Load 2 years of daily data — tries MarketLake, falls back to yfinance
df = load(symbol="^NSEI", interval="1d", lookback_days=720)
close = df["close"].to_numpy()

# Detect current regime on the last 130 bars
result = detect_regime(close[-130:], symbol="^NSEI")
print(f"{result.label.value} — confidence {result.confidence:.0%}")
for reason in result.reasons:
    print(f"  [{reason.factor}] {reason.note}")

# Score transition risk over the full window
risk = score_transition_risk(close, symbol="^NSEI", window=126)
print(f"transition risk: {risk.risk_score:.0%} — {risk.note}")
```

See [`examples/quickstart.py`](../examples/quickstart.py) and [`examples/synthetic_demo.py`](../examples/synthetic_demo.py) for runnable scripts.

## Worked example — running the benchmark

```bash
uv run regime eval
```

This executes the full synthetic walk-forward battery (~80 seconds) and prints per-scenario accuracy plus an overall confusion matrix. The same battery runs in CI as a regression test — the threshold is 70%.

Pass `--plot` to write a confusion-matrix PNG to `artifacts/`.

## Ticker conventions

| Market | Suffix | Example |
|---|---|---|
| NSE equities | `.NS` | `RELIANCE.NS`, `TCS.NS` |
| BSE equities | `.BO` | `RELIANCE.BO` |
| NSE indices | `^` prefix | `^NSEI` (NIFTY 50), `^NSEBANK` (Bank Nifty) |
| US equities | none | `AAPL`, `MSFT` |

Stick to F&O-eligible names where possible — they have the cleanest Yahoo Finance data. Less-liquid small caps may produce gappy bars that the detector misreads as transitions.

## Troubleshooting

**`KeyError: 'ts'` or empty data errors** — Yahoo's API occasionally rate-limits aggressive scans. Wait 30–60 seconds and retry, or use `--delay-ms 500` on large scans. For persistent issues, switch to a more liquid symbol (e.g. `RELIANCE.NS` instead of an obscure mid-cap).

**HMM convergence warnings during `regime eval`** — These are non-fatal and arise from hmmlearn's EM on noisy synthetic data. They do not affect the accuracy score.

**Stale data on intraday intervals** — Yahoo's intraday data has limited history. For `5m` or `15m` intervals, the maximum lookback is roughly 60 days regardless of `--lookback-days`.

## Next steps

- Read [`docs/architecture.md`](architecture.md) to understand the three-voter ensemble.
- Read [`docs/methods.md`](methods.md) for the mathematical foundations.
- Read [`docs/evaluation.md`](evaluation.md) for the benchmark methodology and accuracy framing.
- Read [`docs/features.md`](features.md) for the roadmap and competitive analysis.
