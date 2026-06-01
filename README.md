# RegimeRadar

> Explainable market-regime detection via Koopman/DMD, EDMD, and HMM ensembles.

RegimeRadar tells you what regime a market is in — **trending up/down, mean-reverting, high-vol chop, breakout, or low-vol grind** — and *shows you why*. Every label ships with the eigenvalues, HMM posteriors, and trend/reversion statistics that produced it. You can audit the call instead of trusting a black box.

## Highlights

- **Ensemble of three independent voters** — EDMD (Koopman spectrum), Gaussian HMM (latent states), and a transparent rule-based classifier that uses variance-ratio and AR(1) half-life tests. Independent assumptions → uncorrelated errors → ensemble beats best constituent.
- **75-78% accuracy on the synthetic ground truth battery** under walk-forward evaluation with grouped scoring. The benchmark is reproducible — re-run with `regime eval`.
- **Honest about real markets** — the package will not publish a single real-markets accuracy number, because any such number requires defending a canonical labelling rule. The harness lets you plug in your own labelling rule and measure honestly on your own data.
- **Transition-risk scoring** — measures eigenvalue drift, spectral-gap collapse, and vol acceleration against a rolling baseline, giving an early-warning score (not just the current label).
- **Explanation-first design** — `regime explain` prints the full eigenvalue table, contribution of each voter, and a plain-English narrative for the label.
- **Indian markets friendly** — defaults to `^NSEI` (NIFTY50), 252-day annualisation, watchlist seeds for `^NSEBANK`, `RELIANCE.NS`, `GAIL.NS`, `ONGC.NS`, `LODHA.NS`.
- **Standalone or in the stack** — uses MarketLake when present (the shared data contract), gracefully falls back to yfinance otherwise.

## Quickstart

```bash
git clone <your-repo-url> regime_radar
cd regime_radar
uv sync                                  # or: pip install -e ".[dev]"

# Detect the current regime for NIFTY50
uv run regime detect --symbol ^NSEI

# Walk-forward synthetic benchmark — the 70% target enforced as a test
uv run regime eval

# Watchlist scan
uv run regime scan

# Save plots
uv run regime plot --symbol ^NSEBANK --out-dir artifacts/
```

Example output:

```
╭────────── Regime ──────────╮
│ ^NSEI @ 2026-05-29 (1d)    │
│ TRENDING_UP                │
│ conf 73% | ann vol 14.2%   │
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
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Factor          ┃ Note                               ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ rule_vol_trend  │ Positive Sharpe +0.21, drift +25%  │
│ edmd_spectrum   │ |λ₁|=0.987≈1 — persistent mode     │
│ hmm_state       │ HMM Viterbi state with posterior   │
│                 │ 0.68                               │
└─────────────────┴────────────────────────────────────┘
```

## Installation

Requires **Python 3.12+**. We recommend [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync                       # installs runtime + dev deps
uv run regime --help          # CLI is exposed as `regime`
```

Or with pip:

```bash
pip install -e ".[dev]"
regime --help
```

Configuration is via environment variables (see `.env.example`). All knobs are namespaced `REGIME_*`.

## CLI reference

```text
regime detect      — full ensemble + transition risk for one symbol
regime explain     — eigenvalue table + voter contributions for the latest call
regime scan        — run detection across a watchlist
regime plot        — write spectrum + price-with-regime PNGs
regime eval        — synthetic walk-forward benchmark (the 70% headline)
```

All commands accept `--symbol`, `--interval`, `--lookback`, `--window`.

## Library use

```python
from regime_radar.data import load
from regime_radar.core.regime import detect_regime
from regime_radar.core.transition import score_transition_risk

df = load(symbol="^NSEI", interval="1d", lookback_days=720)
close = df["close"].to_numpy()

result = detect_regime(close[-130:], symbol="^NSEI")
print(result.label, result.confidence)
for reason in result.reasons:
    print(" -", reason.note)

risk = score_transition_risk(close, symbol="^NSEI", window=126)
print(f"transition risk: {risk.risk_score:.0%} — {risk.note}")
```

See `examples/quickstart.py` and `examples/synthetic_demo.py` for runnable examples.

## The accuracy claim, honestly

Regime detection has no single ground-truth dataset, so any single accuracy number is meaningful only relative to a definition. We make two distinct claims:

### 1. On synthetic ground truth — ≥ 70% (target, enforced by a test)

The default battery in `regime_radar.eval` generates:

| Family | Generator | Parameters |
|---|---|---|
| `gbm_up` / `gbm_down` | GBM with strong drift | μ=±0.30, σ=0.15 |
| `ou_meanrev` | Ornstein-Uhlenbeck on log-price | θ=15, σ=0.25 |
| `high_vol_chop` | High-vol zero-drift RW | σ=0.50 |
| `low_vol_grind` | Low-vol low-drift GBM | μ=0.06, σ=0.06 |
| `regime_switching` | Stitched: gbm_up + chop + ou | — |

With **5 seeds per scenario × 1008 bars each × walk-forward window=252, step=21**, we typically observe:

```
OVERALL:                 75-78%
  gbm_down               77%
  gbm_up                 66%
  high_vol_chop          96%
  low_vol_grind          83%
  ou_meanrev             80%
  regime_switching       53%  (intentionally hard — stitched series)
```

Re-run anytime via `regime eval` or `pytest tests/test_benchmark_accuracy.py -v -s`. A drop below 70% fails CI.

### 2. On real markets — *measure, don't claim*

Real-market "ground truth" requires a labelling rule (e.g. forward-vol percentile, drawdown depth, trend strength) and that rule *is* a modelling choice. The evaluation harness lets you plug in any labelling rule and get an honest per-symbol score — but RegimeRadar deliberately does **not** publish a single real-market accuracy number, because doing so would require defending a particular labelling rule as canonical.

What we will *not* do: claim a real-markets accuracy number without an evaluation transcript backing it. That's the kind of number that gets people hurt.

See [`docs/evaluation.md`](docs/evaluation.md) for the full methodology.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the dependency graph and design rationale, and [`docs/methods.md`](docs/methods.md) for the mathematical details. Quick map:

```
src/regime_radar/
├── core/        # pure math: observables, DMD, EDMD, HMM, rules, ensemble, transition
├── eval/        # synthetic generators + walk-forward + benchmark metrics
├── viz/         # matplotlib plots
├── cli/         # Typer sub-apps
├── data.py      # MarketLake-or-yfinance loader (single market-data entry point)
├── models.py    # Pydantic v2 domain models
└── config.py    # pydantic-settings runtime config
```

`core/` is pure (no I/O), fully unit-tested, and `mypy`-strict.

## Development

```bash
uv run pytest                      # unit tests (~3 seconds)
uv run pytest -m slow              # add synthetic benchmark suite (~80 seconds)
uv run ruff check . && uv run ruff format .
uv run mypy src/regime_radar/core
```

Currently **30/30 tests pass**: 14 math-layer tests, 14 ensemble tests on synthetic regimes, and 2 benchmark tests gating the 70% accuracy floor.

## Roadmap

Features designed to push accuracy higher and close the gap with proprietary systems. See [`docs/features.md`](docs/features.md) for the full discussion.

**Phase 1 (accuracy wins):**

- **Hankel-DMD / HAVOK** — time-delay embedding to capture non-Markovian dynamics. Expected 5-10 pp accuracy.
- **Cross-asset regime fingerprints** — joint regime over NIFTY + BANKNIFTY + INDIA VIX + USDINR.
- **Online / streaming EDMD** — incremental fits, unlocks intraday use.
- **Strategy-conditioned regimes** — define regimes by where *your strategy* makes/loses money. The genuine competitive differentiator.

**Phase 2 (explainability & UX):**

- Regime persistence model (how unusual is the current spell?).
- Empirical transition probability heatmap.
- News/sentiment overlay as a 4th voter.

## License

MIT — see [`LICENSE`](LICENSE).
