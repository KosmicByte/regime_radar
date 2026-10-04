# CLI Reference

The CLI is installed as `regime` via `uv sync` or `pip install -e .`.

```bash
uv run regime --help
```

## Commands

| Command | Purpose |
|---|---|
| [`regime detect`](#regime-detect) | Full ensemble and transition risk for one symbol |
| [`regime explain`](#regime-explain) | Eigenvalue table and voter contributions |
| [`regime scan`](#regime-scan) | Watchlist scan |
| [`regime plot`](#regime-plot) | Spectrum and price-with-regime PNGs |
| [`regime eval`](#regime-eval) | Synthetic walk-forward benchmark |

## Common flags

| Flag | Short | Default | Purpose |
|---|---|---|---|
| `--symbol <ticker>` | `-s` | `REGIME_DEFAULT_SYMBOL` (`^NSEI`) | Yahoo-style ticker |
| `--interval <i>` | `-i` | `REGIME_DEFAULT_INTERVAL` (`1d`) | Bar interval: `1d`, `1h`, `5m`, `15m` |
| `--lookback <days>` | — | `720` | Calendar days of history |
| `--window <bars>` | `-w` | `REGIME_WINDOW` (`126`) | Analysis window in bars |

### Ticker conventions

| Market | Format | Example |
|---|---|---|
| NSE equities | `.NS` suffix | `RELIANCE.NS`, `TCS.NS` |
| BSE equities | `.BO` suffix | `RELIANCE.BO` |
| NSE indices | `^` prefix | `^NSEI` (NIFTY 50), `^NSEBANK` (Bank Nifty) |
| US equities | none | `AAPL`, `MSFT` |

---

## `regime detect`

Runs the three-voter ensemble and transition-risk scoring.

```bash
uv run regime detect --symbol RELIANCE.NS
uv run regime detect --symbol ^NSEI --window 252
uv run regime detect --symbol HDFCBANK.NS --json
```

| Flag | Default | Purpose |
|---|---|---|
| `--symbol`, `-s` | `^NSEI` | Ticker |
| `--interval`, `-i` | `1d` | Bar interval |
| `--lookback` | `720` | Calendar days |
| `--window`, `-w` | `126` | Analysis window in bars |
| `--no-transition` | off | Skip transition-risk pass |
| `--json` | off | JSON output |

**Output**: regime label, confidence, probability distribution, top three reasons by contribution, individual votes (EDMD / HMM / Rule), and colour-coded transition-risk score.

---

## `regime explain`

Prints the eigenvalue spectrum and voter-contribution breakdown for the latest detection.

```bash
uv run regime explain --symbol ^NSEI
uv run regime explain --symbol RELIANCE.NS --top-k 8
```

| Flag | Default | Purpose |
|---|---|---|
| `--symbol`, `-s` | `^NSEI` | Ticker |
| `--interval`, `-i` | `1d` | Bar interval |
| `--lookback` | `720` | Calendar days |
| `--top-k` | `5` | Number of eigenvalues displayed |

**Output**: table of top-K Koopman eigenvalues (magnitude, argument, growth rate, frequency, relative energy); table of reasons ordered by contribution weight.

---

## `regime scan`

Scans a watchlist and prints regime and transition risk per symbol. 35 named watchlists are included.

```bash
uv run regime scan                                             # default watchlist
uv run regime scan --watchlist banks-private                   # sector
uv run regime scan --watchlist tata                            # group
uv run regime scan --watchlist broad --delay-ms 500            # large scan
uv run regime scan --symbol HDFCBANK.NS --symbol ICICIBANK.NS  # ad-hoc
uv run regime scan --list                                      # list watchlists
```

| Flag | Short | Default | Purpose |
|---|---|---|---|
| `--symbol` | `-s` | — | Ad-hoc ticker (repeatable); overrides `--watchlist` |
| `--watchlist` | `-W` | — | Named watchlist |
| `--list` | — | off | Print all watchlists and exit |
| `--interval` | `-i` | `1d` | Bar interval |
| `--lookback` | — | `720` | Calendar days |
| `--delay-ms` | — | `0` | Delay between symbols (ms) |

### Named watchlists

| Name | Size | Contents |
|---|---|---|
| `default` | 13 | Top 10 mega-caps + 3 indices |
| `indices` | 19 | Nifty sector and broad-market indices |
| `banks-private` | 10 | HDFC, ICICI, Kotak, Axis, IndusInd, IDFC First, Federal, RBL, Bandhan, AU |
| `banks-psu` | 8 | SBI, PNB, BoB, Canara, Union, Indian, BoI, IOB |
| `financials` | 14 | NBFCs, insurance, AMCs |
| `it` | 15 | Tier-1 and tier-2 IT services |
| `oil-gas` | 11 | RIL, ONGC, OMCs, gas distribution |
| `power` | 10 | Generation, transmission, renewables |
| `fmcg` | 14 | Staples, beverages, personal care |
| `auto-oem` | 10 | Four- and two-wheeler OEMs |
| `auto-ancillary` | 10 | Tyres, components, forgings |
| `pharma` | 18 | Generics, APIs, MNCs |
| `healthcare` | 8 | Hospitals, diagnostics |
| `metals` | 12 | Steel, aluminium, copper, zinc |
| `mining` | 4 | Coal India, NMDC, MOIL, Hindzinc |
| `cement` | 10 | UltraTech, Shree, Ambuja, ACC, mid-caps |
| `capital-goods` | 14 | L&T, Siemens, ABB, defence equipment |
| `infra-realty` | 14 | DLF, Godrej, Lodha, roads, ports |
| `chemicals` | 15 | Specialty and commodity chemicals |
| `paints` | 5 | Asian Paints, Berger, Kansai, Akzo, Indigo |
| `telecom` | 6 | Airtel, Idea, towers, fibre |
| `retail` | 10 | DMart, Trent, Titan, jewellery |
| `hospitality` | 6 | Hotels, IRCTC, travel |
| `logistics` | 10 | Airlines, ports, freight |
| `media` | 8 | Broadcast, print, OTT |
| `consumer-durables` | 13 | Appliances, electronics, kitchenware |
| `agri` | 9 | Fertilisers, agri-chemicals |
| `textiles` | 8 | Page, Vardhman, Trident |
| `adani` | 9 | Adani group |
| `tata` | 12 | Tata group |
| `midcap` | 14 | High-volatility mid-caps |
| `defence` | 9 | HAL, BEL, BDL, shipyards |
| `psu` | 20 | Cross-sector PSUs |
| `global` | 12 | US mega-caps and Indian ADRs |
| `broad` | 69 | Composite of sector leaders |

Full contents: `regime scan --list`.

**Output**: one row per symbol — regime label, confidence, annualised vol, trend Sharpe, colour-coded transition-risk score, note.

**Rate limits**: Yahoo throttles burst requests. Use `--delay-ms 500` for scans above ~20 symbols (notably `broad`, `psu`, `pharma`). Not required when MarketLake is the data source.

---

## `regime plot`

Writes two PNGs to the artifacts directory.

```bash
uv run regime plot --symbol ^NSEBANK
uv run regime plot --symbol RELIANCE.NS --out-dir plots/
```

| Flag | Default | Purpose |
|---|---|---|
| `--symbol`, `-s` | `^NSEI` | Ticker |
| `--interval`, `-i` | `1d` | Bar interval |
| `--lookback` | `720` | Calendar days |
| `--out-dir` | `REGIME_ARTIFACTS_DIR` (`./artifacts`) | Output directory |

**Output**:
- `<symbol>_spectrum.png` — eigenvalues on the complex plane with unit circle
- `<symbol>_price_regime.png` — price chart annotated with regime label, confidence, and top reason

---

## `regime eval`

Runs the synthetic walk-forward benchmark (70% accuracy gate).

```bash
uv run regime eval
uv run regime eval --strict
uv run regime eval --window 252 --plot
```

| Flag | Default | Purpose |
|---|---|---|
| `--window`, `-w` | `126` | Analysis window in bars |
| `--step` | `21` | Stride between walk-forward fits |
| `--edmd-rank` | `10` | EDMD SVD truncation rank |
| `--hmm-states` | `3` | HMM state count |
| `--grouped` / `--strict` | grouped | Grouped folds BREAKOUT into TRENDING_UP |
| `--plot` | off | Write confusion-matrix PNG |

**Output**: per-scenario accuracy, overall accuracy (green ≥ 70%, yellow ≥ 55%, red below), confusion matrix.

**Runtime**: ~80 s (30 series × ~36 walk-forward windows). Methodology: [`docs/evaluation.md`](evaluation.md).

---

## Configuration

Defaults are read from `REGIME_*` environment variables. Template: `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `REGIME_DEFAULT_SYMBOL` | `^NSEI` | Default `--symbol` |
| `REGIME_DEFAULT_INTERVAL` | `1d` | Default `--interval` |
| `REGIME_WINDOW` | `126` | Default `--window` |
| `REGIME_ARTIFACTS_DIR` | `./artifacts` | Default `--out-dir` for `plot` and `eval --plot` |
| `REGIME_EDMD_RANK` | `10` | EDMD truncation rank |
| `REGIME_HMM_N_STATES` | `3` | HMM state count |
| `REGIME_TRANSITION_THRESHOLD` | `0.6` | Transition-risk flag threshold |
| `MARKETLAKE_DATA_DIR` | — | Use MarketLake instead of yfinance when set and importable |

Full list: [`config.py`](../src/regime_radar/config.py).

## JSON output

```bash
uv run regime detect --symbol ^NSEI --json
```

Structure follows the `RegimeResult` and `TransitionRisk` models in [`models.py`](../src/regime_radar/models.py).

## Troubleshooting

| Issue | Resolution |
|---|---|
| Yahoo rate-limit errors | Use `--delay-ms 500`; retry after 30–60 s |
| Insufficient intraday data | Yahoo limits history (~60 days for `5m`, ~730 days for `1h`); reduce `--lookback` |
| HMM convergence warnings in `eval` | Non-fatal; covariance falls back to diagonal, then spherical |
| Unknown watchlist | `regime scan --list` |
