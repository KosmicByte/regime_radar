# CLI Reference

Every command, flag, and named watchlist available in RegimeRadar.

The CLI is exposed as `regime` after `uv sync` or `pip install -e .`.

```bash
uv run regime --help
```

## Commands

| Command | Purpose |
|---|---|
| [`regime detect`](#regime-detect) | Full ensemble + transition risk for one symbol |
| [`regime explain`](#regime-explain) | Eigenvalue table and voter contributions |
| [`regime scan`](#regime-scan) | Watchlist scan with sector groupings |
| [`regime plot`](#regime-plot) | Spectrum and price-with-regime PNG output |
| [`regime eval`](#regime-eval) | Synthetic walk-forward benchmark |

## Global conventions

All commands accept these common flags:

| Flag | Short | Default | Purpose |
|---|---|---|---|
| `--symbol <ticker>` | `-s` | `REGIME_DEFAULT_SYMBOL` (`^NSEI`) | Yahoo-style ticker |
| `--interval <i>` | `-i` | `REGIME_DEFAULT_INTERVAL` (`1d`) | Bar interval — `1d`, `1h`, `5m`, `15m` |
| `--lookback <days>` | — | `720` | Calendar days of history to load |
| `--window <bars>` | `-w` | `REGIME_WINDOW` (`126`) | Analysis window in bars |

### Ticker conventions

| Market | Suffix | Example |
|---|---|---|
| NSE equities | `.NS` | `RELIANCE.NS`, `TCS.NS` |
| BSE equities | `.BO` | `RELIANCE.BO` |
| NSE indices | `^` prefix | `^NSEI` (NIFTY 50), `^NSEBANK` (Bank Nifty) |
| US equities | none | `AAPL`, `MSFT` |

---

## `regime detect`

Runs the full three-voter ensemble plus transition-risk scoring and prints a Rich panel.

```bash
uv run regime detect --symbol RELIANCE.NS
uv run regime detect --symbol ^NSEI --window 252
uv run regime detect --symbol HDFCBANK.NS --json
```

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--symbol`, `-s` | `^NSEI` | Ticker |
| `--interval`, `-i` | `1d` | Bar interval |
| `--lookback` | `720` | Calendar days to load |
| `--window`, `-w` | `126` | Analysis window in bars |
| `--no-transition` | off | Skip transition-risk pass |
| `--json` | off | Emit JSON instead of a Rich panel |

### Output

A Rich panel with the regime label, confidence, probability distribution over all labels, the three reasons that produced the label (ordered by contribution), the individual votes of each method (EDMD / HMM / Rule), and the transition-risk score colour-coded by severity.

---

## `regime explain`

Prints the full eigenvalue spectrum and the voter-contribution breakdown for the latest detection on a symbol. Use when you want to interrogate *why* a label was assigned.

```bash
uv run regime explain --symbol ^NSEI
uv run regime explain --symbol RELIANCE.NS --top-k 8
```

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--symbol`, `-s` | `^NSEI` | Ticker |
| `--interval`, `-i` | `1d` | Bar interval |
| `--lookback` | `720` | Calendar days to load |
| `--top-k` | `5` | Number of top eigenvalues to display |

### Output

A table of top-K Koopman eigenvalues with their magnitude, argument, growth rate, frequency, and relative energy. A second table lists the reasons that produced the label, ordered by contribution weight.

---

## `regime scan`

Scans a watchlist of symbols and prints a regime + transition-risk table for each. Ships with **35 named watchlists** spanning all major NSE sectors plus group-level scans.

```bash
uv run regime scan                                    # tight 13-name default
uv run regime scan --watchlist banks-private          # sector
uv run regime scan --watchlist tata                   # group
uv run regime scan --watchlist broad --delay-ms 500   # big scan
uv run regime scan --symbol HDFCBANK.NS --symbol ICICIBANK.NS  # ad-hoc
uv run regime scan --list                             # show all watchlists
```

### Flags

| Flag | Short | Default | Purpose |
|---|---|---|---|
| `--symbol` | `-s` | — | Ad-hoc ticker (repeatable). Overrides `--watchlist`. |
| `--watchlist` | `-W` | — | Name of a predefined watchlist (see below) |
| `--list` | — | off | Print every named watchlist and exit |
| `--interval` | `-i` | `1d` | Bar interval |
| `--lookback` | — | `720` | Calendar days to load |
| `--delay-ms` | — | `0` | Sleep N ms between symbols — use for big scans to avoid Yahoo rate-limits |

### Named watchlists

| Name | Size | Contents |
|---|---|---|
| `default` | 13 | Bellwether daily scan — top 10 mega-caps + 3 indices |
| `indices` | 19 | All Nifty sector + broad-market indices |
| `banks-private` | 10 | HDFC, ICICI, Kotak, Axis, IndusInd, IDFC First, Federal, RBL, Bandhan, AU |
| `banks-psu` | 8 | SBI, PNB, BoB, Canara, Union, Indian, BoI, IOB |
| `financials` | 14 | NBFCs, insurance, AMCs (Bajaj Finance, Cholamandalam, SBI Life, HDFC Life, …) |
| `it` | 15 | Tier-1 and tier-2 IT services |
| `oil-gas` | 11 | RIL, ONGC, OMCs, gas distribution |
| `power` | 10 | Power generation + transmission + renewables |
| `fmcg` | 14 | Staples, beverages, personal care |
| `auto-oem` | 10 | All four-wheeler and two-wheeler OEMs |
| `auto-ancillary` | 10 | Tyres, components, forgings |
| `pharma` | 18 | Generics, APIs, MNCs in India |
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
| `agri` | 9 | Fertilisers, agri-chem |
| `textiles` | 8 | Page, Vardhman, Trident |
| `adani` | 9 | Adani group — highly correlated regimes |
| `tata` | 12 | Tata group |
| `midcap` | 14 | High-conviction mid-caps with vol |
| `defence` | 9 | HAL, BEL, BDL, shipyards |
| `psu` | 20 | Cross-sector PSU pulse |
| `global` | 12 | US mega-caps + Indian ADRs |
| `broad` | 69 | Composite of sector heads — broadest practical scan |

Run `regime scan --list` for the complete contents of each watchlist.

### Output

A table with one row per symbol: regime label, confidence, annualised vol, trend Sharpe, transition-risk score (colour-coded green / yellow / red by severity), and a one-line note.

### Rate limits

Yahoo throttles burst requests. For scans over ~20 symbols, use `--delay-ms 500`. The `broad`, `psu`, and `pharma` watchlists in particular benefit from this. Once MarketLake is the data source, rate-limiting stops being a concern.

---

## `regime plot`

Emits two PNGs to the artifacts directory: the Koopman eigenvalue spectrum on the complex plane, and the price chart annotated with the current regime label.

```bash
uv run regime plot --symbol ^NSEBANK
uv run regime plot --symbol RELIANCE.NS --out-dir plots/
```

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--symbol`, `-s` | `^NSEI` | Ticker |
| `--interval`, `-i` | `1d` | Bar interval |
| `--lookback` | `720` | Calendar days to load |
| `--out-dir` | `REGIME_ARTIFACTS_DIR` (`./artifacts`) | Output directory |

### Output

Two files in `<out-dir>`:
- `<symbol>_spectrum.png` — eigenvalues plotted on the complex plane with the unit circle for reference
- `<symbol>_price_regime.png` — price chart annotated with the regime label, confidence, and top reason

---

## `regime eval`

Runs the synthetic walk-forward benchmark and prints per-scenario accuracy plus an overall confusion matrix. This is the headline 70% accuracy gate.

```bash
uv run regime eval
uv run regime eval --strict                  # disable BREAKOUT → TRENDING_UP grouping
uv run regime eval --window 252 --plot       # custom window + write confusion PNG
```

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--window`, `-w` | `126` | Analysis window in bars |
| `--step` | `21` | Stride between successive walk-forward fits |
| `--edmd-rank` | `10` | SVD truncation rank for EDMD |
| `--hmm-states` | `3` | HMM state count |
| `--grouped` / `--strict` | grouped | Grouped mode folds BREAKOUT into TRENDING_UP for less-strict scoring |
| `--plot` | off | Write a confusion-matrix PNG to artifacts dir |

### Output

A per-scenario accuracy table, the overall headline accuracy with colour-coded threshold indication (green ≥ 70%, yellow ≥ 55%, red below), and a confusion matrix.

Expected runtime: ~80 seconds for the default battery (30 series × ~36 walk-forward windows each).

See [`docs/evaluation.md`](evaluation.md) for the full methodology and accuracy framing.

---

## `regime calibrate`

Fits the calibration artifact — temperature scaling, the conformal threshold, and the
out-of-distribution reference — by harvesting detections across the synthetic battery, then
writes it to the configured artifact path. Once present, every `detect`/`scan` run uses it
automatically to produce calibrated confidence, a conformal prediction set, and an OOD score.

```bash
uv run regime calibrate                       # fit + write artifacts/calibration.json
uv run regime calibrate --alpha 0.05          # target 95% conformal coverage
uv run regime calibrate --step 42 --out ./artifacts/calibration.json
```

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--window`, `-w` | `252` | Walk-forward window used during harvesting |
| `--step` | `21` | Stride between harvested windows |
| `--edmd-rank` | `10` | SVD truncation rank for EDMD |
| `--hmm-states` | `3` | HMM state count |
| `--alpha` | `0.1` | Miscoverage rate; conformal coverage target is `1 - alpha` |
| `--out`, `-o` | from settings | Artifact output path |

### Output

A before/after calibration-quality table (ECE and Brier), then a summary with the fitted
temperature, empirical conformal coverage vs target, OOD flag rate, and the harvested sample
count. Because the fit is currently on synthetic data, the command prints a note that coverage
and OOD thresholds are indicative rather than real-market guarantees — re-run on real data once
it is wired in to refresh the artifact (no code change required).

> Temperature scaling preserves the argmax, so calibration never changes the regime label —
> only how honest the confidence is. Running `calibrate` therefore cannot regress accuracy.

---

## Configuration

CLI defaults are populated from environment variables, all namespaced `REGIME_*`. A template is provided at `.env.example`:

| Variable | Default | Purpose |
|---|---|---|
| `REGIME_DEFAULT_SYMBOL` | `^NSEI` | Default for `--symbol` |
| `REGIME_DEFAULT_INTERVAL` | `1d` | Default for `--interval` |
| `REGIME_WINDOW` | `126` | Default for `--window` |
| `REGIME_ARTIFACTS_DIR` | `./artifacts` | Default for `--out-dir` on `plot` and `eval --plot` |
| `REGIME_EDMD_RANK` | `10` | Default EDMD truncation rank |
| `REGIME_HMM_N_STATES` | `3` | Default HMM state count |
| `REGIME_TRANSITION_THRESHOLD` | `0.6` | Risk above this is flagged as a crossed threshold |
| `REGIME_CALIBRATION_ENABLED` | `true` | Use the calibration artifact when one is present |
| `REGIME_CALIBRATION_ARTIFACT` | `./artifacts/calibration.json` | Path to the fitted calibration artifact |
| `REGIME_PROVENANCE_ENABLED` | `false` | Append an immutable inference record per detection |
| `MARKETLAKE_DATA_DIR` | — | If set and MarketLake is importable, used in preference to yfinance |

See [`config.py`](../src/regime_radar/config.py) for the full list.

## JSON output

Pass `--json` to `regime detect` for machine-readable output:

```bash
uv run regime detect --symbol ^NSEI --json
```

The structure mirrors the `RegimeResult` and `TransitionRisk` Pydantic models — see [`models.py`](../src/regime_radar/models.py).

## Troubleshooting

**Yahoo rate-limit errors on scan.** Use `--delay-ms 500` for any watchlist over ~20 names. For persistent issues, wait 30–60 seconds and retry.

**Insufficient data on intraday intervals.** Yahoo's intraday history is limited (~60 days for `5m`, ~730 days for `1h`). Reduce `--lookback` if you see "insufficient data" warnings.

**HMM convergence warnings on `regime eval`.** Non-fatal. Arises from hmmlearn's EM on synthetic noise. The detector handles this internally by falling back to diagonal then spherical covariance.

**Unknown watchlist name.** Run `regime scan --list` to see the full registry.
