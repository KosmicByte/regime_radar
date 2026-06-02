"""`regime scan` — run detection across a watchlist of symbols and print a table.

Watchlists are organised by sector / theme. Use `--watchlist <name>` to pick one,
`--symbol <ticker>` (repeatable) for ad-hoc, or no args for the tight daily default.

Run `regime scan --list` to see every named watchlist and its contents.

A note on Yahoo rate limits: scanning more than ~30 symbols in quick succession
will start producing empty-data errors as yfinance gets throttled. For large
scans, expect intermittent errors and re-run. The longer-term fix is MarketLake.
"""

from __future__ import annotations

import time

import typer
from rich.console import Console
from rich.table import Table

from regime_radar.config import get_settings
from regime_radar.core.regime import detect_regime
from regime_radar.core.transition import score_transition_risk
from regime_radar.data import load

app = typer.Typer(invoke_without_command=True)
console = Console()


# ---------------------------------------------------------------------------
# Watchlists — organised by sector / theme. F&O-eligible NSE names where
# possible (most liquid, cleanest Yahoo data). Tickers use Yahoo's NSE suffix
# `.NS`; for BSE use `.BO`. Indices use the `^` prefix.
# ---------------------------------------------------------------------------

# Tight daily-driver default — fast, hits all the bellwethers, no rate limits.
DEFAULT_WATCHLIST = [
    "^NSEI", "^NSEBANK", "^CNXIT",
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "BHARTIARTL.NS", "ITC.NS", "LT.NS", "SBIN.NS", "KOTAKBANK.NS",
]

# All major NSE sector indices + broad-market.
WATCHLIST_INDICES = [
    "^NSEI",         # Nifty 50
    "^NSEBANK",      # Bank Nifty
    "^CNXIT",        # Nifty IT
    "^CNXAUTO",      # Nifty Auto
    "^CNXFMCG",      # Nifty FMCG
    "^CNXPHARMA",    # Nifty Pharma
    "^CNXMETAL",     # Nifty Metal
    "^CNXREALTY",    # Nifty Realty
    "^CNXENERGY",    # Nifty Energy
    "^CNXFIN",       # Nifty Financial Services
    "^CNXMEDIA",     # Nifty Media
    "^CNXINFRA",     # Nifty Infrastructure
    "^CNXPSE",       # Nifty PSE
    "^CNXPSUBANK",   # Nifty PSU Bank
    "^CNX100",       # Nifty 100
    "^CNX200",       # Nifty 200
    "^CNX500",       # Nifty 500
    "^CNXMIDCAP",    # Nifty Midcap 100
    "^CNXSC",        # Nifty Smallcap 100
]

# Private-sector banks + broader financials.
WATCHLIST_BANKS_PRIVATE = [
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "INDUSINDBK.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS", "RBLBANK.NS",
    "BANDHANBNK.NS", "AUBANK.NS",
]

# PSU banks.
WATCHLIST_BANKS_PSU = [
    "SBIN.NS", "PNB.NS", "BANKBARODA.NS", "CANBK.NS",
    "UNIONBANK.NS", "INDIANB.NS", "BANKINDIA.NS", "IOB.NS",
]

# NBFCs, insurance, AMCs — non-banking financials.
WATCHLIST_FINANCIALS = [
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS",
    "MANAPPURAM.NS", "PFC.NS", "RECLTD.NS", "LICHSGFIN.NS",
    "SBILIFE.NS", "HDFCLIFE.NS", "ICICIPRULI.NS", "ICICIGI.NS",
    "HDFCAMC.NS", "MCX.NS",
]

# IT services & products.
WATCHLIST_IT = [
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
    "LTIM.NS", "MPHASIS.NS", "PERSISTENT.NS", "COFORGE.NS", "OFSS.NS",
    "LTTS.NS", "TATAELXSI.NS", "BSOFT.NS", "KPITTECH.NS", "ZENSARTECH.NS",
]

# Oil, gas, refining.
WATCHLIST_OIL_GAS = [
    "RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "HINDPETRO.NS",
    "GAIL.NS", "PETRONET.NS", "OIL.NS", "MGL.NS", "IGL.NS",
    "GUJGASLTD.NS",
]

# Power generation & transmission.
WATCHLIST_POWER = [
    "POWERGRID.NS", "NTPC.NS", "TATAPOWER.NS", "ADANIPOWER.NS",
    "JSWENERGY.NS", "NHPC.NS", "SJVN.NS", "TORNTPOWER.NS",
    "CESC.NS", "ADANIGREEN.NS",
]

# FMCG — staples, personal care, food.
WATCHLIST_FMCG = [
    "ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS",
    "DABUR.NS", "GODREJCP.NS", "MARICO.NS", "TATACONSUM.NS",
    "COLPAL.NS", "EMAMILTD.NS", "VBL.NS", "UBL.NS",
    "RADICO.NS", "MCDOWELL-N.NS",
]

# Auto OEMs.
WATCHLIST_AUTO_OEM = [
    "MARUTI.NS", "M&M.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS",
    "HEROMOTOCO.NS", "EICHERMOT.NS", "TVSMOTOR.NS", "ASHOKLEY.NS",
    "FORCEMOT.NS", "ESCORTS.NS",
]

# Auto ancillaries & tyres.
WATCHLIST_AUTO_ANCILLARY = [
    "BOSCHLTD.NS", "MOTHERSON.NS", "BALKRISIND.NS", "MRF.NS",
    "APOLLOTYRE.NS", "CEATLTD.NS", "BHARATFORG.NS", "SUNDRMFAST.NS",
    "EXIDEIND.NS", "AMARAJABAT.NS",
]

# Pharma & APIs.
WATCHLIST_PHARMA = [
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
    "LUPIN.NS", "AUROPHARMA.NS", "TORNTPHARM.NS", "ZYDUSLIFE.NS",
    "ALKEM.NS", "BIOCON.NS", "GLENMARK.NS", "IPCALAB.NS",
    "ABBOTINDIA.NS", "PFIZER.NS", "GLAXO.NS", "SANOFI.NS",
    "LAURUSLABS.NS", "GRANULES.NS",
]

# Hospitals, diagnostics, healthcare services.
WATCHLIST_HEALTHCARE = [
    "APOLLOHOSP.NS", "MAXHEALTH.NS", "FORTIS.NS", "NARAYANHRT.NS",
    "DRLAL.NS", "METROPOLIS.NS", "THYROCARE.NS", "POLYMED.NS",
]

# Ferrous & non-ferrous metals.
WATCHLIST_METALS = [
    "TATASTEEL.NS", "JSWSTEEL.NS", "SAIL.NS", "JINDALSTEL.NS",
    "HINDALCO.NS", "NATIONALUM.NS", "VEDL.NS", "HINDCOPPER.NS",
    "NMDC.NS", "MOIL.NS", "RATNAMANI.NS", "WELCORP.NS",
]

# Coal & mining.
WATCHLIST_MINING = [
    "COALINDIA.NS", "NMDC.NS", "MOIL.NS", "HINDZINC.NS",
]

# Cement.
WATCHLIST_CEMENT = [
    "ULTRACEMCO.NS", "GRASIM.NS", "SHREECEM.NS", "AMBUJACEM.NS",
    "ACC.NS", "DALBHARAT.NS", "RAMCOCEM.NS", "JKCEMENT.NS",
    "JKLAKSHMI.NS", "BIRLACORPN.NS",
]

# Capital goods, defence, engineering.
WATCHLIST_CAPITAL_GOODS = [
    "LT.NS", "SIEMENS.NS", "ABB.NS", "HAVELLS.NS",
    "BHEL.NS", "BEL.NS", "HAL.NS", "BHARATFORG.NS",
    "CUMMINSIND.NS", "THERMAX.NS", "AIAENG.NS", "TIMKEN.NS",
    "SCHAEFFLER.NS", "GRINDWELL.NS",
]

# Construction, real estate, infrastructure.
WATCHLIST_INFRA_REALTY = [
    "DLF.NS", "GODREJPROP.NS", "LODHA.NS", "OBEROIRLTY.NS",
    "PRESTIGE.NS", "BRIGADE.NS", "PHOENIXLTD.NS", "SOBHA.NS",
    "GMRINFRA.NS", "ADANIPORTS.NS", "IRB.NS", "KNRCON.NS",
    "PNCINFRA.NS", "NCC.NS",
]

# Chemicals & specialty chemicals.
WATCHLIST_CHEMICALS = [
    "PIDILITIND.NS", "SRF.NS", "AARTI INDS.NS", "ATUL.NS",
    "DEEPAKNTR.NS", "GUJFLUORO.NS", "NAVINFLUOR.NS", "VINATIORGA.NS",
    "PIIND.NS", "UPL.NS", "TATACHEM.NS", "ALKYLAMINE.NS",
    "FINEORG.NS", "NOCIL.NS", "CLEAN.NS",
]

# Paints.
WATCHLIST_PAINTS = [
    "ASIANPAINT.NS", "BERGEPAINT.NS", "KANSAINER.NS",
    "AKZOINDIA.NS", "INDIGOPNTS.NS",
]

# Telecom & digital infra.
WATCHLIST_TELECOM = [
    "BHARTIARTL.NS", "IDEA.NS", "TATACOMM.NS", "INDUSTOWER.NS",
    "RAILTEL.NS", "HFCL.NS",
]

# Retail, jewellery, apparel.
WATCHLIST_RETAIL = [
    "DMART.NS", "TRENT.NS", "TITAN.NS", "ABFRL.NS",
    "VMART.NS", "VEDANTFASH.NS", "SHOPERSTOP.NS", "KALYANKJIL.NS",
    "PCJEWELLER.NS", "RAYMOND.NS",
]

# Hospitality, travel, leisure.
WATCHLIST_HOSPITALITY = [
    "INDHOTEL.NS", "EIHOTEL.NS", "CHALET.NS", "LEMONTREE.NS",
    "IRCTC.NS", "EASEMYTRIP.NS",
]

# Aviation & logistics.
WATCHLIST_LOGISTICS = [
    "INTERGLOBE.NS", "SPICEJET.NS", "CONCOR.NS", "ADANIPORTS.NS",
    "GMRINFRA.NS", "GPPL.NS", "BLUEDART.NS", "TCI.NS",
    "MAHLOG.NS", "DELHIVERY.NS",
]

# Media, entertainment.
WATCHLIST_MEDIA = [
    "ZEEL.NS", "SUNTV.NS", "PVRINOX.NS", "TV18BRDCST.NS",
    "NETWORK18.NS", "DBCORP.NS", "JAGRAN.NS", "HATHWAY.NS",
]

# Consumer durables, electronics.
WATCHLIST_CONSUMER_DURABLES = [
    "HAVELLS.NS", "VOLTAS.NS", "WHIRLPOOL.NS", "CROMPTON.NS",
    "DIXON.NS", "AMBER.NS", "BLUESTARCO.NS", "BAJAJELEC.NS",
    "VGUARD.NS", "ORIENTELEC.NS", "TTKPRESTIG.NS", "HAWKINCOOK.NS",
    "RAJESHEXPO.NS",
]

# Fertilisers, agri-chemicals.
WATCHLIST_AGRI = [
    "UPL.NS", "PIIND.NS", "COROMANDEL.NS", "CHAMBLFERT.NS",
    "GNFC.NS", "GSFC.NS", "RCF.NS", "NFL.NS",
    "DEEPAKFERT.NS",
]

# Textiles.
WATCHLIST_TEXTILES = [
    "PAGEIND.NS", "VARDHMAN.NS", "TRIDENT.NS", "WELSPUNIND.NS",
    "GRASIM.NS", "ALOKINDS.NS", "RAYMOND.NS", "ARVIND.NS",
]

# Adani group — useful to scan together because regimes are highly correlated.
WATCHLIST_ADANI = [
    "ADANIENT.NS", "ADANIPORTS.NS", "ADANIGREEN.NS", "ADANIPOWER.NS",
    "ADANIENSOL.NS", "ATGL.NS", "AMBUJACEM.NS", "ACC.NS",
    "NDTV.NS",
]

# Tata group.
WATCHLIST_TATA = [
    "TCS.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TATAPOWER.NS",
    "TATACONSUM.NS", "TATACHEM.NS", "TATACOMM.NS", "TATAELXSI.NS",
    "TITAN.NS", "TRENT.NS", "INDHOTEL.NS", "VOLTAS.NS",
]

# High-conviction mid-caps — more vol, more interesting regime shifts.
WATCHLIST_MIDCAP_POWER = [
    "LODHA.NS", "PERSISTENT.NS", "POLYCAB.NS", "PAGEIND.NS",
    "HAVELLS.NS", "DLF.NS", "GODREJPROP.NS", "MUTHOOTFIN.NS",
    "MAXHEALTH.NS", "DIXON.NS", "PRESTIGE.NS", "KPITTECH.NS",
    "COFORGE.NS", "MPHASIS.NS",
]

# Defence — has its own narrative cycle (orders, geopolitics).
WATCHLIST_DEFENCE = [
    "HAL.NS", "BEL.NS", "BDL.NS", "MAZDOCK.NS",
    "GRSE.NS", "COCHINSHIP.NS", "BEML.NS", "MIDHANI.NS",
    "PARAS.NS",
]

# PSU stocks broadly — different macro drivers than private sector.
WATCHLIST_PSU = [
    "SBIN.NS", "PNB.NS", "BANKBARODA.NS", "CANBK.NS",
    "ONGC.NS", "COALINDIA.NS", "POWERGRID.NS", "NTPC.NS",
    "GAIL.NS", "IOC.NS", "BPCL.NS", "HINDPETRO.NS",
    "BEL.NS", "HAL.NS", "BHEL.NS", "SAIL.NS",
    "NMDC.NS", "PFC.NS", "RECLTD.NS", "RAILTEL.NS",
]

# Global ADRs — US-listed Indian companies + global names for cross-reference.
WATCHLIST_GLOBAL = [
    "AAPL", "MSFT", "NVDA", "GOOGL",
    "META", "AMZN", "TSLA", "JPM",
    "INFY", "WIT", "HDB", "IBN",  # ADRs: Infosys, Wipro, HDFC Bank, ICICI Bank
]


# ---------------------------------------------------------------------------
# Named-watchlist registry — what `--watchlist <name>` accepts.
# ---------------------------------------------------------------------------

NAMED_WATCHLISTS: dict[str, list[str]] = {
    "default": DEFAULT_WATCHLIST,
    "indices": WATCHLIST_INDICES,
    "banks-private": WATCHLIST_BANKS_PRIVATE,
    "banks-psu": WATCHLIST_BANKS_PSU,
    "financials": WATCHLIST_FINANCIALS,
    "it": WATCHLIST_IT,
    "oil-gas": WATCHLIST_OIL_GAS,
    "power": WATCHLIST_POWER,
    "fmcg": WATCHLIST_FMCG,
    "auto-oem": WATCHLIST_AUTO_OEM,
    "auto-ancillary": WATCHLIST_AUTO_ANCILLARY,
    "pharma": WATCHLIST_PHARMA,
    "healthcare": WATCHLIST_HEALTHCARE,
    "metals": WATCHLIST_METALS,
    "mining": WATCHLIST_MINING,
    "cement": WATCHLIST_CEMENT,
    "capital-goods": WATCHLIST_CAPITAL_GOODS,
    "infra-realty": WATCHLIST_INFRA_REALTY,
    "chemicals": WATCHLIST_CHEMICALS,
    "paints": WATCHLIST_PAINTS,
    "telecom": WATCHLIST_TELECOM,
    "retail": WATCHLIST_RETAIL,
    "hospitality": WATCHLIST_HOSPITALITY,
    "logistics": WATCHLIST_LOGISTICS,
    "media": WATCHLIST_MEDIA,
    "consumer-durables": WATCHLIST_CONSUMER_DURABLES,
    "agri": WATCHLIST_AGRI,
    "textiles": WATCHLIST_TEXTILES,
    "adani": WATCHLIST_ADANI,
    "tata": WATCHLIST_TATA,
    "midcap": WATCHLIST_MIDCAP_POWER,
    "defence": WATCHLIST_DEFENCE,
    "psu": WATCHLIST_PSU,
    "global": WATCHLIST_GLOBAL,
}

# Composite "everything liquid" — a broad pulse across the market.
# Built from sector heads so size stays manageable; total ~60 names.
NAMED_WATCHLISTS["broad"] = list(dict.fromkeys(  # dedup preserving order
    WATCHLIST_INDICES[:6]
    + WATCHLIST_BANKS_PRIVATE[:5]
    + WATCHLIST_BANKS_PSU[:3]
    + WATCHLIST_FINANCIALS[:5]
    + WATCHLIST_IT[:6]
    + WATCHLIST_OIL_GAS[:4]
    + WATCHLIST_POWER[:3]
    + WATCHLIST_FMCG[:5]
    + WATCHLIST_AUTO_OEM[:5]
    + WATCHLIST_PHARMA[:5]
    + WATCHLIST_METALS[:4]
    + WATCHLIST_CEMENT[:3]
    + WATCHLIST_CAPITAL_GOODS[:4]
    + WATCHLIST_INFRA_REALTY[:3]
    + WATCHLIST_CHEMICALS[:3]
    + WATCHLIST_TELECOM[:2]
    + WATCHLIST_RETAIL[:3]
))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def main(
    symbols: list[str] = typer.Option(  # noqa: B008
        None,
        "--symbol",
        "-s",
        help="Repeatable ticker. Overrides --watchlist.",
    ),
    watchlist: str = typer.Option(
        None,
        "--watchlist",
        "-W",
        help=(
            "Named watchlist to scan. Run `regime scan --list` to see all available. "
            "If omitted (and no --symbol), uses the tight `default` list."
        ),
    ),
    interval: str = typer.Option(None, "--interval", "-i"),
    lookback_days: int = typer.Option(720, "--lookback"),
    list_watchlists: bool = typer.Option(
        False,
        "--list",
        help="List all named watchlists and exit.",
    ),
    delay_ms: int = typer.Option(
        0,
        "--delay-ms",
        help="Pause between symbols in ms — useful to avoid Yahoo rate-limits on big scans.",
    ),
) -> None:
    """Regime + transition risk across a watchlist."""
    if list_watchlists:
        _print_watchlists()
        return

    settings = get_settings()
    interval = interval or settings.default_interval

    if symbols:
        syms = symbols
        title = f"Watchlist regimes ({len(syms)} symbols)"
    elif watchlist:
        if watchlist not in NAMED_WATCHLISTS:
            console.print(
                f"[red]Unknown watchlist '{watchlist}'.[/red] "
                f"Run `regime scan --list` to see available names."
            )
            raise typer.Exit(code=2)
        syms = NAMED_WATCHLISTS[watchlist]
        title = f"Watchlist regimes — {watchlist} ({len(syms)} symbols)"
    else:
        syms = DEFAULT_WATCHLIST
        title = f"Watchlist regimes — default ({len(syms)} symbols)"

    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Symbol")
    table.add_column("Regime")
    table.add_column("Conf", justify="right")
    table.add_column("Ann Vol", justify="right")
    table.add_column("Trend", justify="right")
    table.add_column("Risk", justify="right")
    table.add_column("Note")

    for i, sym in enumerate(syms):
        if i > 0 and delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        try:
            df = load(symbol=sym, interval=interval, lookback_days=lookback_days)
            if len(df) < settings.window + 30:
                table.add_row(sym, "[yellow]insufficient data[/yellow]", "-", "-", "-", "-", "-")
                continue
            close = df["close"].to_numpy()
            ts = df["ts"]
            result = detect_regime(
                close=close[-(settings.window + 5) :],
                timestamps=ts.iloc[-(settings.window + 5) :],
                symbol=sym,
                interval=interval,
                edmd_rank=settings.edmd_rank,
                hmm_n_states=settings.hmm_n_states,
            )
            try:
                risk = score_transition_risk(
                    close=close,
                    timestamps=ts,
                    symbol=sym,
                    interval=interval,
                    window=settings.window,
                    threshold=settings.transition_threshold,
                )
                risk_str = f"{risk.risk_score:.0%}"
                risk_note = risk.note
                risk_val = risk.risk_score
            except Exception:  # noqa: BLE001
                risk_str = "-"
                risk_note = ""
                risk_val = 0.0
            colour = (
                "red" if risk_val > 0.6 else
                "yellow" if risk_val > 0.4 else
                "green"
            )
            table.add_row(
                sym,
                result.label.value,
                f"{result.confidence:.0%}",
                f"{result.realized_vol:.1%}",
                f"{result.trend_strength:+.2f}",
                f"[{colour}]{risk_str}[/{colour}]",
                risk_note[:60],
            )
        except Exception as exc:  # noqa: BLE001
            table.add_row(sym, "[red]error[/red]", "-", "-", "-", "-", str(exc)[:60])

    console.print(table)


def _print_watchlists() -> None:
    """Pretty-print all named watchlists and their tickers."""
    t = Table(title="Available watchlists", show_header=True, header_style="bold")
    t.add_column("Name")
    t.add_column("#", justify="right")
    t.add_column("Tickers")
    for name, syms in NAMED_WATCHLISTS.items():
        preview = ", ".join(syms[:8])
        if len(syms) > 8:
            preview += f", … (+{len(syms) - 8} more)"
        t.add_row(name, str(len(syms)), preview)
    console.print(t)
    console.print(
        "\n[dim]Use with: [/dim][cyan]regime scan --watchlist <name>[/cyan]"
        "\n[dim]For large scans, add [/dim][cyan]--delay-ms 500[/dim] to avoid Yahoo rate limits."
    )
