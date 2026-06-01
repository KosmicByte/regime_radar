"""`regime scan` — run detection across a watchlist of symbols and print a table."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from regime_radar.config import get_settings
from regime_radar.core.regime import detect_regime
from regime_radar.core.transition import score_transition_risk
from regime_radar.data import load

app = typer.Typer(invoke_without_command=True)
console = Console()

DEFAULT_WATCHLIST = ["^NSEI", "^NSEBANK", "RELIANCE.NS", "GAIL.NS", "ONGC.NS", "LODHA.NS"]


@app.callback(invoke_without_command=True)
def main(
    symbols: list[str] = typer.Option(  # noqa: B008
        None,
        "--symbol",
        "-s",
        help="Repeatable. If omitted, uses the default Indian-markets watchlist.",
    ),
    interval: str = typer.Option(None, "--interval", "-i"),
    lookback_days: int = typer.Option(720, "--lookback"),
) -> None:
    """Regime + transition risk across a watchlist."""
    settings = get_settings()
    interval = interval or settings.default_interval
    syms = symbols or DEFAULT_WATCHLIST

    table = Table(title="Watchlist regimes", show_header=True, header_style="bold")
    table.add_column("Symbol")
    table.add_column("Regime")
    table.add_column("Conf", justify="right")
    table.add_column("Ann Vol", justify="right")
    table.add_column("Trend", justify="right")
    table.add_column("Risk", justify="right")
    table.add_column("Note")

    for sym in syms:
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
                    close=close, timestamps=ts, symbol=sym,
                    interval=interval, window=settings.window,
                    threshold=settings.transition_threshold,
                )
                risk_str = f"{risk.risk_score:.0%}"
                risk_note = risk.note
            except Exception:  # noqa: BLE001
                risk_str = "-"
                risk_note = ""
            colour = "red" if risk_str != "-" and float(risk_str.strip("%"))/100 > 0.6 else \
                     "yellow" if risk_str != "-" and float(risk_str.strip("%"))/100 > 0.4 else \
                     "green"
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
            table.add_row(sym, f"[red]error[/red]", "-", "-", "-", "-", str(exc)[:60])

    console.print(table)
