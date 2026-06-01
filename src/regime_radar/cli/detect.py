"""`regime detect` — run the full ensemble on a symbol and print a Rich summary."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from regime_radar.config import get_settings
from regime_radar.core.regime import detect_regime
from regime_radar.core.transition import score_transition_risk
from regime_radar.data import load

app = typer.Typer(invoke_without_command=True)
console = Console()


@app.callback(invoke_without_command=True)
def main(
    symbol: str = typer.Option(None, "--symbol", "-s", help="Ticker (e.g. ^NSEI, RELIANCE.NS)."),
    interval: str = typer.Option(None, "--interval", "-i", help="1d / 1h / 5m."),
    lookback_days: int = typer.Option(720, "--lookback", help="Calendar days of history to load."),
    window: int = typer.Option(None, "--window", "-w", help="Analysis window (bars)."),
    no_transition: bool = typer.Option(False, "--no-transition", help="Skip transition-risk pass."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a Rich panel."),
) -> None:
    """Detect the current regime for `symbol` and print an explanation."""
    settings = get_settings()
    symbol = symbol or settings.default_symbol
    interval = interval or settings.default_interval
    window = window or settings.window

    df = load(symbol=symbol, interval=interval, lookback_days=lookback_days)
    if len(df) < window + 30:
        console.print(
            f"[red]Need at least {window + 30} bars; loaded {len(df)}.[/red] "
            f"Try a longer --lookback."
        )
        raise typer.Exit(code=2)

    close = df["close"].to_numpy()
    ts = df["ts"]

    result = detect_regime(
        close=close[-(window + 5) :],
        timestamps=ts.iloc[-(window + 5) :],
        symbol=symbol,
        interval=interval,
        edmd_rank=settings.edmd_rank,
        hmm_n_states=settings.hmm_n_states,
    )

    risk = None
    if not no_transition:
        try:
            risk = score_transition_risk(
                close=close,
                timestamps=ts,
                symbol=symbol,
                interval=interval,
                window=window,
                threshold=settings.transition_threshold,
            )
        except Exception as exc:  # noqa: BLE001 — defensive: don't fail the whole command
            console.print(f"[yellow]Transition risk skipped:[/yellow] {exc}")

    if as_json:
        payload = {"regime": result.model_dump(mode="json")}
        if risk is not None:
            payload["transition_risk"] = risk.model_dump(mode="json")
        console.print_json(json.dumps(payload, default=str))
        return

    # Pretty render
    header = (
        f"[bold cyan]{symbol}[/bold cyan] @ {result.as_of:%Y-%m-%d %H:%M} ({interval})\n\n"
        f"[bold]{result.label.value.upper()}[/bold]  "
        f"confidence [bold green]{result.confidence:.0%}[/bold green]  | "
        f"ann vol {result.realized_vol:.1%}  | trend Sharpe {result.trend_strength:+.2f}"
    )
    console.print(Panel(header, title="Regime", border_style="cyan"))

    t = Table(title="Probability distribution", show_header=True, header_style="bold")
    t.add_column("Regime")
    t.add_column("Probability", justify="right")
    for lbl, p in sorted(result.probabilities.items(), key=lambda kv: -kv[1]):
        t.add_row(lbl.value, f"{p:.0%}")
    console.print(t)

    rt = Table(title="Why this label", show_header=True, header_style="bold")
    rt.add_column("Factor")
    rt.add_column("Value")
    rt.add_column("Weight", justify="right")
    rt.add_column("Note")
    for reason in result.reasons:
        rt.add_row(reason.factor, str(reason.value), f"{reason.contribution:.2f}", reason.note)
    console.print(rt)

    vt = Table(title="Method votes", show_header=True, header_style="bold")
    vt.add_column("Method")
    vt.add_column("Label")
    for method, label in result.method_votes.items():
        vt.add_row(method, label.value)
    console.print(vt)

    if risk is not None:
        color = "red" if risk.crossed_threshold else "yellow" if risk.risk_score > 0.4 else "green"
        console.print(
            Panel(
                f"[bold {color}]risk {risk.risk_score:.0%}[/bold {color}]  "
                f"(eig drift {risk.eigenvalue_drift:.2f}, "
                f"gap collapse {risk.spectral_gap_collapse:.3f}, "
                f"vol z {risk.vol_acceleration:+.2f})\n\n{risk.note}",
                title="Transition risk",
                border_style=color,
            )
        )
