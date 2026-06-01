"""`regime plot` — produce spectrum + price-with-label plots to disk."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from regime_radar.config import get_settings
from regime_radar.core.edmd import fit_edmd
from regime_radar.core.regime import detect_regime
from regime_radar.data import load
from regime_radar.viz.plots import plot_price_with_regime, plot_spectrum

app = typer.Typer(invoke_without_command=True)
console = Console()


@app.callback(invoke_without_command=True)
def main(
    symbol: str = typer.Option(None, "--symbol", "-s"),
    interval: str = typer.Option(None, "--interval", "-i"),
    lookback_days: int = typer.Option(720, "--lookback"),
    out_dir: Path = typer.Option(None, "--out-dir", help="Defaults to settings.artifacts_dir."),
) -> None:
    """Emit a Koopman-spectrum plot and a price chart annotated with the regime label."""
    settings = get_settings()
    symbol = symbol or settings.default_symbol
    interval = interval or settings.default_interval
    out_dir = out_dir or settings.artifacts_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(symbol=symbol, interval=interval, lookback_days=lookback_days)
    close = df["close"].to_numpy()
    ts = df["ts"]

    edmd = fit_edmd(close[-(settings.window + 5) :], rank=settings.edmd_rank)
    result = detect_regime(
        close=close[-(settings.window + 5) :],
        timestamps=ts.iloc[-(settings.window + 5) :],
        symbol=symbol,
        interval=interval,
        edmd_rank=settings.edmd_rank,
        hmm_n_states=settings.hmm_n_states,
    )

    safe = symbol.replace("^", "").replace(".", "_")
    spec_path = out_dir / f"{safe}_spectrum.png"
    price_path = out_dir / f"{safe}_price_regime.png"

    plot_spectrum(edmd, spec_path, title=f"{symbol} Koopman spectrum ({interval})")
    plot_price_with_regime(
        ts.iloc[-(settings.window + 5) :],
        close[-(settings.window + 5) :],
        result,
        price_path,
    )

    console.print(f"[green]Wrote[/green] {spec_path}")
    console.print(f"[green]Wrote[/green] {price_path}")
