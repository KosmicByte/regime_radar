"""`regime explain` — detailed eigenvalue breakdown for the latest detection."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from regime_radar.config import get_settings
from regime_radar.core.regime import detect_regime
from regime_radar.data import load

app = typer.Typer(invoke_without_command=True)
console = Console()


@app.callback(invoke_without_command=True)
def main(
    symbol: str = typer.Option(None, "--symbol", "-s"),
    interval: str = typer.Option(None, "--interval", "-i"),
    lookback_days: int = typer.Option(720, "--lookback"),
    top_k: int = typer.Option(5, "--top-k", help="Show top K eigenvalues."),
) -> None:
    """Print the eigenvalue spectrum + reasons that produced the latest regime label."""
    settings = get_settings()
    symbol = symbol or settings.default_symbol
    interval = interval or settings.default_interval

    df = load(symbol=symbol, interval=interval, lookback_days=lookback_days)
    close = df["close"].to_numpy()
    ts = df["ts"]

    window = settings.window
    result = detect_regime(
        close=close[-(window + 5) :],
        timestamps=ts.iloc[-(window + 5) :],
        symbol=symbol,
        interval=interval,
        edmd_rank=settings.edmd_rank,
        hmm_n_states=settings.hmm_n_states,
    )

    console.print(f"[bold]{symbol}[/bold] — label [cyan]{result.label.value}[/cyan]")
    console.print(f"as of {result.as_of:%Y-%m-%d %H:%M}\n")

    et = Table(title=f"Top {top_k} Koopman eigenvalues", show_header=True)
    et.add_column("#")
    et.add_column("λ (Re, Im)")
    et.add_column("|λ|", justify="right")
    et.add_column("arg(λ) rad", justify="right")
    et.add_column("growth/bar", justify="right")
    et.add_column("freq (cyc/bar)", justify="right")
    et.add_column("energy", justify="right")

    n = min(top_k, len(result.modes.eigenvalues))
    for i in range(n):
        lam = result.modes.eigenvalues[i]
        gr = result.modes.growth_rates[i]
        fr = result.modes.frequencies[i]
        en = result.modes.mode_energies[i]
        mag = abs(lam)
        # arg in [-π, π]
        import math
        arg = math.atan2(lam.imag, lam.real)
        et.add_row(
            str(i + 1),
            f"({lam.real:+.4f}, {lam.imag:+.4f}i)",
            f"{mag:.4f}",
            f"{arg:+.3f}",
            f"{gr:+.5f}",
            f"{fr:+.4f}",
            f"{en:.3f}",
        )
    console.print(et)
    console.print(f"[dim]spectral gap |λ₁|-|λ₂| = {result.modes.spectral_gap:.4f}[/dim]\n")

    rt = Table(title="Reasons (ordered by contribution)", show_header=True)
    rt.add_column("Factor")
    rt.add_column("Value")
    rt.add_column("Weight × conf", justify="right")
    rt.add_column("Note")
    for reason in result.reasons:
        rt.add_row(reason.factor, str(reason.value), f"{reason.contribution:.2f}", reason.note)
    console.print(rt)
