"""`regime backtest` — does trading a symbol's regime have an edge?

Walks the detector over a symbol's price history (real data via the usual data path), realises
a transparent regime-driven strategy with transaction costs, and tests it against a
shuffled-regime null. Reports Sharpe vs buy-and-hold, a permutation p-value, and label
stability — the evidence you need before taking a regime-based strategy (including options)
seriously on that symbol.

This is decision *evidence*, not a recommendation or a signal: it tells you whether a simple
regime rule would have had a statistically significant timing edge on the available history.
It is not financial advice, and a positive historical result is not a guarantee of future
performance.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from regime_radar.config import get_settings
from regime_radar.data import load
from regime_radar.eval.economic import (
    DEFAULT_POSITION_MAP,
    evaluate_economic,
    run_regime_backtest,
)
from regime_radar.eval.stability import stability

app = typer.Typer(invoke_without_command=True)
console = Console()


@app.callback(invoke_without_command=True)
def main(
    symbol: str = typer.Option(None, "--symbol", "-s", help="Ticker, e.g. GAIL.NS, ^NSEI."),
    interval: str = typer.Option("1d", "--interval", "-i"),
    lookback_days: int = typer.Option(1460, "--lookback", help="History to pull (days)."),
    window: int = typer.Option(126, "--window", "-w", help="Detection window (bars)."),
    stride: int = typer.Option(5, "--stride", help="Bars between regime recomputes."),
    cost_bps: float = typer.Option(5.0, "--cost-bps", help="Transaction cost per position change."),
    permutations: int = typer.Option(1000, "--permutations", help="Shuffled-regime null draws."),
) -> None:
    """Backtest the regime-driven strategy on a real symbol and report its economic edge."""
    settings = get_settings()
    symbol = symbol or settings.default_symbol

    console.print(
        Panel(
            f"symbol={symbol} interval={interval} lookback={lookback_days}d "
            f"window={window} stride={stride} cost={cost_bps}bps\n"
            f"Loading data, walking the regime point-in-time, and testing vs a shuffled null…",
            title="RegimeRadar backtest",
            border_style="cyan",
        )
    )

    df = load(symbol=symbol, interval=interval, lookback_days=lookback_days)
    if len(df) <= window + 2:
        console.print(f"[red]Not enough data[/red]: got {len(df)} bars, need > {window + 2}.")
        raise typer.Exit(1)

    bt = run_regime_backtest(
        df["close"].to_numpy(),
        df["ts"],
        window=window,
        stride=stride,
        cost_bps=cost_bps,
    )
    summ = evaluate_economic(bt, n_permutations=permutations)
    stab = stability(bt.labels)

    perf = Table(title=f"{symbol} — regime strategy vs buy & hold", header_style="bold")
    perf.add_column("Metric")
    perf.add_column("Regime strategy", justify="right")
    perf.add_column("Buy & hold", justify="right")
    perf.add_row("Annualised return", f"{summ.ann_return:+.1%}", f"{summ.buyhold_ann_return:+.1%}")
    perf.add_row("Sharpe", f"{summ.sharpe:+.2f}", f"{summ.buyhold_sharpe:+.2f}")
    perf.add_row("Max drawdown", f"{summ.max_drawdown:.1%}", "—")
    console.print(perf)

    sig = summ.beats_null()
    colour = "green" if sig else "yellow"
    verdict = (
        "edge is statistically significant vs the shuffled-regime null"
        if sig
        else "NO statistically significant edge over the shuffled-regime null"
    )
    console.print(
        Panel(
            f"permutation p-value = [bold {colour}]{summ.permutation_p_value:.3f}[/bold {colour}] "
            f"({summ.n_permutations} draws) — [bold {colour}]{verdict}[/bold {colour}]\n"
            f"bars tested = {summ.n_bars}   avg turnover = {summ.turnover:.3f}/bar\n"
            f"label stability: whipsaw {stab.whipsaw_rate:.1%}   mean dwell "
            f"{stab.mean_dwell:.0f} bars   switches {stab.n_switches}",
            title="Edge test",
            border_style=colour,
        )
    )

    pos_map = ", ".join(f"{k.value}→{v:+g}" for k, v in DEFAULT_POSITION_MAP.items() if v != 0)
    console.print(
        f"[dim]Position map: {pos_map}; all others flat. Costs {cost_bps}bps/turn. "
        f"Point-in-time (no look-ahead).[/dim]"
    )
    console.print(
        "[yellow]Caveat:[/yellow] decision evidence, not a recommendation or signal. Historical "
        "edge does not guarantee future performance. Not financial advice."
    )
