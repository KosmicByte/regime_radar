"""CLI entry point. Wires sub-commands from regime_radar.cli."""

from __future__ import annotations

import typer

from regime_radar.cli import detect, eval as eval_cmd, explain, plot, scan

app = typer.Typer(
    name="regime",
    help="Explainable market regime detection (Koopman/DMD + HMM ensemble).",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

app.add_typer(detect.app, name="detect", help="Detect the current regime for a symbol.")
app.add_typer(explain.app, name="explain", help="Show the reasoning behind the latest label.")
app.add_typer(scan.app, name="scan", help="Run regime detection across a watchlist.")
app.add_typer(plot.app, name="plot", help="Plot eigenvalue spectrum, regime timeline, risk.")
app.add_typer(eval_cmd.app, name="eval", help="Evaluate on synthetic ground truth (walk-forward).")


if __name__ == "__main__":
    app()
