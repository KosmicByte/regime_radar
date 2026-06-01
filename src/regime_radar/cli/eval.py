"""`regime eval` — run the synthetic benchmark and print the headline accuracy."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from regime_radar.config import get_settings
from regime_radar.eval.metrics import benchmark
from regime_radar.viz.plots import plot_confusion

app = typer.Typer(invoke_without_command=True)
console = Console()


@app.callback(invoke_without_command=True)
def main(
    window: int = typer.Option(126, "--window", "-w"),
    step: int = typer.Option(21, "--step"),
    edmd_rank: int = typer.Option(10, "--edmd-rank"),
    hmm_states: int = typer.Option(3, "--hmm-states"),
    grouped: bool = typer.Option(
        True,
        "--grouped/--strict",
        help="Grouped mode folds BREAKOUT into TRENDING_UP for less-strict scoring.",
    ),
    plot_confusion_matrix: bool = typer.Option(
        False, "--plot", help="Write a confusion-matrix PNG to artifacts_dir."
    ),
) -> None:
    """Run the synthetic regime benchmark and print accuracy by scenario."""
    settings = get_settings()
    console.print(
        Panel(
            f"window={window}  step={step}  edmd_rank={edmd_rank}  hmm_states={hmm_states}  "
            f"grouped={grouped}\nRunning benchmark on default synthetic battery...",
            title="RegimeRadar benchmark",
            border_style="cyan",
        )
    )

    result = benchmark(
        window=window,
        step=step,
        edmd_rank=edmd_rank,
        hmm_n_states=hmm_states,
        grouped=grouped,
    )

    t = Table(title="Per-scenario accuracy", show_header=True, header_style="bold")
    t.add_column("Scenario")
    t.add_column("Windows", justify="right")
    t.add_column("Accuracy", justify="right")
    for name, wf in result.per_scenario.items():
        t.add_row(name, str(len(wf.predicted)), f"{wf.accuracy:.0%}")
    console.print(t)

    overall = result.overall_accuracy
    colour = "green" if overall >= 0.70 else "yellow" if overall >= 0.55 else "red"
    console.print(
        Panel(
            f"[bold {colour}]{overall:.1%}[/bold {colour}] overall accuracy "
            f"({sum(len(r.predicted) for r in result.per_scenario.values())} total windows)",
            title="Headline",
            border_style=colour,
        )
    )

    m, labels = result.confusion()
    label_names = [lbl.value for lbl in labels]
    ct = Table(title="Confusion matrix (rows=true, cols=predicted)", show_header=True)
    ct.add_column("true ↓ / pred →")
    for nm in label_names:
        ct.add_column(nm, justify="right")
    for i, nm in enumerate(label_names):
        ct.add_row(nm, *[str(int(v)) for v in m[i]])
    console.print(ct)

    if plot_confusion_matrix:
        out = Path(settings.artifacts_dir) / "confusion.png"
        plot_confusion(m, label_names, out, title=f"Confusion (overall acc {overall:.0%})")
        console.print(f"[green]Wrote[/green] {out}")
