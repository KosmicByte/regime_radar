"""`regime eval` — run the synthetic benchmark and print the headline accuracy."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from regime_radar.config import get_settings
from regime_radar.eval.metrics import ab_dampening, benchmark
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
    ab_hmm_dampening: bool = typer.Option(
        False,
        "--ab-dampening",
        help="A/B the hand-tuned HMM dampening ladder (on vs off) and report the delta with CIs.",
    ),
) -> None:
    """Run the synthetic regime benchmark and print accuracy by scenario."""
    settings = get_settings()

    if ab_hmm_dampening:
        _run_ab(window, step, edmd_rank, hmm_states, grouped)
        return
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
    acc, lo, hi = result.accuracy_ci()
    macro_f1, _per_f1 = result.macro_f1()
    stab = result.stability_summary()
    colour = "green" if overall >= 0.70 else "yellow" if overall >= 0.55 else "red"
    total_windows = sum(len(r.predicted) for r in result.per_scenario.values())
    console.print(
        Panel(
            f"[bold {colour}]{overall:.1%}[/bold {colour}] overall accuracy  "
            f"[dim]95% CI [{lo:.1%}, {hi:.1%}][/dim]  ({total_windows} windows)\n"
            f"macro-F1 = {macro_f1:.3f}   "
            f"label stability: whipsaw {stab.whipsaw_rate:.1%}, mean dwell {stab.mean_dwell:.0f} windows",
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


def _run_ab(window: int, step: int, edmd_rank: int, hmm_states: int, grouped: bool) -> None:
    """Run and report the HMM-dampening A/B (ladder on vs off)."""
    console.print(
        Panel(
            "Running the benchmark twice on the same battery: HMM dampening ladder ON vs OFF.\n"
            "This decides whether the hand-tuned heuristics still earn their place.",
            title="RegimeRadar A/B — HMM dampening",
            border_style="cyan",
        )
    )
    ab = ab_dampening(
        window=window, step=step, edmd_rank=edmd_rank, hmm_n_states=hmm_states, grouped=grouped
    )
    s = ab.summary()

    t = Table(title="Dampening ladder: ON vs OFF", header_style="bold")
    t.add_column("Config")
    t.add_column("Accuracy", justify="right")
    t.add_column("95% CI", justify="right")
    t.add_column("macro-F1", justify="right")
    t.add_row("ON (current default)", f"{s['on_accuracy']:.1%}", f"[{s['on_ci'][0]:.1%}, {s['on_ci'][1]:.1%}]", f"{s['on_macro_f1']:.3f}")
    t.add_row("OFF (calibration only)", f"{s['off_accuracy']:.1%}", f"[{s['off_ci'][0]:.1%}, {s['off_ci'][1]:.1%}]", f"{s['off_macro_f1']:.3f}")
    console.print(t)

    delta = s["delta_mean"]
    d_lo, d_hi = s["delta_ci"]
    divergence = s["label_divergence"]

    if not s["exercised"]:
        # The ladder changed no labels on this battery — the A/B literally could not test it.
        verdict = (
            f"the dampening ladder changed [bold]no labels[/bold] on this battery "
            f"(label divergence {divergence:.1%}), so this benchmark cannot exercise it. A zero "
            f"delta here means [bold]untested[/bold], not safe to remove — the ladder targets "
            f"real-data disagreement cases the synthetic generators don't reproduce. "
            f"[bold]Keep it on[/bold]; re-run this A/B on real labelled data to decide."
        )
        colour = "cyan"
    elif s["can_retire"]:
        verdict = (
            f"the ladder was exercised (changed labels in {divergence:.1%} of windows) and OFF is "
            f"non-inferior (delta CI lower bound {d_lo:+.1%} ≥ −{s['margin']:.0%}). It can be "
            f"retired — calibration carries the load at no measurable accuracy cost."
        )
        colour = "green"
    else:
        verdict = (
            f"the ladder was exercised ({divergence:.1%} of windows) and OFF loses more than the "
            f"{s['margin']:.0%} margin (delta CI lower bound {d_lo:+.1%}). [bold]Keep it[/bold] — "
            f"calibration does not yet replace it."
        )
        colour = "yellow"
    console.print(
        Panel(
            f"paired delta (OFF − ON) = [bold]{delta:+.1%}[/bold]  "
            f"95% CI [{d_lo:+.1%}, {d_hi:+.1%}]   label divergence = {divergence:.1%}\n"
            f"[{colour}]{verdict}[/{colour}]",
            title="Verdict",
            border_style=colour,
        )
    )
