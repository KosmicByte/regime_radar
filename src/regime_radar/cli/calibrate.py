"""`regime calibrate` — fit the calibration artifact and report calibration quality.

Harvests detections across the synthetic battery, fits temperature scaling, the conformal
threshold, and the OOD reference, then writes the artifact to `settings.calibration_artifact`.
Prints before/after calibration quality so you can see the improvement at a glance.

The artifact is fit on *synthetic* data for now and is marked as such; re-running this command
on real data later (once it is wired in) refreshes it with no code change.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from regime_radar.calibration.fit import fit_calibration
from regime_radar.config import get_settings

app = typer.Typer(invoke_without_command=True)
console = Console()


@app.callback(invoke_without_command=True)
def main(
    window: int = typer.Option(252, "--window", "-w", help="Walk-forward window (bars)."),
    step: int = typer.Option(21, "--step", help="Stride between harvested windows."),
    edmd_rank: int = typer.Option(10, "--edmd-rank"),
    hmm_states: int = typer.Option(3, "--hmm-states"),
    alpha: float = typer.Option(0.1, "--alpha", help="Miscoverage rate; coverage target = 1 - alpha."),
    out: str = typer.Option(None, "--out", "-o", help="Artifact path (default: settings)."),
) -> None:
    """Fit and persist the calibration artifact from the synthetic battery."""
    settings = get_settings()
    target = out or str(settings.calibration_artifact)

    console.print(
        Panel(
            f"window={window} step={step} alpha={alpha} (coverage target {1 - alpha:.0%})\n"
            f"Harvesting detections across the synthetic battery and fitting…",
            title="RegimeRadar calibrate",
            border_style="cyan",
        )
    )

    artifact, diag = fit_calibration(
        window=window,
        step=step,
        edmd_rank=edmd_rank,
        hmm_n_states=hmm_states,
        alpha=alpha,
    )
    artifact.save(target)

    t = Table(title="Calibration quality", show_header=True, header_style="bold")
    t.add_column("Metric")
    t.add_column("Raw", justify="right")
    t.add_column("Calibrated", justify="right")
    t.add_row("ECE (lower better)", f"{diag['ece_raw']:.3f}", f"{diag['ece_calibrated']:.3f}")
    t.add_row("Brier (lower better)", f"{diag['brier_raw']:.3f}", f"{diag['brier_calibrated']:.3f}")
    console.print(t)

    cov, tgt = diag["coverage"], diag["coverage_target"]
    cov_colour = "green" if cov >= tgt else "yellow"
    console.print(
        Panel(
            f"temperature T = [bold]{diag['temperature']:.3f}[/bold]   "
            f"(T>1 means the raw model was over-confident)\n"
            f"conformal coverage = [bold {cov_colour}]{cov:.1%}[/bold {cov_colour}] "
            f"(target {tgt:.0%})   OOD flag rate = {diag['ood_flag_rate']:.1%}\n"
            f"harvested {diag['n_samples']} windows   fit_source = [bold]{artifact.fit_source}[/bold]",
            title="Summary",
            border_style="cyan",
        )
    )
    if artifact.is_synthetic_fit():
        console.print(
            "[yellow]Note:[/yellow] fit on synthetic data — coverage/OOD are indicative, not "
            "real-market guarantees. Re-run on real data once it is wired in."
        )
    console.print(f"[green]Wrote[/green] {target}")
