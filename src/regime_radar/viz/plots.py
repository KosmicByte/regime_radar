"""Matplotlib plots used by the CLI `plot` sub-commands."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regime_radar.core.edmd import EDMDResult
from regime_radar.models import RegimeResult


def plot_spectrum(edmd: EDMDResult, out: Path, title: str = "Koopman spectrum") -> Path:
    """Eigenvalues in the complex plane with the unit circle for reference."""
    fig, ax = plt.subplots(figsize=(6, 6))
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.4, label="unit circle")
    ax.scatter(np.real(edmd.eigenvalues), np.imag(edmd.eigenvalues), c="crimson", s=70, zorder=3)
    for i, lam in enumerate(edmd.eigenvalues[:5]):
        ax.annotate(f"λ{i+1}", (lam.real, lam.imag), fontsize=9, xytext=(5, 5),
                    textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_price_with_regime(
    timestamps: pd.Series,
    close: np.ndarray,
    result: RegimeResult,
    out: Path,
) -> Path:
    """Price chart annotated with the current regime label and confidence."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timestamps, close, lw=1.2, color="navy")
    ax.set_title(
        f"{result.symbol} — {result.label.value} (conf {result.confidence:.0%}) "
        f"| ann vol {result.realized_vol:.1%}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.grid(alpha=0.3)
    # Annotate the top reason
    if result.reasons:
        ax.text(
            0.01, 0.97, result.reasons[0].note,
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gold"),
        )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_confusion(matrix: np.ndarray, labels: list[str], out: Path, title: str) -> Path:
    """Confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                    fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
