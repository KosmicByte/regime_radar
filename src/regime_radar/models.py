"""Domain models. Every label produced by the system flows through these types."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RegimeLabel(str, Enum):
    """The taxonomy of regimes we classify.

    Designed to be richer than 'bull/bear' but still interpretable. Each label maps
    to a structural signature in the Koopman spectrum AND a statistical signature
    in returns/vol (see core/regime.py for the mapping rules).
    """

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOL_CHOP = "high_vol_chop"
    LOW_VOL_GRIND = "low_vol_grind"
    BREAKOUT = "breakout"
    UNKNOWN = "unknown"


class KoopmanModes(BaseModel):
    """The Koopman / DMD spectrum: eigenvalues + their interpretation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    eigenvalues: list[complex] = Field(description="Discrete-time Koopman eigenvalues λ_i.")
    growth_rates: list[float] = Field(description="Re(log λ_i)/Δt — decay/growth per unit time.")
    frequencies: list[float] = Field(description="Im(log λ_i)/(2πΔt) — oscillation cycles per unit time.")
    mode_energies: list[float] = Field(description="Relative energy of each mode (sums to 1).")
    rank: int = Field(description="Effective rank after SVD truncation.")
    spectral_gap: float = Field(description="|λ_1| - |λ_2| — large gap = clear dominant mode.")

    @field_validator("eigenvalues", mode="before")
    @classmethod
    def _accept_numpy(cls, v: object) -> list[complex]:
        if isinstance(v, np.ndarray):
            return [complex(x) for x in v.tolist()]
        return v  # type: ignore[return-value]


class RegimeReason(BaseModel):
    """One human-readable reason a label was assigned."""

    factor: str = Field(description="Short name: e.g. 'dominant_eigenvalue', 'realized_vol'.")
    value: float | str = Field(description="Numeric or categorical value of the factor.")
    contribution: float = Field(description="Weight 0..1 of this factor toward the final label.")
    note: str = Field(description="One-line plain-English explanation.")


class RegimeResult(BaseModel):
    """The output of a single regime-detection run."""

    symbol: str
    interval: str
    as_of: datetime = Field(description="Timestamp of the last bar used.")

    label: RegimeLabel
    confidence: float = Field(ge=0.0, le=1.0, description="Calibrated probability of the label.")
    probabilities: dict[RegimeLabel, float] = Field(
        description="Full distribution over labels — sums to ~1.",
    )

    reasons: list[RegimeReason] = Field(description="Why this label, ordered by contribution.")
    modes: KoopmanModes
    method_votes: dict[str, RegimeLabel] = Field(
        description="Label from each method in the ensemble (edmd, hmm, vol_trend...).",
    )

    realized_vol: float = Field(description="Annualised σ over the analysis window.")
    trend_strength: float = Field(description="Standardised drift, [-1, 1] approximately.")


class TransitionRisk(BaseModel):
    """A scalar 0..1 risk that the current regime is about to change."""

    symbol: str
    interval: str
    as_of: datetime

    risk_score: float = Field(ge=0.0, le=1.0)
    crossed_threshold: bool

    eigenvalue_drift: float = Field(description="L2 distance between current and recent λ vectors.")
    spectral_gap_collapse: float = Field(description="Drop in spectral gap vs. baseline.")
    vol_acceleration: float = Field(description="Recent change in realized vol z-score.")

    note: str = Field(description="One-line summary the user can act on.")
