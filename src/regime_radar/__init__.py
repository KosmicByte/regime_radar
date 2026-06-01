"""RegimeRadar — explainable market regime detection.

Public API:
    detect(prices, ...)         — one-shot regime detection
    explain(result)             — human-readable explanation
    scan(symbols, ...)          — regime across a watchlist
    walk_forward_eval(...)      — honest out-of-sample evaluation
"""

from regime_radar.models import (
    RegimeLabel,
    RegimeResult,
    KoopmanModes,
    TransitionRisk,
)

__version__ = "0.1.0"
__all__ = ["RegimeLabel", "RegimeResult", "KoopmanModes", "TransitionRisk"]
