"""Label-stability metrics — the whipsaw tax.

Accuracy says nothing about how *steady* the label is. A detector that flickers between
TRENDING_UP and HIGH_VOL_CHOP every few bars can be more expensive to trade than a slightly
less accurate one that holds a view, because every flip costs transaction money (see
``economic.py``'s cost model). These metrics quantify that steadiness so a stable-but-slightly
-less-accurate configuration can be recognised as the better one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StabilityReport:
    """Summary of how steady a regime-label sequence is."""

    n: int
    whipsaw_rate: float  # fraction of bars where the label changed from the previous bar
    n_switches: int  # total label changes
    mean_dwell: float  # average run length (bars) the label is held
    max_dwell: int  # longest run


def stability(labels: list) -> StabilityReport:
    """Compute stability metrics for a sequence of regime labels.

    ``labels`` may hold RegimeLabel enums or their string values — only equality is used.
    """
    n = len(labels)
    if n == 0:
        return StabilityReport(0, 0.0, 0, 0.0, 0)
    switches = sum(1 for a, b in zip(labels[:-1], labels[1:], strict=False) if a != b)
    # Run lengths.
    runs: list[int] = []
    run = 1
    for a, b in zip(labels[:-1], labels[1:], strict=False):
        if a == b:
            run += 1
        else:
            runs.append(run)
            run = 1
    runs.append(run)
    return StabilityReport(
        n=n,
        whipsaw_rate=float(switches / max(n - 1, 1)),
        n_switches=int(switches),
        mean_dwell=float(np.mean(runs)),
        max_dwell=int(max(runs)),
    )
