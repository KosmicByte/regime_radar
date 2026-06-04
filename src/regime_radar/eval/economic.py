"""Economic validation — does trading the regime actually have an edge?

Statistical accuracy ("we label regimes correctly 76% of the time") and economic value ("a
strategy that acts on those labels makes risk-adjusted money") are *different claims*. A
detector can be accurate and still worthless to trade, and vice-versa. This module tests the
second claim directly, and honestly.

The method:

1. **Walk forward, point-in-time.** At each decision bar we detect the regime from data up to
   that bar only (via the R0 contract — no look-ahead), map the label to a position, and hold
   it until the next decision. Returns are realised one bar later, so the position is always
   decided before the return it earns.
2. **Charge costs.** Every change in position pays a transaction cost in basis points. Without
   this, a whippy signal looks free; with it, churn is correctly penalised — which is the
   whole point of also tracking stability.
3. **Compare to a shuffled-regime null.** The killer test. We circularly shift the position
   series against the returns many times: this preserves the position series' own composition
   and autocorrelation but destroys its *timing* alignment with the market. If the real
   strategy's Sharpe sits in the top tail of that null distribution, the regime has genuine
   timing skill; if not, the apparent edge is an artifact of the position mix. The p-value is
   the fraction of null Sharpes at least as good as the real one.

This is decision *evidence*, not a trade recommendation: it tells you whether a transparent
regime-driven rule would have had an edge on a symbol's history, with a significance test and
costs — the prerequisite to taking any regime-based strategy (including options) seriously.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_radar.core.contract import PointInTimeFrame
from regime_radar.core.regime import detect_regime
from regime_radar.models import RegimeLabel

TRADING_DAYS = 252

# Transparent default mapping from regime → net position (fraction of capital, long positive).
# Deliberately simple and explainable; override via `position_map`. Mean-reversion and chop sit
# flat because acting on them needs a separate entry signal this gate intentionally does not
# assume.
DEFAULT_POSITION_MAP: dict[RegimeLabel, float] = {
    RegimeLabel.TRENDING_UP: 1.0,
    RegimeLabel.BREAKOUT: 1.0,
    RegimeLabel.TRENDING_DOWN: -1.0,
    RegimeLabel.LOW_VOL_GRIND: 0.3,
    RegimeLabel.MEAN_REVERTING: 0.0,
    RegimeLabel.HIGH_VOL_CHOP: 0.0,
    RegimeLabel.UNKNOWN: 0.0,
}


@dataclass
class BacktestResult:
    """Per-bar series produced by walking the regime over a price history."""

    returns: np.ndarray  # strategy net returns per bar (after costs)
    buyhold_returns: np.ndarray  # simple returns of the underlying
    positions: np.ndarray  # held position per bar
    labels: list[RegimeLabel]  # regime label active at each bar
    equity: np.ndarray  # strategy equity curve (starts at 1.0)


@dataclass
class EconomicSummary:
    """Risk/return summary plus the permutation significance test."""

    n_bars: int
    sharpe: float
    buyhold_sharpe: float
    ann_return: float
    buyhold_ann_return: float
    max_drawdown: float
    turnover: float  # average absolute position change per bar
    permutation_p_value: float  # P(null Sharpe >= strategy Sharpe)
    n_permutations: int

    def beats_null(self, level: float = 0.05) -> bool:
        """True when the strategy's edge is significant at ``level`` against the shuffled null."""
        return self.permutation_p_value <= level


def _sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    if len(r) == 0 or not np.all(np.isfinite(r)):
        r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    val = r.mean() / sd * np.sqrt(TRADING_DAYS)
    return float(val) if np.isfinite(val) else 0.0


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0)) if len(equity) else 0.0


def run_regime_backtest(
    close: np.ndarray,
    timestamps: pd.Series | None = None,
    *,
    window: int = 126,
    stride: int = 5,
    edmd_rank: int = 10,
    hmm_n_states: int = 3,
    cost_bps: float = 5.0,
    position_map: dict[RegimeLabel, float] | None = None,
) -> BacktestResult:
    """Walk the detector over a price history and realise a regime-driven strategy.

    The regime is recomputed every ``stride`` bars (re-detecting on every bar is rarely
    realistic and expensive); the position is held between recomputes. Each position is set
    using data up to the decision bar only, then earns the *next* bar's return.
    """
    close = np.asarray(close, dtype=float)
    pmap = position_map or DEFAULT_POSITION_MAP
    cost = cost_bps / 1e4
    n = len(close)
    if n <= window + 1:
        raise ValueError(f"Need more than window+1={window + 1} bars; got {n}.")

    # Real feeds have the occasional gap / missing close (yfinance returns a NaN bar). Forward-
    # fill carries the last valid price (point-in-time safe — uses only past data) so neither the
    # detector window nor the return series ever sees a NaN; bfill covers a leading gap before the
    # first valid bar, well before any trading decision at `window`.
    if np.isnan(close).any():
        if np.isnan(close).all():
            raise ValueError("close series is entirely NaN.")
        close = pd.Series(close).ffill().bfill().to_numpy()

    simple_ret = np.zeros(n)
    simple_ret[1:] = close[1:] / close[:-1] - 1.0
    # Final guard against any residual non-finite value poisoning mean()/std().
    simple_ret = np.nan_to_num(simple_ret, nan=0.0, posinf=0.0, neginf=0.0)

    positions = np.zeros(n)
    labels: list[RegimeLabel] = [RegimeLabel.UNKNOWN] * n
    current_pos = 0.0
    current_label = RegimeLabel.UNKNOWN

    for t in range(window, n):
        if (t - window) % stride == 0:
            frame = PointInTimeFrame.from_arrays(
                close[: t + 1],
                timestamps.iloc[: t + 1] if timestamps is not None else None,
                symbol="backtest",
            )
            try:
                res = detect_regime(
                    frame=frame,
                    edmd_rank=min(edmd_rank, window // 3),
                    hmm_n_states=hmm_n_states,
                    calibrate=False,  # label is unchanged by calibration; skip the artifact
                )
                current_label = res.label
                current_pos = pmap.get(res.label, 0.0)
            except (ValueError, np.linalg.LinAlgError):
                pass  # keep prior position on a failed fit
        positions[t] = current_pos
        labels[t] = current_label

    # Position decided at t-1 earns the return from t-1 to t; charge cost on position changes.
    held = np.roll(positions, 1)
    held[0] = 0.0
    turnover_series = np.abs(np.diff(np.concatenate([[0.0], held])))
    strat_ret = held * simple_ret - turnover_series * cost

    active = np.arange(n) > window
    equity = np.cumprod(1.0 + strat_ret[active])
    return BacktestResult(
        returns=strat_ret[active],
        buyhold_returns=simple_ret[active],
        positions=held[active],
        labels=[labels[i] for i in range(n) if active[i]],
        equity=equity,
    )


def evaluate_economic(
    bt: BacktestResult, *, n_permutations: int = 1000, seed: int = 0
) -> EconomicSummary:
    """Summarise risk/return and test timing skill against a circular-shift null."""
    rng = np.random.default_rng(seed)
    strat = bt.returns
    held = bt.positions
    underlying = bt.buyhold_returns
    actual_sharpe = _sharpe(strat)

    # Null: rotate the position series against returns, preserving its own structure but
    # destroying timing alignment. Costs are not re-applied (we test directional timing).
    m = len(held)
    null_sharpes = np.empty(n_permutations)
    for i in range(n_permutations):
        k = int(rng.integers(1, m)) if m > 1 else 0
        shifted = np.roll(held, k)
        null_sharpes[i] = _sharpe(shifted * underlying)
    p_value = float((1 + np.sum(null_sharpes >= actual_sharpe)) / (n_permutations + 1))

    return EconomicSummary(
        n_bars=int(len(strat)),
        sharpe=actual_sharpe,
        buyhold_sharpe=_sharpe(underlying),
        ann_return=float(np.mean(strat) * TRADING_DAYS),
        buyhold_ann_return=float(np.mean(underlying) * TRADING_DAYS),
        max_drawdown=_max_drawdown(bt.equity),
        turnover=float(np.mean(np.abs(np.diff(np.concatenate([[0.0], held]))))),
        permutation_p_value=p_value,
        n_permutations=n_permutations,
    )
