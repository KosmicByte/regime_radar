"""Quickstart: detect the current regime for NIFTY50.

Run with:
    uv run python examples/quickstart.py

Or set REGIME_DEFAULT_SYMBOL=^NSEBANK in .env and just run `regime detect`.
"""

from __future__ import annotations

from regime_radar.core.regime import detect_regime
from regime_radar.core.transition import score_transition_risk
from regime_radar.data import load


def main() -> None:
    # 1. Load 2 years of daily data. MarketLake is tried first; falls back to yfinance.
    df = load(symbol="^NSEI", interval="1d", lookback_days=720)
    print(f"Loaded {len(df)} bars for ^NSEI")

    close = df["close"].to_numpy()
    ts = df["ts"]

    # 2. Run the ensemble detector on the most recent window
    result = detect_regime(
        close=close[-130:],
        timestamps=ts.iloc[-130:],
        symbol="^NSEI",
        interval="1d",
    )

    # 3. Print the result
    print(f"\nCurrent regime: {result.label.value}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Annualised vol: {result.realized_vol:.1%}")
    print(f"Trend Sharpe: {result.trend_strength:+.2f}")
    print(f"\nMethod votes:")
    for method, label in result.method_votes.items():
        print(f"  {method:8s}: {label.value}")
    print(f"\nWhy this label (ordered by contribution):")
    for reason in result.reasons:
        print(f"  [{reason.factor}] {reason.note}")

    # 4. Transition risk
    risk = score_transition_risk(close=close, timestamps=ts, symbol="^NSEI", window=126)
    print(f"\nTransition risk: {risk.risk_score:.0%}")
    print(f"  {risk.note}")


if __name__ == "__main__":
    main()
