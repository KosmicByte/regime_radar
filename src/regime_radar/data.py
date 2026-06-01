"""Single market-data entry point.

Tries MarketLake first (when installed), falls back to yfinance so RegimeRadar runs standalone.
Downstream code should ONLY import from this module — never call yfinance/Upstox directly.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


def _try_marketlake(symbol: str, interval: str, start: date, end: date) -> pd.DataFrame | None:
    """Return a MarketLake dataframe if MarketLake is importable; else None."""
    try:
        import marketlake  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        df = marketlake.read(symbol=symbol, interval=interval, start=start, end=end)
        # MarketLake returns Polars by contract; convert to pandas for internal use.
        if hasattr(df, "to_pandas"):
            df = df.to_pandas()
        logger.info("Loaded %s %s from MarketLake (%d rows).", symbol, interval, len(df))
        return df
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("MarketLake call failed (%s); falling back to yfinance.", exc)
        return None


def _yfinance_fallback(symbol: str, interval: str, start: date, end: date) -> pd.DataFrame:
    """Yahoo Finance fallback. Maps our intervals to yfinance's vocabulary."""
    import yfinance as yf

    interval_map = {"1d": "1d", "1h": "60m", "5m": "5m", "15m": "15m"}
    yf_interval = interval_map.get(interval, "1d")

    df = yf.download(
        symbol,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval=yf_interval,
        progress=False,
        auto_adjust=True,
    )
    if df.empty:
        raise ValueError(f"No data for {symbol} {interval} {start}..{end}")

    # yfinance returns MultiIndex columns when multiple tickers; flatten to single ticker case.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower).reset_index()
    df = df.rename(columns={"date": "ts", "datetime": "ts"})
    df["symbol"] = symbol
    df["interval"] = interval

    keep = ["symbol", "interval", "ts", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def load(
    symbol: str,
    interval: str = "1d",
    start: date | None = None,
    end: date | None = None,
    lookback_days: int = 720,
) -> pd.DataFrame:
    """Load OHLCV for `symbol` over the requested window.

    Args:
        symbol: Yahoo-style ticker (e.g. '^NSEI' for NIFTY50, '^NSEBANK' for BANKNIFTY).
        interval: '1d', '1h', '5m', '15m'.
        start, end: Date range. If omitted, uses today minus `lookback_days`.
        lookback_days: Default lookback if start/end aren't given.

    Returns:
        DataFrame with columns: symbol, interval, ts (datetime), open, high, low, close, volume.
        Sorted by ts ascending. No gap interpolation.
    """
    if end is None:
        end = datetime.now().date()
    if start is None:
        start = end - timedelta(days=lookback_days)

    df = _try_marketlake(symbol, interval, start, end)
    if df is None:
        df = _yfinance_fallback(symbol, interval, start, end)

    df = df.sort_values("ts").reset_index(drop=True)
    return df
