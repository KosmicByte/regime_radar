"""Tests for the R0 point-in-time contract, provenance, and the no-look-ahead guarantee.

The headline test is :func:`test_future_bars_do_not_leak`: appending future bars to the
series must leave a past-``as_of`` detection byte-identical. If that ever fails, a backtest
built on this detector is no longer trustworthy.
"""

from __future__ import annotations

import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from regime_radar.core.contract import PointInTimeFrame
from regime_radar.core.regime import detect_regime
from regime_radar.eval.synthetic import gbm_trending
from regime_radar.provenance import InferenceRecord, read_records, write_record
from regime_radar.version import MODEL_VERSION, code_version

warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture
def series() -> tuple[np.ndarray, pd.Series]:
    close = gbm_trending(n=200, direction="up", seed=0).prices
    ts = pd.Series(pd.date_range("2024-01-01", periods=len(close), freq="D"))
    return close, ts


# --------------------------------------------------------------------- contract
def test_frame_truncates_to_as_of(series: tuple[np.ndarray, pd.Series]) -> None:
    close, ts = series
    as_of = ts.iloc[120].to_pydatetime()
    frame = PointInTimeFrame.from_arrays(close, ts, as_of=as_of)
    assert frame.n == 121
    assert frame.last_ts <= as_of
    assert frame.timestamps.max().to_pydatetime() <= as_of


def test_default_as_of_is_last_bar(series: tuple[np.ndarray, pd.Series]) -> None:
    close, ts = series
    frame = PointInTimeFrame.from_arrays(close, ts)
    assert frame.n == len(close)
    assert frame.last_ts == ts.iloc[-1].to_pydatetime()


def test_tail_is_point_in_time_safe(series: tuple[np.ndarray, pd.Series]) -> None:
    close, ts = series
    as_of = ts.iloc[120].to_pydatetime()
    frame = PointInTimeFrame.from_arrays(close, ts, as_of=as_of)
    tail = frame.tail(20)
    assert tail.n == 20
    assert tail.as_of == frame.as_of
    assert tail.timestamps.max() <= pd.Timestamp(as_of)


def test_input_hash_deterministic_and_sensitive(
    series: tuple[np.ndarray, pd.Series],
) -> None:
    close, ts = series
    a = PointInTimeFrame.from_arrays(close, ts).input_hash()
    b = PointInTimeFrame.from_arrays(close, ts).input_hash()
    assert a == b and len(a) == 16
    perturbed = close.copy()
    perturbed[0] += 1e-6
    assert PointInTimeFrame.from_arrays(perturbed, ts).input_hash() != a


def test_from_dataframe_reads_metadata(series: tuple[np.ndarray, pd.Series]) -> None:
    close, ts = series
    df = pd.DataFrame({"symbol": "^NSEI", "interval": "1d", "ts": ts, "close": close})
    frame = PointInTimeFrame.from_dataframe(df)
    assert frame.symbol == "^NSEI" and frame.interval == "1d" and frame.n == len(close)


def test_empty_after_truncation_raises(series: tuple[np.ndarray, pd.Series]) -> None:
    close, ts = series
    too_early = ts.iloc[0].to_pydatetime() - timedelta(days=5)
    with pytest.raises(ValueError, match="No bars at or before"):
        PointInTimeFrame.from_arrays(close, ts, as_of=too_early)


def test_with_exogenous_is_documented_stub(series: tuple[np.ndarray, pd.Series]) -> None:
    close, ts = series
    with pytest.raises(NotImplementedError):
        PointInTimeFrame.from_arrays(close, ts).with_exogenous()


# --------------------------------------------------------------------- detection wiring
def test_detect_stamps_provenance_fields(series: tuple[np.ndarray, pd.Series]) -> None:
    close, ts = series
    result = detect_regime(close=close, timestamps=ts, symbol="^NSEI", edmd_rank=10)
    assert result.model_version == MODEL_VERSION
    assert result.code_version == code_version()
    assert len(result.input_hash) == 16
    assert result.as_of == ts.iloc[-1].to_pydatetime()


def test_detect_is_deterministic(series: tuple[np.ndarray, pd.Series]) -> None:
    close, ts = series
    a = detect_regime(close=close, timestamps=ts, edmd_rank=10)
    b = detect_regime(close=close, timestamps=ts, edmd_rank=10)
    assert a.input_hash == b.input_hash
    assert a.label == b.label


def test_frame_matches_explicit_slice(series: tuple[np.ndarray, pd.Series]) -> None:
    close, ts = series
    as_of = ts.iloc[120].to_pydatetime()
    via_frame = detect_regime(
        frame=PointInTimeFrame.from_arrays(close, ts, as_of=as_of, symbol="^NSEI"),
        edmd_rank=10,
    )
    via_slice = detect_regime(
        close=close[:121], timestamps=ts.iloc[:121], symbol="^NSEI", edmd_rank=10
    )
    assert via_frame.input_hash == via_slice.input_hash
    assert via_frame.label == via_slice.label


def test_future_bars_do_not_leak(series: tuple[np.ndarray, pd.Series]) -> None:
    """The guarantee: appending future bars must not change a past-as_of decision."""
    close, ts = series
    as_of = ts.iloc[120].to_pydatetime()
    base = detect_regime(
        frame=PointInTimeFrame.from_arrays(close, ts, as_of=as_of, symbol="^NSEI"),
        edmd_rank=10,
    )

    rng = np.random.default_rng(1)
    extra = close[-1] * np.exp(np.cumsum(rng.normal(0.01, 0.02, 50)))
    close_ext = np.concatenate([close, extra])
    ts_ext = pd.Series(pd.date_range("2024-01-01", periods=len(close_ext), freq="D"))

    with_future = detect_regime(
        frame=PointInTimeFrame.from_arrays(close_ext, ts_ext, as_of=as_of, symbol="^NSEI"),
        edmd_rank=10,
    )
    assert with_future.input_hash == base.input_hash
    assert with_future.label == base.label
    assert with_future.confidence == base.confidence


def test_detect_requires_frame_or_close() -> None:
    with pytest.raises(ValueError, match="either `frame=` or `close=`"):
        detect_regime()


# --------------------------------------------------------------------- provenance
def test_provenance_roundtrip(tmp_path) -> None:
    rec = InferenceRecord(
        as_of="2024-05-01T00:00:00",
        symbol="^NSEI",
        interval="1d",
        model_version=MODEL_VERSION,
        code_version=code_version(),
        input_hash="deadbeefdeadbeef",
        label="trending_up",
        confidence=0.62,
        probabilities={"trending_up": 0.62, "mean_reverting": 0.20},
    )
    path = tmp_path / "^NSEI.jsonl"
    write_record(rec, path)
    write_record(rec, path)
    back = read_records(path)
    assert len(back) == 2
    assert back[0].input_hash == "deadbeefdeadbeef"
    assert back[0].label == "trending_up"


def test_provenance_written_when_enabled(
    tmp_path, monkeypatch, series: tuple[np.ndarray, pd.Series]
) -> None:
    monkeypatch.setenv("REGIME_PROVENANCE_ENABLED", "true")
    monkeypatch.setenv("REGIME_PROVENANCE_DIR", str(tmp_path))
    # Settings is cached; clear it so the env override takes effect.
    import regime_radar.config as cfg

    cfg._settings = None  # type: ignore[attr-defined]

    close, ts = series
    detect_regime(close=close, timestamps=ts, symbol="TEST", edmd_rank=10)
    records = read_records(tmp_path / "TEST.jsonl")
    assert len(records) == 1
    assert records[0].model_version == MODEL_VERSION
    cfg._settings = None  # type: ignore[attr-defined]
