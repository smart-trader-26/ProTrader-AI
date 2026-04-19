"""
Unit tests for A5 macro feature extraction.

Network path is smoke-tested once under the `network` marker; all transform
tests use a hand-built close-price frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.macro import extract_macro_features, fetch_macro_series


def _prices(start: str, n: int, drift: float = 1.001) -> pd.Series:
    idx = pd.date_range(start, periods=n, freq="B")
    return pd.Series([100 * (drift ** i) for i in range(n)], index=idx)


def test_extract_features_on_empty_input():
    assert extract_macro_features(pd.DataFrame()).empty
    assert extract_macro_features(None).empty


def test_extract_features_produces_1d_and_5d_log_returns():
    df = pd.DataFrame(
        {
            "USDINR_Close": _prices("2026-01-01", 10, drift=1.002),
            "Crude_Close":  _prices("2026-01-01", 10, drift=0.998),
        }
    )
    out = extract_macro_features(df)

    assert {"USDINR_1d_chg", "USDINR_5d_chg", "Crude_1d_chg", "Crude_5d_chg"}.issubset(out.columns)
    # First row's 1d return is NaN → filled with 0.0
    assert out.iloc[0]["USDINR_1d_chg"] == 0.0
    # Day-3 vs day-2 1d return for USDINR ≈ log(1.002)
    assert out.iloc[3]["USDINR_1d_chg"] == pytest.approx(np.log(1.002), abs=1e-6)
    # Crude drifts down → 1d return negative
    assert out.iloc[3]["Crude_1d_chg"] < 0


def test_extract_features_clips_extreme_returns():
    """Outliers get clipped at ±0.2 (1d) / ±0.4 (5d) so a bad tick can't poison the model."""
    idx = pd.date_range("2026-01-01", periods=6, freq="B")
    # 10× jump day 2 → log(10) = 2.3 → must be clipped to 0.2
    df = pd.DataFrame(
        {"Gold_Close": [100, 100, 1000, 1000, 1000, 1000]}, index=idx
    )
    out = extract_macro_features(df)
    assert out["Gold_1d_chg"].max() <= 0.2 + 1e-9
    assert out["Gold_1d_chg"].min() >= -0.2 - 1e-9
    assert out["Gold_5d_chg"].max() <= 0.4 + 1e-9


def test_extract_features_ignores_non_close_columns():
    df = pd.DataFrame(
        {
            "Gold_Close": _prices("2026-01-01", 8),
            "Gold_Volume": [0] * 8,  # should be ignored
            "noise":      [1] * 8,   # should be ignored
        }
    )
    out = extract_macro_features(df)
    assert set(out.columns) == {"Gold_1d_chg", "Gold_5d_chg"}


@pytest.mark.network
@pytest.mark.slow
def test_fetch_macro_series_smoke():
    """Real yfinance pull — tolerant of rate limits."""
    import datetime as dt

    end = dt.date.today()
    start = end - dt.timedelta(days=30)
    df = fetch_macro_series(start, end, tickers={"SP500": "^GSPC"})
    if df.empty:
        pytest.skip("yfinance unreachable / empty")
    assert "SP500_Close" in df.columns
    assert (df["SP500_Close"] > 0).all()
