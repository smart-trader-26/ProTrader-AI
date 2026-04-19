"""
Tests for A3.1 intraday bars + A3.5 intraday-feature extractor.

Network tests for the yfinance puller are behind the `network` marker.
All feature-math tests build a synthetic OHLCV frame locally.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from data.intraday import _clamp_period, _period_days, get_intraday_bars, is_nse_market_open
from data.intraday_features import (
    _opening_range_breakout,
    _rsi,
    _session_vwap,
    extract_intraday_features,
)


def _synthetic_5min(n: int = 60, start_price: float = 100.0, drift: float = 1.0001) -> pd.DataFrame:
    idx = pd.date_range("2026-04-17 09:15", periods=n, freq="5min")
    close = [start_price * (drift ** i) for i in range(n)]
    return pd.DataFrame(
        {
            "Open":   close,
            "High":   [c * 1.002 for c in close],
            "Low":    [c * 0.998 for c in close],
            "Close":  close,
            "Volume": [1000 + i for i in range(n)],
        },
        index=idx,
    )


def test_period_clamp_picks_stricter():
    assert _clamp_period("60d", "5d") == "5d"
    assert _clamp_period("2d", "5d") == "2d"
    assert _period_days("2y") == 730
    assert _period_days("max") == 99999


def test_empty_frame_produces_zero_defaults():
    feats = extract_intraday_features(pd.DataFrame())
    assert feats["intraday_available"] == 0.0
    assert feats["rsi_5m"] == 50.0
    assert feats["orb_flag"] == 0.0


def test_rsi_on_uptrend_is_bullish():
    df = _synthetic_5min(n=30, drift=1.002)  # steady uptrend
    rsi = _rsi(df["Close"], period=14)
    # Pure uptrend → RSI must cross above 70
    assert rsi.iloc[-1] > 70


def test_rsi_on_downtrend_is_bearish():
    df = _synthetic_5min(n=30, drift=0.998)
    rsi = _rsi(df["Close"], period=14)
    assert rsi.iloc[-1] < 30


def test_vwap_matches_manual_calc_on_flat_session():
    df = _synthetic_5min(n=10, drift=1.0)  # constant price
    vwap = _session_vwap(df)
    # All bars at the same typical price → VWAP equals that price.
    assert vwap.iloc[-1] == pytest.approx(df["Close"].iloc[-1], rel=1e-4)


def test_orb_flag_triggers_on_breakout_above_opening_range():
    # Build a session where the first 15 min stays [99, 101] and then jumps.
    idx = pd.date_range("2026-04-17 09:15", periods=6, freq="5min")
    df = pd.DataFrame(
        {
            "Open":   [100, 100, 100, 102, 103, 105],
            "High":   [101, 101, 101, 103, 104, 106],
            "Low":    [99,  99,  99,  101, 102, 104],
            "Close":  [100, 100, 100, 103, 104, 106],
            "Volume": [1000] * 6,
        },
        index=idx,
    )
    assert _opening_range_breakout(df, minutes=15) == 1


def test_orb_flag_triggers_on_breakdown_below_opening_range():
    idx = pd.date_range("2026-04-17 09:15", periods=6, freq="5min")
    df = pd.DataFrame(
        {
            "Open":   [100, 100, 100, 98, 96, 95],
            "High":   [101, 101, 101, 99, 97, 96],
            "Low":    [99,  99,  99,  97, 95, 94],
            "Close":  [100, 100, 100, 97, 96, 95],
            "Volume": [1000] * 6,
        },
        index=idx,
    )
    assert _opening_range_breakout(df, minutes=15) == -1


def test_extract_intraday_features_end_to_end():
    df = _synthetic_5min(n=40, drift=1.0015)
    feats = extract_intraday_features(df)
    assert feats["intraday_available"] == 1.0
    assert feats["rsi_5m"] > 50  # uptrend
    assert -0.1 <= feats["vwap_distance"] <= 0.1  # clipped to ±10%


def test_is_nse_market_open_at_various_hours():
    # Monday 10:00 IST → open
    assert is_nse_market_open(datetime(2026, 4, 20, 10, 0)) is True
    # Monday 08:00 IST → pre-open, closed
    assert is_nse_market_open(datetime(2026, 4, 20, 8, 0)) is False
    # Monday 16:00 IST → post-close
    assert is_nse_market_open(datetime(2026, 4, 20, 16, 0)) is False
    # Saturday → closed
    assert is_nse_market_open(datetime(2026, 4, 25, 11, 0)) is False


@pytest.mark.network
@pytest.mark.slow
def test_get_intraday_bars_smoke():
    df = get_intraday_bars("RELIANCE.NS", interval="5m", period="5d")
    if df.empty:
        pytest.skip("yfinance intraday unavailable")
    assert {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns)
    assert df.attrs.get("interval") in {"1m", "2m", "5m", "15m", "30m", "60m"}
