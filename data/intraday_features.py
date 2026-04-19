"""
Intraday-aware technical features (A3.5).

Three scalars per ticker, computed from an intraday bar DataFrame:

    rsi_5m         14-bar RSI on 5-min Close — short-horizon momentum
    vwap_distance  (last_close - session_VWAP) / session_VWAP — % above/below
                   today's volume-weighted average. Positive = strength,
                   negative = weakness vs. institutional accumulation zone.
    orb_flag       Opening-range breakout signal, +1 if above the first
                   15-min range high, -1 if below the low, 0 otherwise.

These plug into `create_hybrid_model` only when intraday data exists;
callers pass a dict of scalars (like `get_option_features` does) and the
model broadcasts across training rows + uses them live on prediction day.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from data.intraday import get_intraday_bars

_DEFAULT_ZERO = {
    "intraday_available": 0.0,
    "rsi_5m": 50.0,
    "vwap_distance": 0.0,
    "orb_flag": 0.0,
}


def extract_intraday_features(df: pd.DataFrame) -> dict[str, float]:
    """
    Reduce an intraday OHLCV frame to scalar features. Empty frame → zeros.

    The frame must be indexed by timestamps within a single trading day
    for VWAP + ORB to be meaningful. If multi-day bars are passed, only
    the most recent day's bars are used for the session calcs.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return dict(_DEFAULT_ZERO)

    # RSI wants a long run of bars; cap to 5-min bars if the index is denser.
    close = df["Close"].astype(float)
    rsi = _rsi(close, period=14)
    rsi_val = float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50.0

    # Session-scope calcs: restrict to the latest trading day in the frame.
    last_day = df.index[-1].date() if hasattr(df.index[-1], "date") else None
    session = df[df.index.date == last_day] if last_day else df

    vwap = _session_vwap(session)
    vwap_distance = 0.0
    if not vwap.empty and vwap.iloc[-1] > 0:
        vwap_distance = float((close.iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1])
        vwap_distance = float(np.clip(vwap_distance, -0.1, 0.1))

    orb = _opening_range_breakout(session, minutes=15)

    return {
        "intraday_available": 1.0,
        "rsi_5m": rsi_val,
        "vwap_distance": vwap_distance,
        "orb_flag": float(orb),
    }


def get_intraday_features(ticker: str, interval: str = "5m", period: str = "5d") -> dict[str, float]:
    """Convenience: fetch + extract. Zeros on failure."""
    df = get_intraday_bars(ticker, interval=interval, period=period)
    return extract_intraday_features(df)


# ───────────────────────── helpers ─────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Classic Wilder RSI. Handles pure trends correctly:
        pure uptrend → avg_loss ≈ 0 → rsi → 100
        pure downtrend → avg_gain ≈ 0 → rsi → 0
        flat → both ≈ 0 → rsi ≈ 50
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    # Epsilon avoids both div-by-zero and the NaN cascade that buries pure trends.
    rs = (avg_gain + 1e-12) / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _session_vwap(session: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP across one session's bars."""
    if session.empty:
        return pd.Series(dtype=float)
    tp = (session["High"] + session["Low"] + session["Close"]) / 3.0
    vol = session["Volume"].replace(0, np.nan)
    cum_vol = vol.cumsum()
    cum_pv = (tp * vol).cumsum()
    return (cum_pv / cum_vol).fillna(method="ffill").fillna(session["Close"])


def _opening_range_breakout(session: pd.DataFrame, minutes: int = 15) -> int:
    """
    Opening-Range Breakout flag for the session.
    +1 if last close > first `minutes` high, -1 if < first `minutes` low, 0 else.
    """
    if session.empty:
        return 0
    first_ts = session.index[0]
    cutoff = first_ts + timedelta(minutes=minutes)
    opening = session[session.index <= cutoff]
    if opening.empty:
        return 0

    or_high = float(opening["High"].max())
    or_low = float(opening["Low"].min())
    last_close = float(session["Close"].iloc[-1])

    if last_close > or_high:
        return 1
    if last_close < or_low:
        return -1
    return 0


# Keep `date` importable for tests / typing.
__all__ = [
    "extract_intraday_features",
    "get_intraday_features",
    "date",
]
