"""
Intraday bar fetcher (A3.1) — free, keyless, via yfinance.

Yahoo's granularity ladder (as of 2026-04):
    1m   → last 7 days only
    2m   → last 60 days
    5m   → last 60 days
    15m  → last 60 days
    60m  → last 2 years
    1d   → full history

Calls fall back gracefully: requested granularity → next-coarser → empty.
The hybrid model reads whichever interval succeeded and stamps the metric
dict so downstream code knows whether intraday features are valid.

Latency: NSE/BSE ticks are ~15 min delayed on Yahoo (free-tier cost). For
true real-time see A3.3 (Yahoo WS) and A3.4 (Dhan/Upstox).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

# Ordered from finest → coarsest. Each entry is (interval, max_period).
# yfinance rejects period > max for a given interval with a 400.
_FALLBACK_LADDER: list[tuple[str, str]] = [
    ("1m",  "5d"),
    ("2m",  "5d"),
    ("5m",  "30d"),
    ("15m", "60d"),
    ("30m", "60d"),
    ("60m", "2y"),
]


def get_intraday_bars(
    ticker: str,
    interval: str = "1m",
    period: str = "5d",
) -> pd.DataFrame:
    """
    Return an OHLCV DataFrame indexed by tz-naive UTC timestamps.

    Args:
        ticker:   yfinance symbol (e.g. `RELIANCE.NS`)
        interval: one of {1m,2m,5m,15m,30m,60m} — falls back to coarser
                  if the requested granularity hits Yahoo's period limit.
        period:   yfinance period string (`5d`, `30d`, `60d`, `2y`). Auto-
                  clamped to the interval's max per Yahoo rules.

    Returns:
        DataFrame with columns Open/High/Low/Close/Volume. Empty on any
        failure so callers can treat "no intraday" as a valid state.
    """
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()

    # Find the ladder entry matching the requested interval so we know
    # the max allowed period, then walk downward on failure.
    start_idx = 0
    for i, (iv, _) in enumerate(_FALLBACK_LADDER):
        if iv == interval:
            start_idx = i
            break

    for iv, max_period in _FALLBACK_LADDER[start_idx:]:
        effective_period = _clamp_period(period, max_period)
        try:
            df = yf.Ticker(ticker).history(period=effective_period, interval=iv)
        except Exception:
            continue
        if df is None or df.empty:
            continue

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.attrs["interval"] = iv
        df.attrs["period"] = effective_period
        return df

    return pd.DataFrame()


def is_nse_market_open(now: datetime | None = None) -> bool:
    """
    NSE trading hours: 09:15–15:30 IST, Mon–Fri (excluding holidays).
    Holiday calendar is not included — use this only as a coarse gate.
    """
    now = now or datetime.now()
    # IST is UTC+5:30 — if caller passed UTC-aware, convert to IST naive.
    if now.tzinfo is not None:
        now = (now + now.utcoffset() - timedelta(hours=-5, minutes=-30)).replace(tzinfo=None)  # type: ignore[operator]
    if now.weekday() >= 5:
        return False
    open_at = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_at = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_at <= now <= close_at


def _clamp_period(requested: str, max_allowed: str) -> str:
    """Return the stricter of `requested` and `max_allowed`."""
    if _period_days(requested) > _period_days(max_allowed):
        return max_allowed
    return requested


def _period_days(p: str) -> int:
    """`5d` → 5, `60d` → 60, `2y` → 730, `max` → 99999. Unknown → 0."""
    p = p.strip().lower()
    if p == "max":
        return 99999
    try:
        n = int(p[:-1])
    except (ValueError, IndexError):
        return 0
    unit = p[-1]
    return {"d": n, "w": n * 7, "mo": n * 30, "y": n * 365}.get(unit, 0)
