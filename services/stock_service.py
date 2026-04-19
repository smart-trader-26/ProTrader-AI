"""
Stock data service — OHLCV + fundamentals.

Thin wrapper around `data.stock_data` that returns typed DTOs from
`schemas.stock`. Streamlit-agnostic: the Streamlit UI wraps `get_history`
with `st.cache_data` for A1.4; FastAPI will wrap it with a Redis cache.
"""

from __future__ import annotations

from datetime import date
from typing import Union

import pandas as pd

from data.stock_data import (
    get_fundamental_data as _raw_fundamentals,
)
from data.stock_data import (
    get_stock_data as _raw_history,
)
from data.stock_data import (
    get_stock_info as _raw_info,
)
from schemas.stock import Fundamentals, StockBar, StockHistory

DateLike = Union[date, str, pd.Timestamp]


def get_history(ticker: str, start: DateLike, end: DateLike) -> StockHistory:
    """Fetch OHLCV history as a typed `StockHistory`. Caller caches."""
    df = _raw_history(ticker, start, end)
    bars: list[StockBar] = []
    if df is not None and not df.empty:
        for ts, row in df.iterrows():
            bars.append(
                StockBar(
                    ts=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row.get("Volume", 0.0)),
                )
            )
    return StockHistory(
        ticker=ticker,
        start=pd.to_datetime(start).date(),
        end=pd.to_datetime(end).date(),
        bars=bars,
    )


def get_history_df(ticker: str, start: DateLike, end: DateLike) -> pd.DataFrame:
    """Raw DataFrame shape — used by models that expect the legacy layout."""
    return _raw_history(ticker, start, end)


def get_fundamentals(ticker: str) -> Fundamentals:
    raw = _raw_fundamentals(ticker) or {}
    return Fundamentals(
        ticker=ticker,
        forward_pe=_coerce(raw.get("Forward P/E")),
        peg_ratio=_coerce(raw.get("PEG Ratio")),
        price_to_book=_coerce(raw.get("Price/Book")),
        debt_to_equity=_coerce(raw.get("Debt/Equity")),
        roe=_coerce(raw.get("ROE")),
        profit_margin=_coerce(raw.get("Profit Margins")),
        revenue_growth=_coerce(raw.get("Revenue Growth")),
        free_cashflow=_coerce(raw.get("Free Cashflow")),
        target_price=_coerce(raw.get("Target Price (Analyst)")),
    )


def get_info(ticker: str) -> dict:
    """Passthrough for the human-readable info dict the UI still renders."""
    return _raw_info(ticker)


def _coerce(v) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None
