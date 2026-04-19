"""
Smoke tests for stock_service — real yfinance, no mocks (CLAUDE.md §2.4).

Uses RELIANCE.NS, the most liquid NSE ticker; if yfinance is unreachable or
Yahoo rate-limits, the `network` marker lets the suite skip gracefully.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from schemas.stock import Fundamentals, StockHistory
from services import stock_service

TICKER = "RELIANCE.NS"


@pytest.mark.network
@pytest.mark.slow
def test_get_history_returns_typed_bars():
    end = date.today()
    start = end - timedelta(days=30)
    hist = stock_service.get_history(TICKER, start, end)
    assert isinstance(hist, StockHistory)
    assert hist.ticker == TICKER
    # Yahoo may skip weekends / holidays, but 30 days should include > 10 trading days
    if hist.n_bars == 0:
        pytest.skip("yfinance returned empty frame (rate limit or holiday cluster)")
    bar = hist.bars[-1]
    assert bar.high >= bar.low
    assert bar.close > 0


@pytest.mark.network
@pytest.mark.slow
def test_get_fundamentals_typed():
    f = stock_service.get_fundamentals(TICKER)
    assert isinstance(f, Fundamentals)
    assert f.ticker == TICKER
