"""
Smoke tests for backtest_service.

End-to-end: fetch real OHLCV, generate signals, run VectorizedBacktester,
verify the typed `BacktestResult` shape.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from schemas.backtest import BacktestResult
from services import backtest_service


@pytest.mark.network
@pytest.mark.slow
def test_ma_crossover_round_trip():
    end = date.today()
    start = end - timedelta(days=365)
    try:
        res = backtest_service.run_backtest("RELIANCE.NS", strategy="ma_crossover",
                                            start=start, end=end, initial_capital=100_000)
    except ValueError as e:
        pytest.skip(f"yfinance unavailable: {e}")
    assert isinstance(res, BacktestResult)
    assert res.strategy == "ma_crossover"
    assert res.initial_capital == 100_000
    assert 0.0 <= res.metrics.win_rate_pct <= 100.0


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        backtest_service.run_backtest("X", strategy="nope",
                                      start=date(2024, 1, 1), end=date(2024, 2, 1))
