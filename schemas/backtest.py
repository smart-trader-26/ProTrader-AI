"""Backtest DTOs."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class BacktestMetrics(BaseModel):
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    sortino: float | None = None
    calmar: float | None = None
    max_drawdown_pct: float
    profit_factor: float | None = None
    win_rate_pct: float = Field(ge=0.0, le=100.0)
    avg_win_pct: float | None = None
    avg_loss_pct: float | None = None
    expectancy_pct: float | None = None
    n_trades: int = 0
    dm_pvalue: float | None = Field(
        default=None, description="Diebold-Mariano p-value vs. buy-and-hold"
    )


class BacktestResult(BaseModel):
    ticker: str
    start: date
    end: date
    strategy: str
    initial_capital: float
    final_equity: float
    metrics: BacktestMetrics
