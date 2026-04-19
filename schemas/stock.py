"""OHLCV + fundamentals DTOs."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class StockBar(BaseModel):
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class StockHistory(BaseModel):
    ticker: str
    start: date
    end: date
    bars: list[StockBar] = Field(default_factory=list)

    @property
    def n_bars(self) -> int:
        return len(self.bars)


class Fundamentals(BaseModel):
    ticker: str
    market_cap: float | None = None
    pe_ratio: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None
    price_to_book: float | None = None
    debt_to_equity: float | None = None
    roe: float | None = None
    profit_margin: float | None = None
    revenue_growth: float | None = None
    free_cashflow: float | None = None
    target_price: float | None = None
    dividend_yield: float | None = None
