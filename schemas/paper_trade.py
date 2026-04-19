"""
Paper-trading DTOs (A9).

A *paper fill* records one simulated trade: qty, entry / exit prices, and
computed P&L after the NSE cost stack. The `PaperPosition` is the live
book — a ticker has at most one open position at a time.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Side = Literal["long", "short", "flat"]


class PaperFill(BaseModel):
    """One round-trip simulated trade."""

    ticker: str
    opened_at: datetime
    closed_at: datetime | None = None
    side: Side
    qty: int = Field(ge=1)
    entry_price: float = Field(gt=0)
    exit_price: float | None = None
    gross_pnl: float = 0.0       # (exit - entry) * qty × side_multiplier
    costs: float = 0.0           # round-trip transaction costs
    net_pnl: float = 0.0
    reason_entry: str = ""       # e.g. "prob_up=0.72, thr=0.55"
    reason_exit: str = ""        # e.g. "target_hit", "stop_hit", "time_exit"


class PaperPosition(BaseModel):
    """Current open position for a ticker. `side='flat'` → no position."""

    ticker: str
    side: Side = "flat"
    qty: int = 0
    entry_price: float = 0.0
    opened_at: datetime | None = None
    stop_price: float | None = None
    target_price: float | None = None


class PaperBookState(BaseModel):
    """Snapshot of the paper-trading book — used by the UI."""

    cash: float
    realised_pnl: float
    unrealised_pnl: float
    n_fills: int
    n_open: int
    equity: float  # cash + unrealised
