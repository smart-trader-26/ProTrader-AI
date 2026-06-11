"""
Swing-screener DTOs (P4-2).

The 30-day directional signal fires only ~5-6% of the time, so viewing it one
ticker at a time means users almost always see NEUTRAL. The screener inverts
the question — "which names are firing *today*?" — by running the persisted
signal across the whole training universe in one batch.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SwingScanRow(BaseModel):
    """One ticker's swing-signal snapshot from a universe scan."""

    ticker: str
    signal: str = "NEUTRAL"          # "UP" | "NEUTRAL"
    prob_up: float = Field(default=0.5, ge=0.0, le=1.0)
    conviction: float = Field(default=0.0, ge=0.0, le=1.0)
    last_close: float | None = None
    asof: str | None = None          # date of the last bar used (YYYY-MM-DD)


class SwingScanResult(BaseModel):
    """Universe-wide swing scan + the honesty stats backing every number."""

    generated_at: datetime
    horizon_days: int = 30
    trained: bool = False
    tau_star: float | None = None
    expected_hit_rate: float | None = None   # measured OOS precision of fired bucket
    base_rate: float | None = None           # always-up base rate (the bar to beat)
    coverage: float | None = None            # how often the signal fires historically
    n_scanned: int = 0
    n_fired: int = 0
    n_errors: int = 0
    cached: bool = False
    rows: list[SwingScanRow] = Field(default_factory=list)
