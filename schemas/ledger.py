"""
Prediction ledger DTOs (A7).

A prediction ledger row captures what the model said *at decision time*,
then gets filled in after the target date passes with the actual outcome.
Every downstream accuracy metric (rolling hit-rate, ECE, Brier) reads
from this one table.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

LedgerDirection = Literal["up", "down", "flat"]


class LedgerRow(BaseModel):
    """One prediction + (optionally) its resolved outcome."""

    id: int | None = None
    ticker: str
    made_at: datetime
    target_date: date
    pred_dir: LedgerDirection
    pred_price: float
    anchor_price: float | None = None  # close on made_at's trading day
    ci_low: float | None = None
    ci_high: float | None = None
    confidence_level: float = 0.90
    prob_up: float | None = Field(default=None, ge=0.0, le=1.0)
    horizon_days: int = 1
    model_version: str

    # Filled in by the backfill job once target_date <= today
    actual_price: float | None = None
    hit: bool | None = None


class AccuracyWindow(BaseModel):
    """Rolling-window accuracy summary for a ticker (or all tickers)."""

    ticker: str | None = None
    window_days: int
    n_predictions: int
    n_resolved: int
    directional_accuracy: float | None = None  # 0..1
    brier_score: float | None = None
    ece: float | None = None
    mae_price: float | None = None
