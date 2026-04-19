"""
Accuracy ledger router (A7 surfaced over HTTP).

  GET /api/v1/accuracy?ticker=&days=        → AccuracyWindow rollup
  GET /api/v1/accuracy/recent?ticker=       → recent ledger rows

The ledger is the single source of truth for "is the model actually
right?". Both Streamlit and the future Next.js dashboard read from here.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from schemas.ledger import AccuracyWindow, LedgerRow
from services import ledger_service

router = APIRouter(prefix="/accuracy", tags=["accuracy"])


@router.get("", response_model=AccuracyWindow, summary="Rolling accuracy window")
def accuracy_window(
    ticker: str | None = Query(default=None),
    days: int = Query(default=30, ge=1, le=365),
) -> AccuracyWindow:
    return ledger_service.accuracy_window(ticker=ticker, days=days)


@router.get(
    "/recent",
    response_model=list[LedgerRow],
    summary="Most recent ledger rows for a ticker",
)
def recent(
    ticker: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[LedgerRow]:
    return ledger_service.recent_rows(ticker=ticker, limit=limit)
