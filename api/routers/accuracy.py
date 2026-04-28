"""
Accuracy ledger router (A7 surfaced over HTTP).

  GET  /api/v1/accuracy?ticker=&days=        → AccuracyWindow rollup
  GET  /api/v1/accuracy/recent?ticker=       → recent ledger rows
  POST /api/v1/accuracy/resolve              → manually trigger backfill_actuals

The ledger is the single source of truth for "is the model actually
right?". Both Streamlit and the future Next.js dashboard read from here.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, status
from pydantic import BaseModel

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


class ResolveResponse(BaseModel):
    resolved: int
    message: str


class MigrateResponse(BaseModel):
    updated: int
    message: str


@router.post(
    "/migrate/nullify-non-day1-prob",
    response_model=MigrateResponse,
    status_code=status.HTTP_200_OK,
    summary="One-time migration: null out prob_up for non-day-1 ledger rows",
)
def migrate_prob_up() -> MigrateResponse:
    """
    Fixes ledger rows written before Phase-1C: every horizon row (days 2-10)
    had the same prob_up as day-1, inflating the ECE/Brier denominator and
    making 7/30/90-day accuracy windows show identical stats.

    Safe to run multiple times. After running, call /resolve to re-score.
    """
    updated = ledger_service.migrate_nullify_non_day1_prob_up()
    return MigrateResponse(
        updated=updated,
        message=(
            f"Nullified prob_up on {updated} non-day-1 row(s). "
            "Run /accuracy/resolve next to re-score directional hits."
            if updated > 0
            else "Nothing to migrate — all non-day-1 rows already have prob_up=NULL."
        ),
    )


@router.post(
    "/resolve",
    response_model=ResolveResponse,
    status_code=status.HTTP_200_OK,
    summary="Manually trigger ledger backfill (dev/testing workaround)",
)
def resolve_now() -> ResolveResponse:
    """
    Fetch actual close prices for every unresolved prediction whose
    target_date has already passed and compute directional hit flags.

    Normally this runs as a daily background job at market close.
    This endpoint lets you trigger it immediately for testing without
    waiting for the cron schedule.
    """
    resolved = ledger_service.backfill_actuals()
    return ResolveResponse(
        resolved=resolved,
        message=(
            f"Resolved {resolved} prediction(s). "
            "Refresh the Accuracy tab to see results."
            if resolved > 0
            else "No pending predictions to resolve yet. "
                 "Predictions can only be resolved after their target date has passed."
        ),
    )
