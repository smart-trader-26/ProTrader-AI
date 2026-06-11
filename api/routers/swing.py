"""
Swing screener router (P4-2).

  GET /api/v1/swing/scan          → SwingScanResult (cached ≤30 min)
  GET /api/v1/swing/scan?refresh=1 → force a fresh scan

Runs the persisted 30-day directional signal across the training universe in
one batch (single yfinance download + LR inference — seconds, not minutes).
Free-tier and keyless; returns `trained=false` with no rows when the model
artifact is missing.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from schemas.swing import SwingScanResult
from services import swing_scan_service

router = APIRouter(prefix="/swing", tags=["swing"])


@router.get(
    "/scan",
    response_model=SwingScanResult,
    summary="Run the 30-day swing signal across the universe",
)
def swing_scan(
    refresh: bool = Query(default=False, description="Bypass the 30-min cache"),
) -> SwingScanResult:
    return swing_scan_service.scan(refresh=refresh)
