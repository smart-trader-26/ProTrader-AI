"""
Backtest router (B1.5).

  POST /api/v1/stocks/{ticker}/backtest   → 202 Accepted, {job_id}
  GET  /api/v1/jobs/{id}                  → poll for status / result

Same job pattern as predict — backtests over multi-year windows can run
into the tens of seconds. Strategies supported by `backtest_service`:
  • "ma_crossover" (default)
  • "momentum"
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field

from api.deps import get_job_store
from api.jobs import JobStore
from api.rate_limit import limiter

router = APIRouter(tags=["backtests"])

Strategy = Literal["ma_crossover", "momentum"]


class BacktestRequest(BaseModel):
    strategy: Strategy = "ma_crossover"
    start: date | None = None
    end: date | None = None
    initial_capital: float | None = Field(default=None, ge=1000)
    include_costs: bool = True


class BacktestAccepted(BaseModel):
    job_id: str
    status: str
    poll_url: str


@router.post(
    "/stocks/{ticker}/backtest",
    response_model=BacktestAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue a vectorized backtest job",
)
@limiter.limit("20/minute")
def enqueue_backtest(
    request: Request,
    ticker: str,
    body: BacktestRequest | None = None,
    store: JobStore = Depends(get_job_store),
) -> BacktestAccepted:
    body = body or BacktestRequest()
    job = store.enqueue(
        "backtest",
        ticker=ticker,
        strategy=body.strategy,
        start=body.start,
        end=body.end,
        initial_capital=body.initial_capital,
        include_costs=body.include_costs,
    )
    return BacktestAccepted(
        job_id=job.id,
        status=job.status,
        poll_url=f"/api/v1/jobs/{job.id}",
    )
