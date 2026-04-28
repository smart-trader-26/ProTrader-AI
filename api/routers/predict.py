"""
Prediction router (B1.4).

  POST /api/v1/stocks/{ticker}/predict   → 202 Accepted, {job_id}
  GET  /api/v1/jobs/{id}                 → poll for status / result

Long path because `services.prediction_service.predict()` fits XGB / LGBM /
GRU on demand — typically 30-90s. Returning a job_id keeps the connection
short and lets the frontend show a progress UI without HTTP timeouts.

A7 wiring is already inside prediction_service.predict() — every job
appends to the SQLite ledger and `bundle.accuracy_30d` carries the rolling
hit-rate badge for the response.

Backend selection: when REDIS_URL is set the job runs on the Celery worker;
otherwise it runs on the in-process thread pool. Router code is identical
for both — the JobStore picks based on env (B2).
"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field

from api.deps import get_job_store
from api.jobs import JobStore
from api.rate_limit import limiter

router = APIRouter(tags=["predictions"])


class FiiDiiInputRow(BaseModel):
    """One day of FII/DII flow data supplied manually by the user."""
    date: str
    fii_net: float | None = None
    dii_net: float | None = None
    fii_buy: float | None = None
    fii_sell: float | None = None
    dii_buy: float | None = None
    dii_sell: float | None = None


class PredictRequest(BaseModel):
    horizon_days: int = Field(default=10, ge=1, le=60)
    start: date | None = None
    end: date | None = None
    n_paths: int = Field(default=200, ge=10, le=2000)
    log_to_ledger: bool = True
    # Optional FII/DII data provided by the user when auto-fetch from NSE fails.
    # If provided, this is passed directly as the fii_dii_data feature input to
    # the model, bypassing the auto-fetch. Rows should cover at least 30 days.
    fii_dii_rows: list[FiiDiiInputRow] | None = None


class PredictAccepted(BaseModel):
    job_id: str
    status: str
    poll_url: str


@router.post(
    "/stocks/{ticker}/predict",
    response_model=PredictAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue a hybrid-model prediction job",
)
@limiter.limit("10/minute")
def enqueue_predict(
    request: Request,
    ticker: str,
    body: PredictRequest | None = None,
    store: JobStore = Depends(get_job_store),
) -> PredictAccepted:
    body = body or PredictRequest()

    # Convert FII/DII input rows to a plain list of dicts for the job queue
    # (Pydantic models don't survive JSON serialisation through Celery).
    fii_dii_payload: list[dict] | None = None
    if body.fii_dii_rows:
        fii_dii_payload = [r.model_dump() for r in body.fii_dii_rows]

    job = store.enqueue(
        "predict",
        ticker=ticker,
        horizon_days=body.horizon_days,
        start=body.start,
        end=body.end,
        n_paths=body.n_paths,
        log_to_ledger=body.log_to_ledger,
        fii_dii_payload=fii_dii_payload,
    )
    return PredictAccepted(
        job_id=job.id,
        status=job.status,
        poll_url=f"/api/v1/jobs/{job.id}",
    )
