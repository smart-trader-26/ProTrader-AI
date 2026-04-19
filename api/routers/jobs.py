"""
Job polling router (B1.4 / B1.5 reads).

  GET /api/v1/jobs/{id}   → status + result (when terminal) or error

The result field is a JSON-serialised Pydantic model. Status flow:
  queued → running → succeeded | failed
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.deps import get_job_store
from api.jobs import JobStore

router = APIRouter(prefix="/jobs", tags=["jobs"])


class JobView(BaseModel):
    id: str
    kind: str
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result: Any = None
    error: str | None = None


@router.get("/{job_id}", response_model=JobView, summary="Poll an enqueued job")
def get_job(job_id: str, store: JobStore = Depends(get_job_store)) -> JobView:
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="job not found or expired")

    result_payload = rec.result
    # Pydantic v2 → dump to JSON-safe dict so FastAPI doesn't choke on dates.
    if hasattr(result_payload, "model_dump"):
        result_payload = result_payload.model_dump(mode="json")

    return JobView(
        id=rec.id,
        kind=rec.kind,
        status=rec.status,
        created_at=rec.created_at.isoformat(),
        started_at=rec.started_at.isoformat() if rec.started_at else None,
        finished_at=rec.finished_at.isoformat() if rec.finished_at else None,
        result=result_payload,
        error=rec.error,
    )
