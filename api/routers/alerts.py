"""
Alerts router (B3, framework mode) — per-user CRUD over Supabase REST.

  POST   /api/v1/alerts                → create
  GET    /api/v1/alerts                → list mine
  PATCH  /api/v1/alerts/{id}           → toggle active / update threshold
  DELETE /api/v1/alerts/{id}           → delete

Firing happens in the worker (`workers.tasks._alert_eval_sync`); this router
doesn't send notifications. The `triggered_at` column is the poll target.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.auth import AuthUser, current_user
from db import supabase_client as sb
from db.supabase_client import get_user_client

router = APIRouter(prefix="/alerts", tags=["alerts"])

AlertKind = Literal["price_above", "price_below", "prob_up_above", "prob_up_below"]


class AlertView(BaseModel):
    id: int
    ticker: str
    kind: AlertKind
    threshold: float
    active: bool
    triggered_at: datetime | None
    created_at: datetime


class CreateAlertRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=32)
    kind: AlertKind
    threshold: float = Field(gt=0)


class UpdateAlertRequest(BaseModel):
    active: bool | None = None
    threshold: float | None = Field(default=None, gt=0)


_COLUMNS = "id,ticker,kind,threshold,active,triggered_at,created_at"


def _require_db() -> None:
    if not sb.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase is not configured on this server",
        )


@router.get("", response_model=list[AlertView], summary="List my alerts")
def list_alerts(user: Annotated[AuthUser, Depends(current_user)]) -> list[AlertView]:
    _require_db()
    client = get_user_client(user.access_token)
    rows = (
        client.table("alerts")
        .select(_COLUMNS)
        .order("active", desc=True)
        .order("created_at", desc=True)
        .execute()
        .data
        or []
    )
    return [AlertView(**r) for r in rows]


@router.post(
    "",
    response_model=AlertView,
    status_code=status.HTTP_201_CREATED,
    summary="Create an alert",
)
def create_alert(
    body: CreateAlertRequest,
    user: Annotated[AuthUser, Depends(current_user)],
) -> AlertView:
    _require_db()
    client = get_user_client(user.access_token)
    row = {
        "user_id":   user.id,
        "ticker":    body.ticker.strip().upper(),
        "kind":      body.kind,
        "threshold": body.threshold,
        "active":    True,
    }
    resp = client.table("alerts").insert(row).execute()
    out = (resp.data or [None])[0]
    if out is None:
        raise HTTPException(status_code=500, detail="insert returned no row")
    return AlertView(**{k: out.get(k) for k in AlertView.model_fields})


@router.patch(
    "/{alert_id}",
    response_model=AlertView,
    summary="Toggle active or change threshold",
)
def update_alert(
    alert_id: int,
    body: UpdateAlertRequest,
    user: Annotated[AuthUser, Depends(current_user)],
) -> AlertView:
    _require_db()
    client = get_user_client(user.access_token)

    patch: dict = {}
    if body.active is not None:
        patch["active"] = body.active
        if body.active:
            patch["triggered_at"] = None  # re-arm
    if body.threshold is not None:
        patch["threshold"] = body.threshold
    if not patch:
        raise HTTPException(status_code=400, detail="no fields to update")

    resp = client.table("alerts").update(patch).eq("id", alert_id).execute()
    out = (resp.data or [None])[0]
    if out is None:
        raise HTTPException(status_code=404, detail="alert not found")
    return AlertView(**{k: out.get(k) for k in AlertView.model_fields})


@router.delete(
    "/{alert_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an alert",
)
def delete_alert(
    alert_id: int,
    user: Annotated[AuthUser, Depends(current_user)],
) -> None:
    _require_db()
    client = get_user_client(user.access_token)
    resp = client.table("alerts").delete().eq("id", alert_id).execute()
    if not resp.data:
        raise HTTPException(status_code=404, detail="alert not found")
