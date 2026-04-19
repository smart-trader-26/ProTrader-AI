"""
Watchlists router (B3, framework mode) — per-user CRUD over Supabase REST.

  POST   /api/v1/watchlists                       → create
  GET    /api/v1/watchlists                       → list mine
  DELETE /api/v1/watchlists/{wl_id}               → delete
  POST   /api/v1/watchlists/{wl_id}/tickers       → add ticker
  DELETE /api/v1/watchlists/{wl_id}/tickers/{t}   → remove ticker

Every endpoint requires a valid Supabase JWT. We use a per-request
`get_user_client(user.access_token)` — PostgREST enforces RLS off the
`sub` claim so the user CAN'T see rows they don't own, even if the query
code below forgets the `user_id` filter. That's defense-in-depth: app
code is the first layer, RLS is the hard backstop.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.auth import AuthUser, current_user
from db import supabase_client as sb
from db.supabase_client import get_user_client

router = APIRouter(prefix="/watchlists", tags=["watchlists"])


class WatchlistTickerView(BaseModel):
    ticker: str
    added_at: datetime


class WatchlistView(BaseModel):
    id: int
    name: str
    created_at: datetime
    tickers: list[WatchlistTickerView] = Field(default_factory=list)


class CreateWatchlistRequest(BaseModel):
    name: str = Field(default="Default", min_length=1, max_length=64)


class AddTickerRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=32)


def _require_db() -> None:
    if not sb.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase is not configured on this server",
        )


def _shape(wl: dict, tickers: list[dict]) -> WatchlistView:
    return WatchlistView(
        id=wl["id"],
        name=wl["name"],
        created_at=wl["created_at"],
        tickers=[
            WatchlistTickerView(ticker=t["ticker"], added_at=t["added_at"])
            for t in tickers if t.get("watchlist_id") == wl["id"]
        ],
    )


@router.get("", response_model=list[WatchlistView], summary="List my watchlists")
def list_watchlists(user: Annotated[AuthUser, Depends(current_user)]) -> list[WatchlistView]:
    _require_db()
    client = get_user_client(user.access_token)

    wls = (
        client.table("watchlists")
        .select("id,name,created_at")
        .order("created_at", desc=False)
        .execute()
        .data
        or []
    )
    if not wls:
        return []

    tickers = (
        client.table("watchlist_tickers")
        .select("watchlist_id,ticker,added_at")
        .in_("watchlist_id", [w["id"] for w in wls])
        .execute()
        .data
        or []
    )
    return [_shape(w, tickers) for w in wls]


@router.post(
    "",
    response_model=WatchlistView,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new watchlist",
)
def create_watchlist(
    body: CreateWatchlistRequest,
    user: Annotated[AuthUser, Depends(current_user)],
) -> WatchlistView:
    _require_db()
    client = get_user_client(user.access_token)

    # RLS policy on `watchlists.user_id = auth.uid()` forces us to include
    # user_id in the insert — PostgREST rejects rows that don't match.
    try:
        resp = (
            client.table("watchlists")
            .insert({"user_id": user.id, "name": body.name})
            .execute()
        )
    except Exception as e:  # noqa: BLE001 — uniq violation surfaces as APIError
        if "duplicate" in str(e).lower() or "unique" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"watchlist name {body.name!r} already exists",
            ) from e
        raise
    row = (resp.data or [None])[0]
    if row is None:
        raise HTTPException(status_code=500, detail="insert returned no row")
    return _shape(row, [])


@router.delete(
    "/{wl_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a watchlist (and its tickers)",
)
def delete_watchlist(
    wl_id: int,
    user: Annotated[AuthUser, Depends(current_user)],
) -> None:
    _require_db()
    client = get_user_client(user.access_token)

    # Tickers go via the ON DELETE CASCADE on watchlist_id — Postgres
    # handles the child rows once the parent is gone.
    resp = client.table("watchlists").delete().eq("id", wl_id).execute()
    if not resp.data:
        raise HTTPException(status_code=404, detail="watchlist not found")


@router.post(
    "/{wl_id}/tickers",
    response_model=WatchlistView,
    summary="Add a ticker to a watchlist",
)
def add_ticker(
    wl_id: int,
    body: AddTickerRequest,
    user: Annotated[AuthUser, Depends(current_user)],
) -> WatchlistView:
    _require_db()
    client = get_user_client(user.access_token)

    wl = (
        client.table("watchlists")
        .select("id,name,created_at")
        .eq("id", wl_id)
        .execute()
        .data
    )
    if not wl:
        raise HTTPException(status_code=404, detail="watchlist not found")

    ticker = body.ticker.strip().upper()
    # Idempotent upsert on the composite PK (watchlist_id, ticker).
    client.table("watchlist_tickers").upsert(
        {"watchlist_id": wl_id, "ticker": ticker},
        on_conflict="watchlist_id,ticker",
        ignore_duplicates=True,
    ).execute()

    tickers = (
        client.table("watchlist_tickers")
        .select("watchlist_id,ticker,added_at")
        .eq("watchlist_id", wl_id)
        .execute()
        .data
        or []
    )
    return _shape(wl[0], tickers)


@router.delete(
    "/{wl_id}/tickers/{ticker}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a ticker from a watchlist",
)
def remove_ticker(
    wl_id: int,
    ticker: str,
    user: Annotated[AuthUser, Depends(current_user)],
) -> None:
    _require_db()
    client = get_user_client(user.access_token)

    # RLS on watchlist_tickers joins through watchlist_id → watchlists.user_id;
    # a user can't delete rows belonging to someone else even without this
    # check, but we 404 explicitly for a cleaner error.
    resp = (
        client.table("watchlist_tickers")
        .delete()
        .eq("watchlist_id", wl_id)
        .eq("ticker", ticker.upper())
        .execute()
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="ticker not in watchlist")
