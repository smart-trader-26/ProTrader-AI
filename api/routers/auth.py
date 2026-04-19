"""
Auth router (B3.4, framework mode) — `/me` endpoint.

  GET   /api/v1/me   → 200 {id, email, plan, display_name}
  PATCH /api/v1/me   → update display_name

Sign-up / sign-in happens in the frontend via the Supabase JS SDK; we just
verify the token and return the profile row. The row itself is created by
the `handle_new_user` trigger in the SQL schema at signup time.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api.auth import AuthUser, current_user
from db import supabase_client as sb
from db.supabase_client import get_user_client

router = APIRouter(tags=["auth"])


class MeResponse(BaseModel):
    id: str
    email: str | None = None
    display_name: str | None = None
    plan: str = "free"


@router.get("/me", response_model=MeResponse, summary="Current user profile")
def me(user: Annotated[AuthUser, Depends(current_user)]) -> MeResponse:
    if not sb.is_configured():
        # Token decoded but no DB to look up the profile — return JWT claims.
        return MeResponse(id=user.id, email=user.email)

    client = get_user_client(user.access_token)
    rows = (
        client.table("user_profiles")
        .select("id,email,display_name,plan")
        .eq("id", user.id)
        .limit(1)
        .execute()
        .data
    )
    if not rows:
        # Trigger should create this on signup; fall back gracefully.
        return MeResponse(id=user.id, email=user.email)

    r = rows[0]
    return MeResponse(
        id=r["id"],
        email=r.get("email") or user.email,
        display_name=r.get("display_name"),
        plan=r.get("plan") or "free",
    )


class UpdateProfileRequest(BaseModel):
    display_name: str | None = None


@router.patch("/me", response_model=MeResponse, summary="Update display name")
def update_me(
    body: UpdateProfileRequest,
    user: Annotated[AuthUser, Depends(current_user)],
) -> MeResponse:
    if not sb.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="profile updates require Supabase to be configured",
        )

    client = get_user_client(user.access_token)

    # Upsert so first-time users (pre-trigger edge cases) still work.
    patch = {"id": user.id}
    if body.display_name is not None:
        patch["display_name"] = body.display_name
    if user.email:
        patch["email"] = user.email

    resp = client.table("user_profiles").upsert(patch, on_conflict="id").execute()
    rows = resp.data or []
    r = rows[0] if rows else {}
    return MeResponse(
        id=r.get("id", user.id),
        email=r.get("email") or user.email,
        display_name=r.get("display_name"),
        plan=r.get("plan") or "free",
    )
