"""
JWT auth (B3.4) — verifies Supabase access tokens.

Supabase issues HS256-signed JWTs to logged-in users. The frontend forwards
that token in `Authorization: Bearer <jwt>`. We verify the signature with
`SUPABASE_JWT_SECRET` (Project Settings → API → JWT Settings) and pull
`sub` (the user's UUID) + `email` out of the payload.

Two FastAPI deps are exported:

  • `current_user`  — required. Returns `AuthUser` or 401.
  • `optional_user` — soft. Returns `AuthUser | None` for routes that work
                      with or without a login (predict, backtest, etc.).

Dev mode: when `SUPABASE_JWT_SECRET` is unset (the default for a local
checkout) every request is treated as anonymous — `optional_user` returns
None, `current_user` 401s. This keeps `pytest` and `uvicorn --reload`
working out-of-the-box without forcing a Supabase setup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import SUPABASE_JWT_SECRET

log = logging.getLogger(__name__)

# `auto_error=False` so missing-token requests reach our handler instead of
# FastAPI raising 403 itself. Lets `optional_user` distinguish "no creds"
# from "bad creds".
_bearer = HTTPBearer(auto_error=False)


@dataclass
class AuthUser:
    id: str        # UUID from `sub`
    email: str | None
    role: str = "authenticated"
    # Raw bearer token — routes pass this to `get_user_client()` so RLS
    # enforces ownership for everything the user reads/writes.
    access_token: str = ""


def _decode_token(token: str) -> dict:
    """Verify HS256 + audience and return the claims dict."""
    if not SUPABASE_JWT_SECRET:
        # Should be guarded upstream; bail loudly if we land here in prod.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="auth: SUPABASE_JWT_SECRET is not configured",
        )
    try:
        return jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token expired") from e
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"invalid token: {e}") from e


def current_user(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> AuthUser:
    """Required-auth dep — raises 401 when no/invalid token."""
    if not SUPABASE_JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication is not configured on this server",
        )
    if creds is None or not creds.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")

    claims = _decode_token(creds.credentials)
    return AuthUser(
        id=claims["sub"],
        email=claims.get("email"),
        role=claims.get("role", "authenticated"),
        access_token=creds.credentials,
    )


def optional_user(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> AuthUser | None:
    """
    Soft-auth dep — returns `None` when no token (anonymous use OK).
    A bad token still 401s; we only allow the *no-creds* case.
    """
    if creds is None or not creds.credentials:
        return None
    if not SUPABASE_JWT_SECRET:
        # Token sent but server has no secret to verify it. Be loud.
        log.warning("auth: bearer token sent but SUPABASE_JWT_SECRET is unset")
        return None
    claims = _decode_token(creds.credentials)
    return AuthUser(
        id=claims["sub"],
        email=claims.get("email"),
        role=claims.get("role", "authenticated"),
        access_token=creds.credentials,
    )
