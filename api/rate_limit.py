"""
Rate limiting (B3.5) — `slowapi` per-user (or per-IP) throttling.

The router decorators (`@limiter.limit("10/minute")` on `/predict`,
`@limiter.limit("20/minute")` on `/backtest`) apply *per key*. The key is:

  • `sub` claim from the bearer JWT, when present (so a logged-in user
    has their quota tied to their account, not their NAT'd office IP).
  • Falls back to `request.client.host` for anonymous traffic.

Storage backend:
  • In-memory by default (single uvicorn worker — fine for dev / small
    deploys).
  • Redis when `REDIS_URL` is set, so quotas survive a worker restart and
    are shared across replicas. slowapi reuses our existing Redis URL.

`tests/conftest.py` calls `limiter.reset()` between tests so quotas don't
leak between cases.
"""

from __future__ import annotations

import logging

import jwt
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from config.settings import REDIS_URL, SUPABASE_JWT_SECRET

log = logging.getLogger(__name__)


def _key_func(request: Request) -> str:
    """Prefer the JWT `sub` (user id) over IP for fairness across NATs."""
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer ") and SUPABASE_JWT_SECRET:
        token = auth.split(" ", 1)[1].strip()
        try:
            claims = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated",
                options={"verify_exp": False},  # the auth dep handles freshness
            )
            sub = claims.get("sub")
            if sub:
                return f"user:{sub}"
        except jwt.InvalidTokenError:
            pass  # fall through to IP-based limiting
    return f"ip:{get_remote_address(request)}"


_storage_uri = REDIS_URL or "memory://"

limiter = Limiter(
    key_func=_key_func,
    storage_uri=_storage_uri,
    # Per-route decorators provide the actual limits; this is a safety net.
    default_limits=["120/minute"],
    # `headers_enabled=True` would attach X-RateLimit-* headers but slowapi
    # then requires every limited endpoint to declare a `Response` param.
    # Skip the headers for now — the FE only needs the 429 status to retry.
    headers_enabled=False,
)

log.info("rate_limit: storage=%s", "redis" if REDIS_URL else "memory")
