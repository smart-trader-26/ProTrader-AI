"""
Supabase client factory (framework mode, B3.6).

Two flavors of client — pick by caller context:

  • `get_admin_client()` — service-role key, bypasses RLS. For backend jobs
    that write/read across users (predictions ledger, alert evaluator, the
    Celery beat tasks). Cached per-process.

  • `get_user_client(access_token)` — fresh client authenticated as the end
    user via their Supabase JWT. RLS enforces `auth.uid() = user_id` so
    callers CAN'T see rows they don't own, even if router code forgot the
    filter. Use this in every per-user route (`/watchlists`, `/alerts`, `/me`).

Rationale for framework mode: direct Postgres on Supabase free tier is
IPv6-only; PostgREST/GoTrue sit behind the same domain and are IPv4-reachable
everywhere. No psycopg, no connection pool, no DNS issues.

`is_configured()` is True iff `SUPABASE_URL` + a working key are set.
Anything that needs the DB should check this first; falls back to the local
SQLite ledger (`services.ledger_service`) when False so dev mode stays free.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from supabase import Client, create_client

from config.settings import (
    SUPABASE_ANON_KEY,
    SUPABASE_SERVICE_ROLE_KEY,
    SUPABASE_URL,
)

log = logging.getLogger(__name__)


def is_configured() -> bool:
    """True when the backend can reach Supabase at all.

    For admin writes (ledger, alert eval), you ALSO need the service role
    key. Check via `has_service_role()` for those paths.
    """
    return bool(SUPABASE_URL) and bool(SUPABASE_ANON_KEY or SUPABASE_SERVICE_ROLE_KEY)


def has_service_role() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)


@lru_cache(maxsize=1)
def get_admin_client() -> Client:
    """Service-role client — RLS bypassed. One per process."""
    if not has_service_role():
        raise RuntimeError(
            "get_admin_client() called without SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY set."
            " These are required for backend writes (predictions ledger, alert eval)."
        )
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def get_user_client(access_token: str) -> Client:
    """Per-request client authenticated as the end user via their JWT.

    RLS policies evaluate `auth.uid()` from the JWT `sub` claim, so this
    client can only see rows the user owns. Cheap to construct — don't
    cache, the token changes per request.
    """
    if not is_configured():
        raise RuntimeError("get_user_client() called without SUPABASE_URL configured")
    # Prefer the anon key as the apikey header (PostgREST requires *some*
    # apikey); the bearer Authorization header is what makes RLS fire.
    key = SUPABASE_ANON_KEY or SUPABASE_SERVICE_ROLE_KEY
    client: Client = create_client(SUPABASE_URL, key)
    client.postgrest.auth(access_token)
    return client


def reset_for_tests() -> None:
    """Drop cached admin client — call from test fixtures between cases."""
    get_admin_client.cache_clear()
