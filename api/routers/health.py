"""Liveness + readiness probes (Railway / Render health-checks hit these)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, Response

from config.settings import REDIS_URL, SUPABASE_ANON_KEY, SUPABASE_URL
from db import supabase_client as sb

log = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/healthz", summary="Liveness probe")
def healthz() -> dict:
    return {"status": "ok", "ts": datetime.now(UTC).isoformat()}


def _check_db() -> tuple[str, str | None]:
    """Ping the Supabase PostgREST endpoint — framework mode has no direct PG."""
    if not sb.is_configured():
        return "unconfigured", None
    try:
        # Hit the PostgREST root; a 200 means the project is live and the
        # anon key is valid. We do NOT issue a query — keeps the probe cheap.
        r = httpx.get(
            f"{SUPABASE_URL}/rest/v1/",
            headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {SUPABASE_ANON_KEY}"},
            timeout=3.0,
        )
        if r.status_code >= 500:
            return "error", f"HTTP {r.status_code}"
        return "ok", None
    except Exception as e:  # DNS, TLS, timeout — all surface here
        return "error", f"{type(e).__name__}: {e}"


def _check_redis() -> tuple[str, str | None]:
    if not REDIS_URL:
        return "unconfigured", None
    try:
        import redis  # lazy — only installed when Celery path is used

        client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=2)
        client.ping()
        return "ok", None
    except Exception as e:
        return "error", f"{type(e).__name__}: {e}"


@router.get("/readyz", summary="Readiness probe")
def readyz(response: Response) -> dict:
    """Pings Supabase + Redis when configured. 503s if either is
    configured-but-down so a deploy's health check fails fast instead of
    silently serving broken routes. Unconfigured deps report as
    'unconfigured' and don't fail the probe (dev mode)."""
    db_status, db_err = _check_db()
    redis_status, redis_err = _check_redis()

    healthy = db_status != "error" and redis_status != "error"
    if not healthy:
        response.status_code = 503
        log.warning("readyz: db=%s redis=%s", db_err or db_status, redis_err or redis_status)

    return {
        "status": "ready" if healthy else "degraded",
        "ts": datetime.now(UTC).isoformat(),
        "checks": {
            "database": {"status": db_status, "error": db_err},
            "redis":    {"status": redis_status, "error": redis_err},
        },
    }
