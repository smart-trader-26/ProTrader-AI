"""Liveness + readiness probes — must always be 200 OK."""

from __future__ import annotations


def test_healthz_returns_ok(client):
    r = client.get("/api/v1/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "ts" in body


def test_readyz_returns_ready(client):
    """readyz returns 200 when all configured deps are reachable, 503 otherwise.
    The test environment may or may not have DATABASE_URL / REDIS_URL wired,
    so we only assert the route is live and the schema is well-formed."""
    r = client.get("/api/v1/readyz")
    assert r.status_code in (200, 503)
    body = r.json()
    assert body["status"] in ("ready", "degraded")
    assert "checks" in body and "database" in body["checks"] and "redis" in body["checks"]


def test_openapi_schema_lists_v1_routes(client):
    """B1.6: OpenAPI must expose every router so codegen produces a usable client."""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    expected = [
        "/api/v1/healthz",
        "/api/v1/stocks",
        "/api/v1/stocks/{ticker}/ohlcv",
        "/api/v1/stocks/{ticker}/fundamentals",
        "/api/v1/stocks/{ticker}/sentiment",
        "/api/v1/stocks/{ticker}/predict",
        "/api/v1/stocks/{ticker}/backtest",
        "/api/v1/jobs/{job_id}",
        "/api/v1/accuracy",
        "/api/v1/models/active",
    ]
    for p in expected:
        assert p in paths, f"missing OpenAPI path: {p}"
