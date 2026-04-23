"""
Tests for api.observability — offline, no Sentry/OTLP endpoints needed.
"""

from __future__ import annotations

import logging
import os
from unittest.mock import patch

import pytest


# ─── B7.1: Sentry ───────────────────────────────────────


def test_sentry_skips_when_dsn_unset(monkeypatch):
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    from api.observability.sentry import init_sentry
    assert init_sentry() is False


def test_sentry_activates_when_dsn_set(monkeypatch):
    monkeypatch.setenv("SENTRY_DSN", "https://examplePublicKey@o0.ingest.sentry.io/0")
    monkeypatch.setenv("SENTRY_ENVIRONMENT", "test")
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "0.0")  # don't send anything
    from api.observability.sentry import init_sentry
    result = init_sentry()
    assert result is True
    # Shut down Sentry immediately to prevent "pending events" on test exit
    import sentry_sdk
    client = sentry_sdk.get_client()
    if client and hasattr(client, 'close'):
        client.close(timeout=0)


# ─── B7.2: structlog ────────────────────────────────────


def test_structlog_setup_succeeds():
    from api.observability.logging import setup_logging
    # Should not raise
    setup_logging(json_logs=False, log_level="WARNING")
    import structlog
    logger = structlog.get_logger("test")
    assert logger is not None


def test_structlog_json_mode():
    from api.observability.logging import setup_logging
    setup_logging(json_logs=True, log_level="INFO")
    import structlog
    logger = structlog.get_logger("test_json")
    assert logger is not None


# ─── B7.2: RequestIdMiddleware ───────────────────────────


@pytest.mark.anyio
async def test_request_id_middleware_sets_header():
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from api.observability.middleware import RequestIdMiddleware

    async def homepage(request):
        return PlainTextResponse("OK")

    app = Starlette(routes=[Route("/", homepage)])
    app.add_middleware(RequestIdMiddleware)

    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "X-Request-ID" in resp.headers
    assert len(resp.headers["X-Request-ID"]) > 10  # UUID length


@pytest.mark.anyio
async def test_request_id_reuses_client_header():
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from api.observability.middleware import RequestIdMiddleware

    async def homepage(request):
        return PlainTextResponse("OK")

    app = Starlette(routes=[Route("/", homepage)])
    app.add_middleware(RequestIdMiddleware)

    client = TestClient(app)
    resp = client.get("/", headers={"X-Request-ID": "my-trace-123"})
    assert resp.headers["X-Request-ID"] == "my-trace-123"


# ─── B7.3: OTLP tracing ─────────────────────────────────


def test_tracing_skips_when_endpoint_unset(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    from api.observability.tracing import init_tracing
    assert init_tracing() is False
