"""
Request-id correlation middleware (B7.2).

Generates a unique ``request_id`` per HTTP request and binds it to
structlog's context vars so every log line from that request carries the
same ID. Also sets the ``X-Request-ID`` response header for client-side
correlation.

If the client sends ``X-Request-ID``, it is reused (useful for tracing
across API gateway → backend → Celery worker).

Usage — add to FastAPI:

    from api.observability.middleware import RequestIdMiddleware
    app.add_middleware(RequestIdMiddleware)
"""

from __future__ import annotations

import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

_REQUEST_ID_HEADER = "X-Request-ID"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assign and propagate a per-request correlation ID."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(_REQUEST_ID_HEADER) or str(uuid.uuid4())

        # Bind into structlog context — every log line inside this request
        # will carry ``request_id``.
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        response = await call_next(request)
        response.headers[_REQUEST_ID_HEADER] = request_id
        return response
