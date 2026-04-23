"""
Structured logging configuration (B7.2).

Sets up ``structlog`` with JSON output for production (Railway / Render
ingest it natively) and human-readable coloured output for local dev.

Usage:
    from api.observability.logging import setup_logging
    setup_logging()  # call once at app startup

All log output is ``structlog`` — FastAPI, Celery, and application code
share the same processor chain. A ``request_id`` is bound per-request
via middleware (see ``api.observability.middleware``).
"""

from __future__ import annotations

import logging
import os
import sys

import structlog


def setup_logging(*, json_logs: bool | None = None, log_level: str | None = None) -> None:
    """
    Configure structlog + stdlib logging for the whole process.

    Parameters
    ----------
    json_logs : bool | None
        Force JSON (True) or console (False). Defaults to JSON when
        ``ENVIRONMENT`` != ``development``.
    log_level : str | None
        Root log level. Defaults to ``LOG_LEVEL`` env var, then ``INFO``.
    """
    if json_logs is None:
        json_logs = os.environ.get("ENVIRONMENT", "production") != "development"
    if log_level is None:
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    timestamper = structlog.processors.TimeStamper(fmt="iso")

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_logs:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, log_level, logging.INFO))

    # Quiet noisy loggers
    for noisy in ("uvicorn.access", "httpcore", "httpx", "hpack"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
