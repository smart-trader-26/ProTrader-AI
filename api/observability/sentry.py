"""
Sentry integration for FastAPI + Celery (B7.1).

Initialisation is gated on ``SENTRY_DSN`` — when unset, Sentry is silently
skipped. This keeps the free dev and CI paths working unchanged.

Usage:
    from api.observability.sentry import init_sentry
    init_sentry()  # call once at app startup

The DSN for the free tier (5K events/mo) is obtained from:
  https://sentry.io → Project Settings → Client Keys (DSN)

Set via environment:
    SENTRY_DSN=https://xxx@oNNN.ingest.sentry.io/NNNNNNN
    SENTRY_ENVIRONMENT=production   # or staging / development
    SENTRY_TRACES_SAMPLE_RATE=0.2   # 0.0–1.0, default 0.2
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def init_sentry() -> bool:
    """
    Initialise Sentry SDK. Returns True if Sentry was activated.

    Safe to call multiple times — ``sentry_sdk.init`` is idempotent.
    """
    dsn = os.environ.get("SENTRY_DSN", "")
    if not dsn:
        log.debug("SENTRY_DSN not set — Sentry disabled")
        return False

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        integrations = [
            StarletteIntegration(transaction_style="endpoint"),
            FastApiIntegration(transaction_style="endpoint"),
            LoggingIntegration(
                level=logging.INFO,        # breadcrumbs from INFO+
                event_level=logging.ERROR,  # events from ERROR+
            ),
        ]

        # Optional: Celery integration (only when celery is installed)
        try:
            from sentry_sdk.integrations.celery import CeleryIntegration
            integrations.append(CeleryIntegration())
        except ImportError:
            pass

        traces_rate = float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.2"))
        environment = os.environ.get("SENTRY_ENVIRONMENT", "production")

        sentry_sdk.init(
            dsn=dsn,
            integrations=integrations,
            traces_sample_rate=max(0.0, min(1.0, traces_rate)),
            environment=environment,
            send_default_pii=False,
            # Attach request bodies only for errors, not for performance traces.
            max_request_body_size="medium",
        )
        log.info("Sentry initialised (env=%s, traces=%.0f%%)", environment, traces_rate * 100)
        return True
    except ImportError:
        log.debug("sentry-sdk not installed — Sentry disabled")
        return False
    except Exception as exc:
        log.warning("Sentry init failed: %s", exc)
        return False
