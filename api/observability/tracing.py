"""
OpenTelemetry trace setup (B7.3).

Configures the OTLP exporter so traces span FastAPI → Celery → Supabase.
Gated on ``OTEL_EXPORTER_OTLP_ENDPOINT`` — when unset, tracing is silently
skipped.

Compatible free-tier endpoints:
  - Grafana Cloud:  https://otlp-gateway-prod-xx.grafana.net/otlp
  - Honeycomb:      https://api.honeycomb.io

Set via environment:
    OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp-gateway-prod-us-east-0.grafana.net/otlp
    OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64>
    OTEL_SERVICE_NAME=protrader-api
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def init_tracing() -> bool:
    """
    Initialise OpenTelemetry with OTLP export. Returns True if tracing
    was activated.

    Safe to call multiple times — the global TracerProvider is set once.
    """
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if not endpoint:
        log.debug("OTEL_EXPORTER_OTLP_ENDPOINT not set — tracing disabled")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        service_name = os.environ.get("OTEL_SERVICE_NAME", "protrader-api")

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        exporter = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)

        # Auto-instrument FastAPI — adds spans for every request
        FastAPIInstrumentor().instrument()

        # Optional: instrument Celery
        try:
            from opentelemetry.instrumentation.celery import CeleryInstrumentor
            CeleryInstrumentor().instrument()
        except ImportError:
            pass

        # Optional: instrument httpx (outbound API calls)
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
            HTTPXClientInstrumentor().instrument()
        except ImportError:
            pass

        # Optional: instrument requests (outbound API calls)
        try:
            from opentelemetry.instrumentation.requests import RequestsInstrumentor
            RequestsInstrumentor().instrument()
        except ImportError:
            pass

        log.info("OpenTelemetry tracing initialised → %s", endpoint)
        return True
    except ImportError as e:
        log.debug("OTLP deps not installed (%s) — tracing disabled", e)
        return False
    except Exception as exc:
        log.warning("OpenTelemetry init failed: %s", exc)
        return False
