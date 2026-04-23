"""
FastAPI app factory (B1.1 / B3 wiring).

  uvicorn api.main:app --reload --port 8000

Surfaces:
  /api/v1/healthz                          — liveness
  /api/v1/readyz                           — readiness
  /api/v1/me                               — current Supabase user (B3.4)
  /api/v1/stocks                           — search + OHLCV + fundamentals (B1.2)
  /api/v1/stocks/{ticker}/sentiment[/v2]   — sentiment + v2 ensemble (B1.3)
  /api/v1/stocks/{ticker}/predict          — enqueue prediction (B1.4) → job_id
  /api/v1/stocks/{ticker}/backtest         — enqueue backtest (B1.5)  → job_id
  /api/v1/jobs/{id}                        — poll job status / result
  /api/v1/accuracy[/recent]                — A7 ledger rollups
  /api/v1/models/active                    — active model version (B6.3)
  /api/v1/watchlists                       — per-user CRUD (B3)
  /api/v1/alerts                           — per-user CRUD (B3)
  /api/v1/ws/prices?tickers=               — live price stream stub (B5.3)
  /docs                                    — Swagger UI (B1.6)
  /openapi.json                            — schema for `openapi-typescript` codegen

Hard invariants the API inherits from CLAUDE.md §2:
  • No `streamlit` imports — services and schemas are the contract.
  • Secrets only via `config.settings._get_secret()`.
  • Free-tier first — every endpoint works without paid keys, except v2
    ensemble (HF_TOKEN) which 503s cleanly. Auth-only endpoints 401 in
    dev (when SUPABASE_JWT_SECRET is unset).
"""

from __future__ import annotations

# Silence third-party noise BEFORE any transitive import pulls them in:
#   • TF: oneDNN + CPU-feature notices on the `import tensorflow` path
#     triggered by data/* modules that import FinBERT transitively.
#   • Keras: `np.object` FutureWarning on `tf-keras` 2.15.
#   • Streamlit: "No runtime found" INFO spam — every `@st.cache_*` decorator
#     on an imported data/* module logs when called outside `streamlit run`.
import os as _os
import warnings as _warnings

_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")       # FATAL only
_os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
# Streamlit re-applies its config-driven log level during `import streamlit`,
# so patching `streamlit.logger.set_log_level` at runtime is racy. Setting the
# env var Streamlit's config reads is durable across any import order.
_os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
# (?s) = DOTALL so `.*` spans the leading newlines in google.generativeai's
# FutureWarning message. Same trick for np.object spillover from tf-keras.
_warnings.filterwarnings("ignore", category=FutureWarning, module="keras")
_warnings.filterwarnings("ignore", message=r"(?s).*np\.object.*")
_warnings.filterwarnings("ignore", message=r"(?s).*google\.generativeai.*")
_warnings.filterwarnings("ignore", message=r"(?s).*google-generativeai.*")

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

# Streamlit emits "No runtime found, using MemoryCacheStorageManager" whenever
# an `@st.cache_*` decorated function is instantiated outside `streamlit run`.
# Transitively imported data/* modules trip this. Setting the logger level or
# STREAMLIT_LOGGER_LEVEL is racy — streamlit re-applies its own config during
# `import streamlit`. A logging filter on the streamlit handler is durable.


class _DropStreamlitNoRuntime(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "No runtime found" not in record.getMessage()


try:
    import streamlit  # noqa: F401 — force init so handlers exist
    import streamlit.logger as _st_logger

    _st_filter = _DropStreamlitNoRuntime()
    for _name, _lg in list(logging.Logger.manager.loggerDict.items()):
        if _name.startswith("streamlit") and isinstance(_lg, logging.Logger):
            _lg.addFilter(_st_filter)
            for _h in _lg.handlers:
                _h.addFilter(_st_filter)
    # Cover loggers streamlit creates *after* our init by patching the factory.
    _orig_get_logger = _st_logger.get_logger

    def _patched_get_logger(name: str) -> logging.Logger:
        lg = _orig_get_logger(name)
        lg.addFilter(_st_filter)
        return lg

    _st_logger.get_logger = _patched_get_logger
except Exception:  # streamlit not installed in some deploy envs — fine
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.rate_limit import limiter
from api.routers import (
    accuracy as accuracy_router,
)
from api.routers import (
    alerts as alerts_router,
)
from api.routers import (
    auth as auth_router,
)
from api.routers import (
    backtest as backtest_router,
)
from api.routers import (
    health as health_router,
)
from api.routers import (
    jobs as jobs_router,
)
from api.routers import (
    models as models_router,
)
from api.routers import (
    predict as predict_router,
)
from api.routers import (
    sentiment as sentiment_router,
)
from api.routers import (
    stocks as stocks_router,
)
from api.routers import (
    watchlists as watchlists_router,
)
from api.routers import (
    ws as ws_router,
)

API_PREFIX = "/api/v1"

# CORS: in dev allow Vite (5173) + Next.js dev (3000); in prod the deploy env
# must override via `PROTRADER_CORS_ORIGINS`.
_default_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
_env_origins = [
    o.strip()
    for o in os.environ.get("PROTRADER_CORS_ORIGINS", "").split(",")
    if o.strip()
]
_ALLOWED_ORIGINS = _env_origins or _default_origins


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # ── B7: Observability bootstrap (all gated on env vars) ──
    from api.observability.logging import setup_logging
    from api.observability.sentry import init_sentry
    from api.observability.tracing import init_tracing

    setup_logging()    # B7.2 — structlog (always; JSON in prod, console in dev)
    init_sentry()      # B7.1 — Sentry (only when SENTRY_DSN is set)
    init_tracing()     # B7.3 — OTLP traces (only when OTEL_EXPORTER_OTLP_ENDPOINT is set)

    yield
    # Drain in-process job workers cleanly so SIGTERM doesn't lose jobs
    # in flight. With Celery (B2) this is a no-op.
    from api.jobs import _STORE

    if _STORE is not None:
        _STORE.shutdown()


def _rate_limit_handler(request, exc: RateLimitExceeded):
    """Map slowapi's RateLimitExceeded to a JSON 429."""
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=429,
        content={"detail": f"rate limit exceeded: {exc.detail}"},
        headers={"Retry-After": "60"},
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="ProTrader AI API",
        version="0.2.0",
        description=(
            "Backend for the ProTrader AI Next.js frontend (Track B). "
            "Wraps the same `services/` layer as the Streamlit UI so every "
            "feature stays consistent across both surfaces."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=_lifespan,
    )

    # Rate limiting — slowapi reads `app.state.limiter` from the middleware.
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
    app.add_middleware(SlowAPIMiddleware)

    # B7.2 — request-id correlation (binds to structlog contextvars)
    from api.observability.middleware import RequestIdMiddleware
    app.add_middleware(RequestIdMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers — public.
    app.include_router(health_router.router,    prefix=API_PREFIX)
    app.include_router(stocks_router.router,    prefix=API_PREFIX)
    app.include_router(sentiment_router.router, prefix=API_PREFIX)
    app.include_router(predict_router.router,   prefix=API_PREFIX)
    app.include_router(backtest_router.router,  prefix=API_PREFIX)
    app.include_router(jobs_router.router,      prefix=API_PREFIX)
    app.include_router(accuracy_router.router,  prefix=API_PREFIX)
    app.include_router(models_router.router,    prefix=API_PREFIX)
    app.include_router(ws_router.router,        prefix=API_PREFIX)

    # Routers — auth-required (B3).
    app.include_router(auth_router.router,       prefix=API_PREFIX)
    app.include_router(watchlists_router.router, prefix=API_PREFIX)
    app.include_router(alerts_router.router,     prefix=API_PREFIX)

    # Root redirect — hitting `/` in a browser should land on Swagger UI.
    @app.get("/", include_in_schema=False)
    def _root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    logging.getLogger("uvicorn").info("ProTrader API ready at %s", API_PREFIX)
    return app


app = create_app()
