"""
Celery app factory (B2.1).

Single Celery process / app shared by the FastAPI backend (which enqueues
jobs) and the worker / beat processes (which execute them). Broker +
result backend are both Redis — Upstash's free tier is enough for the
roadmap's expected load.

`IS_AVAILABLE` flips on when `REDIS_URL` is set in the environment. When
it's off, importing this module is still safe — the app object exists with
a placeholder broker — but `api/jobs.py` will pick the in-process backend
instead of routing through Celery.
"""

from __future__ import annotations

from celery import Celery
from celery.schedules import crontab

from config.settings import REDIS_URL

IS_AVAILABLE: bool = bool(REDIS_URL)

# Always construct the app — even without a broker — so `@app.task`
# decorators in workers/tasks.py work for unit tests / typing.
_BROKER = REDIS_URL or "redis://localhost:6379/0"

app = Celery(
    "protrader",
    broker=_BROKER,
    backend=_BROKER,
    include=["workers.tasks"],
)

app.conf.update(
    # JSON serialization: Pydantic models are dumped via .model_dump(mode='json')
    # inside each task, so the result backend never sees Python objects that
    # would need pickle. Safer cross-version.
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Retry-friendly defaults:
    task_acks_late=True,                    # re-queue on worker crash
    worker_prefetch_multiplier=1,           # one heavy job at a time per worker
    task_reject_on_worker_lost=True,
    result_expires=60 * 60 * 6,             # results live 6 h in Redis
    broker_connection_retry_on_startup=True,
)

# Beat schedule lives in its own module so the worker process doesn't need
# to import scheduling logic when it's only running tasks.
from workers.beat_schedule import build_beat_schedule  # noqa: E402

app.conf.beat_schedule = build_beat_schedule(crontab)
