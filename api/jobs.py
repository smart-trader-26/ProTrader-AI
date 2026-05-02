"""
Job store (B2) — pluggable backend.

Two implementations sit behind one interface:

  • `InProcessJobStore` — thread-pool, no external deps. Default in dev /
                          tests / when REDIS_URL is unset.
  • `CeleryJobStore`    — enqueues to the Redis-backed Celery broker.
                          Tasks run on a separate `celery worker` process.

`get_store()` picks the right implementation at runtime based on
`config.settings.REDIS_URL`. Routers always call `store.enqueue(kind,
**kwargs)` — they never name a callable, so swapping backends is a config
change, not a code change.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

JobStatus = Literal["queued", "running", "succeeded", "failed"]

_MAX_JOBS = 512
_TTL_SECONDS = 60 * 30  # finished jobs cleared 30 min after completion


@dataclass
class JobRecord:
    id: str
    kind: str
    status: JobStatus = "queued"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: Any = None
    error: str | None = None

    def to_public(self) -> dict:
        return asdict(self)


class JobStore(Protocol):
    """Common surface implemented by every backend."""

    def enqueue(self, kind: str, **kwargs) -> JobRecord: ...
    def get(self, job_id: str) -> JobRecord | None: ...
    def shutdown(self) -> None: ...


class InProcessJobStore:

    def __init__(self, max_workers: int = 4):
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="api-job"
        )

    def enqueue(self, kind: str, **kwargs) -> JobRecord:
        from workers.tasks import TASK_REGISTRY

        if kind not in TASK_REGISTRY:
            raise ValueError(f"unknown job kind: {kind!r}")
        fn = TASK_REGISTRY[kind].sync_fn

        job = JobRecord(id=uuid.uuid4().hex, kind=kind)
        with self._lock:
            self._gc_locked()
            self._jobs[job.id] = job
        self._executor.submit(self._run, job.id, fn, kwargs)
        return job

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _run(self, job_id: str, fn, kwargs: dict) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return
            rec.status = "running"
            rec.started_at = datetime.now(UTC)
        try:
            result = fn(**kwargs)
            with self._lock:
                rec = self._jobs.get(job_id)
                if rec is None:
                    return
                rec.result = result
                rec.status = "succeeded"
                rec.finished_at = datetime.now(UTC)
        except Exception as e:  # noqa: BLE001
            with self._lock:
                rec = self._jobs.get(job_id)
                if rec is None:
                    return
                rec.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                rec.status = "failed"
                rec.finished_at = datetime.now(UTC)

    def _gc_locked(self) -> None:
        now = time.time()
        expired = [
            jid
            for jid, rec in self._jobs.items()
            if rec.finished_at is not None
            and (now - rec.finished_at.timestamp()) > _TTL_SECONDS
        ]
        for jid in expired:
            self._jobs.pop(jid, None)
        if len(self._jobs) > _MAX_JOBS:
            ordered = sorted(self._jobs.items(), key=lambda kv: kv[1].created_at)
            for jid, rec in ordered:
                if len(self._jobs) <= _MAX_JOBS:
                    break
                if rec.status in ("succeeded", "failed"):
                    self._jobs.pop(jid, None)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


_CELERY_STATE_MAP: dict[str, JobStatus] = {
    "PENDING":  "queued",
    "RECEIVED": "queued",
    "STARTED":  "running",
    "RETRY":    "running",
    "SUCCESS":  "succeeded",
    "FAILURE":  "failed",
    "REVOKED":  "failed",
}


class CeleryJobStore:
    """
    Redis-backed backend. Queues work to `celery worker`; reads results via Celery.
    Stores a tiny `protrader:jobs:{id}` hash in Redis (kind + created_at, 6h TTL)
    so the jobs endpoint returns the same shape as InProcessJobStore.
    """

    META_PREFIX = "protrader:jobs:"
    META_TTL_SECONDS = 60 * 60 * 6

    def __init__(self):
        from workers.celery_app import app

        self._app = app
        # `app.backend.client` is the raw Redis connection (StrictRedis).
        try:
            self._redis = app.backend.client
        except Exception:  # pragma: no cover
            self._redis = None

    def enqueue(self, kind: str, **kwargs) -> JobRecord:
        from workers.tasks import TASK_REGISTRY

        if kind not in TASK_REGISTRY:
            raise ValueError(f"unknown job kind: {kind!r}")
        task = TASK_REGISTRY[kind].celery_task

        async_result = task.apply_async(kwargs=_jsonable_kwargs(kwargs))
        created = datetime.now(UTC).isoformat()
        if self._redis is not None:
            self._redis.hset(
                self.META_PREFIX + async_result.id,
                mapping={"kind": kind, "created_at": created},
            )
            self._redis.expire(self.META_PREFIX + async_result.id, self.META_TTL_SECONDS)
        return JobRecord(
            id=async_result.id,
            kind=kind,
            status="queued",
            created_at=datetime.fromisoformat(created),
        )

    def get(self, job_id: str) -> JobRecord | None:
        from celery.result import AsyncResult

        meta: dict[str, str] = {}
        if self._redis is not None:
            raw = self._redis.hgetall(self.META_PREFIX + job_id) or {}
            meta = {
                (k.decode() if isinstance(k, bytes) else k):
                (v.decode() if isinstance(v, bytes) else v)
                for k, v in raw.items()
            }
        if not meta:
            # Never stored, or TTL expired — either way treat as 404.
            return None

        ar = AsyncResult(job_id, app=self._app)
        status = _CELERY_STATE_MAP.get(ar.state, "queued")
        result = ar.result if status == "succeeded" else None
        error = repr(ar.result) if status == "failed" else None

        return JobRecord(
            id=job_id,
            kind=meta["kind"],
            status=status,
            created_at=datetime.fromisoformat(meta["created_at"]),
            result=result,
            error=error,
        )

    def shutdown(self) -> None:
        return None  # Celery owns its connection pool; nothing to do


def _jsonable_kwargs(kwargs: dict) -> dict:
    """Coerce Pydantic models and dates to JSON-serializable types for Celery."""
    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        if hasattr(v, "model_dump"):
            out[k] = v.model_dump(mode="json")
        elif hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


_STORE: JobStore | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> JobStore:
    """Singleton: CeleryJobStore when REDIS_URL is set, else InProcessJobStore."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            from workers.celery_app import IS_AVAILABLE

            _STORE = CeleryJobStore() if IS_AVAILABLE else InProcessJobStore()
        return _STORE


def reset_store_for_tests() -> None:
    """Reset the singleton to a fresh InProcessJobStore for test isolation."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is not None:
            _STORE.shutdown()
        _STORE = InProcessJobStore()
