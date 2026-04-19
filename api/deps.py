"""
FastAPI dependencies — injected via `Depends(...)`.

Centralised here so routers stay test-overridable. To stub the job store in
tests, override `get_job_store`:

    app.dependency_overrides[get_job_store] = lambda: my_fake_store
"""

from __future__ import annotations

from api.jobs import JobStore, get_store


def get_job_store() -> JobStore:
    return get_store()
