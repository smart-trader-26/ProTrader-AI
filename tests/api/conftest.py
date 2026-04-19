"""
Test fixtures for the FastAPI app.

Each test gets:
  • A fresh `JobStore` (no cross-test pollution).
  • A `TestClient` bound to the real ASGI app — no mocking of FastAPI itself.

We do NOT mock the underlying services (CLAUDE.md §2.4). For tests that
would normally hit the network (yfinance, FinBERT), we override individual
service functions via `app.dependency_overrides` or `monkeypatch` against
the *service module*, not the data source.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import jobs as jobs_module
from api.main import create_app


@pytest.fixture
def app():
    jobs_module.reset_store_for_tests()
    return create_app()


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c
