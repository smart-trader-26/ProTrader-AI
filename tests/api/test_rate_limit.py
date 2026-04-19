"""
Rate limit — slowapi 429 after the per-route quota is exhausted.

Predict is capped at 10/minute. We patch the sync task to be instant so
the loop runs fast, then prove that the 11th request comes back 429.

We reset the limiter between tests via `_reset_limiter` so quotas don't
bleed across cases.
"""

from __future__ import annotations

import pytest

from api.rate_limit import limiter


@pytest.fixture(autouse=True)
def _reset_limiter():
    limiter.reset()
    yield
    limiter.reset()


def _patch_predict_to_noop(monkeypatch):
    from workers.tasks import TASK_REGISTRY

    monkeypatch.setattr(TASK_REGISTRY["predict"], "sync_fn", lambda **kw: {"ok": True})


def test_predict_429_after_quota(client, monkeypatch):
    _patch_predict_to_noop(monkeypatch)

    accepted = 0
    rate_limited = 0
    for _ in range(15):
        r = client.post("/api/v1/stocks/TEST.NS/predict", json={})
        if r.status_code == 202:
            accepted += 1
        elif r.status_code == 429:
            rate_limited += 1

    assert accepted == 10, f"expected 10 accepted, got {accepted}"
    assert rate_limited >= 1, "expected at least one 429"


def test_backtest_429_after_quota(client, monkeypatch):
    from workers.tasks import TASK_REGISTRY

    monkeypatch.setattr(TASK_REGISTRY["backtest"], "sync_fn", lambda **kw: {"ok": True})

    accepted = 0
    rate_limited = 0
    for _ in range(25):
        r = client.post("/api/v1/stocks/TEST.NS/backtest", json={})
        if r.status_code == 202:
            accepted += 1
        elif r.status_code == 429:
            rate_limited += 1

    assert accepted == 20
    assert rate_limited >= 1
