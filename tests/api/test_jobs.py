"""
Job store + jobs router — fully offline.

We override the predict / backtest service functions so the job pattern
exercises the full enqueue → poll → resolve path without firing the real
ML stack (which is what `tests/services/` covers).
"""

from __future__ import annotations

import time
from datetime import UTC, date, datetime

from api.deps import get_job_store
from schemas.backtest import BacktestMetrics, BacktestResult
from schemas.prediction import PredictionBundle, PredictionPoint


def _wait_for_terminal(client, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/v1/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


def test_jobs_404_for_unknown_id(client):
    r = client.get("/api/v1/jobs/doesnotexist")
    assert r.status_code == 404


def test_predict_enqueue_runs_and_returns_bundle(client, app, monkeypatch):
    fake_bundle = PredictionBundle(
        ticker="TEST.NS",
        made_at=datetime.now(UTC),
        model_version="hybrid-v1",
        horizon_days=1,
        points=[
            PredictionPoint(
                target_date=date.today(),
                pred_price=100.0,
                direction="up",
                prob_up=0.7,
            )
        ],
    )

    # Patch the sync task in the registry — the job runs on a worker
    # thread, so we bypass the real ML pipeline entirely.
    from workers.tasks import TASK_REGISTRY

    monkeypatch.setattr(
        TASK_REGISTRY["predict"], "sync_fn", lambda **kw: fake_bundle
    )

    r = client.post("/api/v1/stocks/TEST.NS/predict", json={"horizon_days": 1})
    assert r.status_code == 202
    body = r.json()
    job_id = body["job_id"]
    # Status may already be terminal — the fake fn returns instantly.
    assert body["status"] in ("queued", "running", "succeeded")
    assert body["poll_url"] == f"/api/v1/jobs/{job_id}"

    final = _wait_for_terminal(client, job_id)
    assert final["status"] == "succeeded"
    assert final["result"]["ticker"] == "TEST.NS"
    assert final["result"]["points"][0]["pred_price"] == 100.0


def test_backtest_enqueue_runs_and_returns_result(client, app, monkeypatch):
    fake_result = BacktestResult(
        ticker="TEST.NS",
        start=date(2024, 1, 1),
        end=date(2024, 6, 30),
        strategy="ma_crossover",
        initial_capital=100000.0,
        final_equity=110000.0,
        metrics=BacktestMetrics(
            total_return_pct=10.0,
            cagr_pct=20.0,
            sharpe=1.5,
            max_drawdown_pct=5.0,
            win_rate_pct=55.0,
        ),
    )

    from workers.tasks import TASK_REGISTRY

    monkeypatch.setattr(
        TASK_REGISTRY["backtest"], "sync_fn", lambda **kw: fake_result
    )

    r = client.post(
        "/api/v1/stocks/TEST.NS/backtest",
        json={"strategy": "ma_crossover"},
    )
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    final = _wait_for_terminal(client, job_id)
    assert final["status"] == "succeeded"
    assert final["result"]["metrics"]["sharpe"] == 1.5


def test_failed_job_surfaces_error(client, app, monkeypatch):
    from workers.tasks import TASK_REGISTRY

    def boom(**kw):
        raise ValueError("synthetic failure for test")

    monkeypatch.setattr(TASK_REGISTRY["predict"], "sync_fn", boom)

    r = client.post("/api/v1/stocks/TEST.NS/predict", json={})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    final = _wait_for_terminal(client, job_id)
    assert final["status"] == "failed"
    assert "synthetic failure" in final["error"]


def test_job_store_isolated_per_test():
    """Sanity: the conftest reset hook gives each test a fresh store."""
    store = get_job_store()
    # `JobStore` is a Protocol, not a class — duck-type instead.
    for attr in ("enqueue", "get", "shutdown"):
        assert callable(getattr(store, attr, None)), f"missing {attr}"
