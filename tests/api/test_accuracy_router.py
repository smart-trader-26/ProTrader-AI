"""Accuracy router — uses an isolated SQLite ledger via monkeypatched DB path."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from schemas.prediction import PredictionBundle, PredictionPoint
from services import ledger_service


def test_accuracy_window_returns_zero_when_empty(client, tmp_path, monkeypatch):
    db = tmp_path / "ledger.sqlite"
    monkeypatch.setattr(ledger_service, "DEFAULT_DB_PATH", db)

    r = client.get("/api/v1/accuracy", params={"ticker": "EMPTY.NS", "days": 30})
    assert r.status_code == 200
    body = r.json()
    assert body["n_resolved"] == 0


def test_recent_returns_logged_rows(client, tmp_path, monkeypatch):
    db = tmp_path / "ledger.sqlite"
    monkeypatch.setattr(ledger_service, "DEFAULT_DB_PATH", db)

    bundle = PredictionBundle(
        ticker="LOG.NS",
        made_at=datetime.now(UTC),
        model_version="hybrid-v1",
        horizon_days=2,
        points=[
            PredictionPoint(
                target_date=date.today() + timedelta(days=1),
                pred_price=101.0,
                direction="up",
                prob_up=0.6,
            ),
            PredictionPoint(
                target_date=date.today() + timedelta(days=2),
                pred_price=102.0,
                direction="up",
                prob_up=0.62,
            ),
        ],
    )
    ledger_service.log_prediction(bundle, anchor_price=100.0, db_path=db)

    r = client.get("/api/v1/accuracy/recent", params={"ticker": "LOG.NS", "limit": 10})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 2
    assert rows[0]["ticker"] == "LOG.NS"
