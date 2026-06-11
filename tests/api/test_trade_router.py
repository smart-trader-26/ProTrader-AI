"""
Tests for the P5 Trade Desk router. Service layer is monkeypatched —
queue/broker mechanics are covered in tests/services/.
"""

from __future__ import annotations

from datetime import UTC, datetime

from schemas.trade import (
    BrokerStatus,
    ProposalCreateResponse,
    SkippedTicker,
    TradeProposal,
)


def _proposal(status: str = "PROPOSED", **over) -> TradeProposal:
    base = dict(
        id=1,
        created_at=datetime.now(UTC),
        ticker="AAA.NS",
        side="BUY",
        qty=166,
        entry_price=150.0,
        capital_required=24_900.0,
        stop_price=135.0,
        target_price=None,
        exit_by="2026-07-22",
        engine="swing30d",
        prob_up=0.71,
        tau_star=0.63,
        expected_hit_rate=0.606,
        status=status,
        note="",
    )
    base.update(over)
    return TradeProposal(**base)


def test_broker_status_defaults_to_dry_run(client):
    resp = client.get("/api/v1/trade/broker/status")
    assert resp.status_code == 200
    body = resp.json()
    # Without KITE_* keys + LIVE_TRADING the backend must report simulation.
    assert body["broker"] == "dry_run"
    assert body["live_trading"] is False


def test_create_proposals(client, monkeypatch):
    def fake_create(tickers, capital_per_slot=None):
        assert tickers == ["AAA.NS", "BBB.NS"]
        assert capital_per_slot == 30000
        return ProposalCreateResponse(
            proposals=[_proposal()],
            skipped=[SkippedTicker(ticker="BBB.NS", reason="NEUTRAL", prob_up=0.4)],
        )

    monkeypatch.setattr("services.trade_proposal_service.create_proposals", fake_create)
    resp = client.post(
        "/api/v1/trade/proposals",
        json={"tickers": ["AAA.NS", "BBB.NS"], "capital_per_slot": 30000},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["proposals"][0]["ticker"] == "AAA.NS"
    assert body["skipped"][0]["reason"] == "NEUTRAL"


def test_create_requires_tickers(client):
    resp = client.post("/api/v1/trade/proposals", json={"tickers": []})
    assert resp.status_code == 422


def test_approve_and_conflict(client, monkeypatch):
    monkeypatch.setattr(
        "services.trade_proposal_service.approve",
        lambda pid: _proposal(status="PLACED", dry_run=True,
                              broker="dry_run", broker_order_id="DRY-ABC123"),
    )
    resp = client.post("/api/v1/trade/proposals/1/approve")
    assert resp.status_code == 200
    assert resp.json()["status"] == "PLACED"

    def conflict(pid):
        raise ValueError("proposal 1 is PLACED, not PROPOSED")

    monkeypatch.setattr("services.trade_proposal_service.approve", conflict)
    resp = client.post("/api/v1/trade/proposals/1/approve")
    assert resp.status_code == 409


def test_approve_broker_error_returns_502(client, monkeypatch):
    from services.broker_service import BrokerError

    def boom(pid):
        raise BrokerError("token expired")

    monkeypatch.setattr("services.trade_proposal_service.approve", boom)
    resp = client.post("/api/v1/trade/proposals/1/approve")
    assert resp.status_code == 502
    assert "token expired" in resp.json()["detail"]


def test_reject(client, monkeypatch):
    monkeypatch.setattr(
        "services.trade_proposal_service.reject",
        lambda pid: _proposal(status="REJECTED"),
    )
    resp = client.post("/api/v1/trade/proposals/1/reject")
    assert resp.status_code == 200
    assert resp.json()["status"] == "REJECTED"


def test_list_proposals(client, monkeypatch):
    monkeypatch.setattr(
        "services.trade_proposal_service.list_proposals",
        lambda status=None, limit=100: [_proposal()],
    )
    resp = client.get("/api/v1/trade/proposals")
    assert resp.status_code == 200
    assert resp.json()[0]["ticker"] == "AAA.NS"
