"""
Tests for GET /api/v1/swing/scan (P4-2 swing screener).

The service layer is monkeypatched — scan mechanics are covered in
tests/services/test_swing_scan_service.py.
"""

from __future__ import annotations

from datetime import UTC, datetime

from schemas.swing import SwingScanResult, SwingScanRow


def _fake_result() -> SwingScanResult:
    return SwingScanResult(
        generated_at=datetime.now(UTC),
        horizon_days=30,
        trained=True,
        tau_star=0.63,
        expected_hit_rate=0.606,
        base_rate=0.58,
        coverage=0.056,
        n_scanned=2,
        n_fired=1,
        rows=[
            SwingScanRow(ticker="AAA.NS", signal="UP", prob_up=0.71,
                         conviction=0.6, last_close=150.0, asof="2026-06-09"),
            SwingScanRow(ticker="BBB.NS", signal="NEUTRAL", prob_up=0.41,
                         conviction=0.0, last_close=80.0, asof="2026-06-09"),
        ],
    )


def test_swing_scan_returns_result(client, monkeypatch):
    captured = {}

    def fake_scan(refresh: bool = False):
        captured["refresh"] = refresh
        return _fake_result()

    monkeypatch.setattr("services.swing_scan_service.scan", fake_scan)

    resp = client.get("/api/v1/swing/scan")
    assert resp.status_code == 200
    body = resp.json()
    assert captured["refresh"] is False
    assert body["trained"] is True
    assert body["n_fired"] == 1
    assert body["rows"][0]["ticker"] == "AAA.NS"
    assert body["rows"][0]["signal"] == "UP"
    assert body["expected_hit_rate"] == 0.606


def test_swing_scan_refresh_param(client, monkeypatch):
    captured = {}

    def fake_scan(refresh: bool = False):
        captured["refresh"] = refresh
        return _fake_result()

    monkeypatch.setattr("services.swing_scan_service.scan", fake_scan)

    resp = client.get("/api/v1/swing/scan?refresh=true")
    assert resp.status_code == 200
    assert captured["refresh"] is True
