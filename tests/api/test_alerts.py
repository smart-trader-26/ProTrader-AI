"""
Alerts CRUD — auth-required, isolated per user, re-arm semantics.

Uses the in-memory `FakeSupabase` harness so tests stay offline.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest

from tests.api.supabase_fake import install as install_fake_supabase


@pytest.fixture
def jwt_secret(monkeypatch):
    secret = "test-secret-alerts-32chars-min-len"
    monkeypatch.setattr("config.settings.SUPABASE_JWT_SECRET", secret)
    monkeypatch.setattr("api.auth.SUPABASE_JWT_SECRET", secret)
    monkeypatch.setattr("api.rate_limit.SUPABASE_JWT_SECRET", secret)
    return secret


@pytest.fixture
def fake_supabase(monkeypatch):
    return install_fake_supabase(monkeypatch)


def _token(secret, sub="00000000-0000-0000-0000-00000000beef"):
    payload = {
        "sub": sub,
        "email": "a@x",
        "aud": "authenticated",
        "exp": int((datetime.now(UTC) + timedelta(minutes=5)).timestamp()),
        "role": "authenticated",
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def _auth(secret, **kw):
    return {"Authorization": f"Bearer {_token(secret, **kw)}"}


def test_alerts_create_list_update_delete(client, jwt_secret, fake_supabase):
    h = _auth(jwt_secret)

    r = client.post(
        "/api/v1/alerts",
        headers=h,
        json={"ticker": "reliance", "kind": "price_above", "threshold": 1500.0},
    )
    assert r.status_code == 201
    alert = r.json()
    assert alert["ticker"] == "RELIANCE"
    assert alert["active"] is True
    aid = alert["id"]

    r = client.get("/api/v1/alerts", headers=h)
    assert r.status_code == 200
    assert len(r.json()) == 1

    r = client.patch(f"/api/v1/alerts/{aid}", headers=h, json={"active": False})
    assert r.status_code == 200
    assert r.json()["active"] is False

    r = client.patch(f"/api/v1/alerts/{aid}", headers=h, json={"active": True, "threshold": 1600})
    assert r.status_code == 200
    body = r.json()
    assert body["active"] is True
    assert body["threshold"] == 1600.0
    assert body["triggered_at"] is None

    r = client.delete(f"/api/v1/alerts/{aid}", headers=h)
    assert r.status_code == 204

    r = client.get("/api/v1/alerts", headers=h)
    assert r.json() == []


def test_alerts_validates_kind(client, jwt_secret, fake_supabase):
    h = _auth(jwt_secret)
    r = client.post(
        "/api/v1/alerts",
        headers=h,
        json={"ticker": "TCS", "kind": "totally_made_up", "threshold": 1.0},
    )
    assert r.status_code == 422


def test_alerts_isolated_per_user(client, jwt_secret, fake_supabase):
    a = _auth(jwt_secret, sub="00000000-0000-0000-0000-00000000000a")
    b = _auth(jwt_secret, sub="00000000-0000-0000-0000-00000000000b")

    r = client.post(
        "/api/v1/alerts",
        headers=a,
        json={"ticker": "INFY", "kind": "price_below", "threshold": 1000.0},
    )
    aid = r.json()["id"]

    r = client.get("/api/v1/alerts", headers=b)
    assert r.json() == []

    r = client.delete(f"/api/v1/alerts/{aid}", headers=b)
    assert r.status_code == 404
