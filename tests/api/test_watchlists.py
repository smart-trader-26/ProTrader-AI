"""
Watchlists CRUD — auth required, isolated per user.

Uses the in-memory `FakeSupabase` (see `supabase_fake.py`) so the tests
stay offline. RLS is simulated: a client bound to user A can't read or
delete rows owned by user B, matching the behaviour of the real
`auth.uid() = user_id` policies.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest

from tests.api.supabase_fake import install as install_fake_supabase


@pytest.fixture
def jwt_secret(monkeypatch):
    secret = "test-secret-watchlists-32chars-min-len"
    monkeypatch.setattr("config.settings.SUPABASE_JWT_SECRET", secret)
    monkeypatch.setattr("api.auth.SUPABASE_JWT_SECRET", secret)
    monkeypatch.setattr("api.rate_limit.SUPABASE_JWT_SECRET", secret)
    return secret


@pytest.fixture
def fake_supabase(monkeypatch):
    return install_fake_supabase(monkeypatch)


def _token(secret, sub="00000000-0000-0000-0000-000000000abc", email="u@example.com"):
    payload = {
        "sub": sub,
        "email": email,
        "aud": "authenticated",
        "exp": int((datetime.now(UTC) + timedelta(minutes=5)).timestamp()),
        "role": "authenticated",
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def _auth(secret, **kw):
    return {"Authorization": f"Bearer {_token(secret, **kw)}"}


def test_watchlists_crud_round_trip(client, jwt_secret, fake_supabase):
    h = _auth(jwt_secret)

    r = client.get("/api/v1/watchlists", headers=h)
    assert r.status_code == 200
    assert r.json() == []

    r = client.post("/api/v1/watchlists", headers=h, json={"name": "NIFTY50"})
    assert r.status_code == 201
    wl = r.json()
    wl_id = wl["id"]
    assert wl["name"] == "NIFTY50"

    r = client.post(f"/api/v1/watchlists/{wl_id}/tickers", headers=h, json={"ticker": "RELIANCE"})
    assert r.status_code == 200
    assert {t["ticker"] for t in r.json()["tickers"]} == {"RELIANCE"}

    r = client.post(f"/api/v1/watchlists/{wl_id}/tickers", headers=h, json={"ticker": "tcs"})
    assert r.status_code == 200
    assert {t["ticker"] for t in r.json()["tickers"]} == {"RELIANCE", "TCS"}

    r = client.delete(f"/api/v1/watchlists/{wl_id}/tickers/RELIANCE", headers=h)
    assert r.status_code == 204

    r = client.get("/api/v1/watchlists", headers=h)
    assert {t["ticker"] for t in r.json()[0]["tickers"]} == {"TCS"}

    r = client.delete(f"/api/v1/watchlists/{wl_id}", headers=h)
    assert r.status_code == 204

    r = client.get("/api/v1/watchlists", headers=h)
    assert r.json() == []


def test_watchlists_isolated_per_user(client, jwt_secret, fake_supabase):
    a = _auth(jwt_secret, sub="00000000-0000-0000-0000-00000000000a", email="a@x")
    b = _auth(jwt_secret, sub="00000000-0000-0000-0000-00000000000b", email="b@x")

    r = client.post("/api/v1/watchlists", headers=a, json={"name": "alpha"})
    assert r.status_code == 201
    a_wl = r.json()["id"]

    r = client.get("/api/v1/watchlists", headers=b)
    assert r.json() == []

    r = client.delete(f"/api/v1/watchlists/{a_wl}", headers=b)
    assert r.status_code == 404


def test_watchlists_401_without_token(client):
    r = client.get("/api/v1/watchlists")
    assert r.status_code in (401, 503)
