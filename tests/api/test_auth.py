"""
Auth dep — JWT verification, dev-mode 401, /me round-trip.

We don't need Supabase running: we set `SUPABASE_JWT_SECRET` for the
duration of the test, mint our own HS256 token with the same secret, and
hand it to the TestClient. That exercises the real `jwt.decode` path.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest


@pytest.fixture
def jwt_secret(monkeypatch):
    secret = "test-secret-for-pytest-only-32chars-min-len"
    monkeypatch.setattr("config.settings.SUPABASE_JWT_SECRET", secret)
    monkeypatch.setattr("api.auth.SUPABASE_JWT_SECRET", secret)
    monkeypatch.setattr("api.rate_limit.SUPABASE_JWT_SECRET", secret)
    return secret


def _token(secret: str, *, sub="u-1", email="x@y.z", ttl_seconds=300, audience="authenticated"):
    payload = {
        "sub": sub,
        "email": email,
        "aud": audience,
        "exp": int((datetime.now(UTC) + timedelta(seconds=ttl_seconds)).timestamp()),
        "role": "authenticated",
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def test_me_401_without_token(client):
    r = client.get("/api/v1/me")
    assert r.status_code in (401, 503)


def test_me_401_with_bad_token(client, jwt_secret):
    r = client.get("/api/v1/me", headers={"Authorization": "Bearer not.a.jwt"})
    assert r.status_code == 401


def test_me_401_with_expired_token(client, jwt_secret):
    expired = _token(jwt_secret, ttl_seconds=-30)
    r = client.get("/api/v1/me", headers={"Authorization": f"Bearer {expired}"})
    assert r.status_code == 401
    assert "expired" in r.json()["detail"].lower()


def test_me_returns_claims_when_db_unconfigured(client, jwt_secret, monkeypatch):
    """Dev mode: no Supabase configured → /me echoes the JWT claims."""
    from db import supabase_client as sb

    monkeypatch.setattr(sb, "is_configured", lambda: False)
    tok = _token(jwt_secret, sub="00000000-0000-0000-0000-000000000001", email="dev@local")
    r = client.get("/api/v1/me", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "00000000-0000-0000-0000-000000000001"
    assert body["email"] == "dev@local"
