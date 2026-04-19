"""B6.3 — model registry endpoint."""

from __future__ import annotations


def test_active_model_returns_version(client):
    r = client.get("/api/v1/models/active")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "hybrid"
    assert body["version"]  # non-empty
    assert body["source"] == "in-process"
