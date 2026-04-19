"""B6 — model registry endpoint. Uses the bootstrap `models-registry/active.json`
that ships with the repo so no network / S3 access is required."""

from __future__ import annotations


def test_active_model_returns_version(client):
    r = client.get("/api/v1/models/active")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "hybrid"
    assert body["version"]
    assert body["backend"] in {"local", "s3"}
    assert "artifacts" in body and isinstance(body["artifacts"], list)


def test_versions_endpoint_returns_registry_listing(client):
    r = client.get("/api/v1/models/versions")
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] in {"local", "s3"}
    assert isinstance(body["versions"], list)
