"""
Sentiment router — score endpoint validates input shape; ticker endpoint
is network-gated since it hits the news fetcher + FinBERT.
"""

from __future__ import annotations

import pytest


def test_score_rejects_empty_text(client):
    r = client.post("/api/v1/sentiment/score", json={"text": ""})
    assert r.status_code == 422


def test_v2_endpoint_503s_without_token(client, monkeypatch):
    """When HF_TOKEN is unset, v2 must 503 (not 500) so the FE can fall back."""
    from api.routers import sentiment as sent_router

    monkeypatch.setattr(sent_router, "HF_TOKEN", "")
    r = client.get("/api/v1/stocks/RELIANCE.NS/sentiment/v2")
    assert r.status_code == 503


@pytest.mark.network
@pytest.mark.slow
def test_score_text_returns_label(client):
    r = client.post(
        "/api/v1/sentiment/score",
        json={"text": "Reliance beats earnings estimates"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ("positive", "negative", "neutral")
    assert 0.0 <= body["confidence"] <= 1.0
