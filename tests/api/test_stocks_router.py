"""Stocks router — search uses local CSV (no network), OHLCV/fundamentals are network-gated."""

from __future__ import annotations

from datetime import date, timedelta

import pytest


def test_search_returns_results_from_local_csv(client):
    """The CSV ships with the repo, so this test is offline-safe."""
    r = client.get("/api/v1/stocks", params={"q": "RELIANCE", "limit": 10})
    assert r.status_code == 200
    body = r.json()
    assert "count" in body
    assert "results" in body
    assert isinstance(body["results"], list)


def test_search_validates_limit_range(client):
    r = client.get("/api/v1/stocks", params={"limit": 0})
    assert r.status_code == 422


def test_ohlcv_rejects_inverted_window(client):
    r = client.get(
        "/api/v1/stocks/RELIANCE.NS/ohlcv",
        params={"start": "2024-01-10", "end": "2024-01-01"},
    )
    assert r.status_code == 422


@pytest.mark.network
@pytest.mark.slow
def test_ohlcv_returns_typed_history(client):
    end = date.today()
    start = end - timedelta(days=14)
    r = client.get(
        "/api/v1/stocks/RELIANCE.NS/ohlcv",
        params={"start": start.isoformat(), "end": end.isoformat()},
    )
    if r.status_code == 502:
        pytest.skip("yfinance unreachable")
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "RELIANCE.NS"
    assert "bars" in body
