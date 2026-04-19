"""B5 — Redis Pub/Sub tick broker. Offline: exercises the JSON contract and
availability gating. A real round-trip test would need fakeredis / a live
Redis, which the repo doesn't depend on; that check runs in CI against a
Redis service."""

from __future__ import annotations

import json

import pytest

from workers import tick_broker
from workers.tick_broker import Tick, channel_for


def test_channel_for_is_upper_prefixed():
    assert channel_for("reliance.ns") == "ticks:RELIANCE.NS"


def test_tick_round_trips_through_json():
    t = Tick(ticker="RELIANCE.NS", ts="2026-04-19T09:42:00+00:00", price=2954.6, source="yfinance")
    back = Tick.from_json(t.to_json())
    assert back == t
    # Payload shape — must stay stable for the frontend.
    assert set(json.loads(t.to_json())) == {"ticker", "ts", "price", "source"}


def test_tick_from_json_accepts_bytes():
    t = Tick(ticker="TCS.NS", ts="2026-04-19T09:42:00+00:00", price=4100.0)
    back = Tick.from_json(t.to_json().encode("utf-8"))
    assert back.price == pytest.approx(4100.0)
    assert back.source == "yfinance"  # default


def test_is_available_reflects_redis_url(monkeypatch):
    monkeypatch.setattr(tick_broker, "REDIS_URL", "")
    assert tick_broker.is_available() is False
    monkeypatch.setattr(tick_broker, "REDIS_URL", "redis://example:6379/0")
    assert tick_broker.is_available() is True


def test_publish_raises_without_redis_url(monkeypatch):
    """publish() must fail loudly rather than silently dropping ticks."""
    monkeypatch.setattr(tick_broker, "REDIS_URL", "")
    monkeypatch.setattr(tick_broker, "_SYNC_R", None)
    with pytest.raises(RuntimeError, match="REDIS_URL"):
        tick_broker.publish(Tick(ticker="X", ts="2026-04-19T00:00:00Z", price=1.0))
