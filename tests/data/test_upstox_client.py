"""
Tests for `data.upstox_client` — keyless defaults, instrument-map resolution,
and graceful fallback on token / network failure.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _reset_instrument_cache(monkeypatch):
    """Isolate each test from the module-level instrument_map cache."""
    from data import upstox_client as uc
    monkeypatch.setattr(uc, "_instrument_map", None, raising=False)
    yield
    monkeypatch.setattr(uc, "_instrument_map", None, raising=False)


def test_is_configured_false_when_token_missing(monkeypatch):
    from data import upstox_client as uc
    monkeypatch.setattr(uc, "_get_secret", lambda k, d="": "", raising=True)
    assert uc.is_configured() is False


def test_is_configured_true_when_token_present(monkeypatch):
    from data import upstox_client as uc
    monkeypatch.setattr(uc, "_get_secret", lambda k, d="": "tok_abc" if k == "UPSTOX_ACCESS_TOKEN" else "")
    assert uc.is_configured() is True


def test_get_ltp_returns_none_when_token_missing(monkeypatch):
    from data import upstox_client as uc
    monkeypatch.setattr(uc, "_get_secret", lambda k, d="": "")
    assert uc.get_ltp("RELIANCE.NS") is None


def test_get_ltp_returns_none_when_instrument_unmapped(monkeypatch, tmp_path):
    from data import upstox_client as uc

    empty_map = tmp_path / "upstox_instruments.json"
    empty_map.write_text("{}", encoding="utf-8")

    def secret(key, default=""):
        if key == "UPSTOX_ACCESS_TOKEN":
            return "tok_abc"
        if key == "UPSTOX_INSTRUMENTS_JSON":
            return str(empty_map)
        return default

    monkeypatch.setattr(uc, "_get_secret", secret)
    assert uc.get_ltp("RELIANCE.NS") is None


def test_resolve_instrument_key_reads_from_map(monkeypatch, tmp_path):
    from data import upstox_client as uc

    map_path = tmp_path / "instruments.json"
    map_path.write_text(
        json.dumps({"RELIANCE.NS": "NSE_EQ|INE002A01018"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        uc,
        "_get_secret",
        lambda k, d="": str(map_path) if k == "UPSTOX_INSTRUMENTS_JSON" else "",
    )
    assert uc.resolve_instrument_key("RELIANCE.NS") == "NSE_EQ|INE002A01018"
    assert uc.resolve_instrument_key("reliance.ns") == "NSE_EQ|INE002A01018"  # case-insensitive
    assert uc.resolve_instrument_key("UNMAPPED.NS") is None
    assert uc.resolve_instrument_key("") is None


def test_get_ltp_swallows_network_errors(monkeypatch, tmp_path):
    from data import upstox_client as uc

    map_path = tmp_path / "instruments.json"
    map_path.write_text(
        json.dumps({"RELIANCE.NS": "NSE_EQ|INE002A01018"}), encoding="utf-8"
    )

    def secret(k, d=""):
        if k == "UPSTOX_ACCESS_TOKEN":
            return "tok_abc"
        if k == "UPSTOX_INSTRUMENTS_JSON":
            return str(map_path)
        return d

    monkeypatch.setattr(uc, "_get_secret", secret)

    class BadSession:
        def get(self, *a, **k):
            raise RuntimeError("dns blew up")

    monkeypatch.setattr(uc, "_session", lambda: BadSession())
    assert uc.get_ltp("RELIANCE.NS") is None


def test_get_ltp_parses_upstream_response(monkeypatch, tmp_path):
    from data import upstox_client as uc

    map_path = tmp_path / "instruments.json"
    map_path.write_text(
        json.dumps({"RELIANCE.NS": "NSE_EQ|INE002A01018"}), encoding="utf-8"
    )

    def secret(k, d=""):
        if k == "UPSTOX_ACCESS_TOKEN":
            return "tok_abc"
        if k == "UPSTOX_INSTRUMENTS_JSON":
            return str(map_path)
        return d

    monkeypatch.setattr(uc, "_get_secret", secret)

    class Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            # Upstox returns the key with a ":" separator in the response body.
            return {"data": {"NSE_EQ:INE002A01018": {"last_price": 2841.5}}}

    class OKSession:
        def get(self, *a, **k):
            return Resp()

    monkeypatch.setattr(uc, "_session", lambda: OKSession())
    assert uc.get_ltp("RELIANCE.NS") == pytest.approx(2841.5)


def test_default_fill_source_falls_back_to_yfinance(monkeypatch):
    """PaperTradeService default path: Upstox None → yfinance price."""
    from services import paper_trade_service as pts

    monkeypatch.setattr("data.upstox_client.get_ltp", lambda t: None)
    monkeypatch.setattr(pts, "_yf_fill", lambda t: 2500.0)
    assert pts.default_fill_source("RELIANCE.NS") == 2500.0


def test_default_fill_source_prefers_upstox_when_available(monkeypatch):
    from services import paper_trade_service as pts

    monkeypatch.setattr("data.upstox_client.get_ltp", lambda t: 2840.0)
    # yf_fill should not even be consulted — assert that via a sentinel.
    called = {"yf": 0}

    def _should_not_run(t):
        called["yf"] += 1
        return 9999.0

    monkeypatch.setattr(pts, "_yf_fill", _should_not_run)
    assert pts.default_fill_source("RELIANCE.NS") == 2840.0
    assert called["yf"] == 0
