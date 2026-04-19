"""B5 — tick publisher CLI / poll loop. Offline by stubbing yfinance +
publish()."""

from __future__ import annotations

from workers import tick_publisher


def test_resolve_symbols_cli_wins(monkeypatch):
    monkeypatch.setenv("PROTRADER_TICK_SYMBOLS", "IGNORE.ME")
    assert tick_publisher._resolve_symbols("reliance.ns, tcs.ns") == ["RELIANCE.NS", "TCS.NS"]


def test_resolve_symbols_env_fallback(monkeypatch):
    monkeypatch.setenv("PROTRADER_TICK_SYMBOLS", "infy.ns,wipro.ns")
    assert tick_publisher._resolve_symbols(None) == ["INFY.NS", "WIPRO.NS"]


def test_resolve_symbols_defaults(monkeypatch):
    monkeypatch.delenv("PROTRADER_TICK_SYMBOLS", raising=False)

    # Force sb.has_service_role() → False so we hit the DataConfig fallback.
    from db import supabase_client as sb

    monkeypatch.setattr(sb, "has_service_role", lambda: False)
    syms = tick_publisher._resolve_symbols(None)
    assert len(syms) > 0
    assert all(isinstance(s, str) for s in syms)


def test_poll_once_skips_none_prices(monkeypatch):
    published: list[tuple[str, float]] = []

    def fake_last_close(sym: str):
        return 100.0 if sym == "A" else None

    def fake_publish(tick):
        published.append((tick.ticker, tick.price))
        return 1

    monkeypatch.setattr(tick_publisher, "_last_close", fake_last_close)
    monkeypatch.setattr(tick_publisher, "publish", fake_publish)

    n = tick_publisher._poll_once(["A", "B", "C"])
    assert n == 1
    assert published == [("A", 100.0)]
