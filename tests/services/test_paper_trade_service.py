"""
Tests for A9.2 broker-agnostic paper-trading service.

Uses an injected in-memory fill source — no yfinance dependency, per the
same pattern as the ledger tests.
"""

from __future__ import annotations

import pytest

from services.paper_trade_service import PaperTradeService


class FakeFills:
    """Mutable price tape — set `price` before each call."""

    def __init__(self, price: float = 100.0):
        self.price = price

    def __call__(self, _ticker: str) -> float:
        return self.price


def _svc(tmp_path, starting_cash: float = 1_000_000.0) -> tuple[PaperTradeService, FakeFills]:
    fills = FakeFills()
    svc = PaperTradeService(
        db_path=tmp_path / "book.sqlite",
        starting_cash=starting_cash,
        fill_source=fills,
    )
    return svc, fills


def test_opens_long_when_prob_above_threshold(tmp_path):
    svc, fills = _svc(tmp_path)
    fills.price = 100.0
    pos = svc.on_signal("TCS.NS", prob_up=0.70, threshold=0.55, qty=10)
    assert pos is not None
    assert pos.side == "long"
    assert pos.entry_price == 100.0
    assert pos.qty == 10


def test_does_not_open_when_prob_below_threshold(tmp_path):
    svc, fills = _svc(tmp_path)
    fills.price = 100.0
    pos = svc.on_signal("TCS.NS", prob_up=0.40, threshold=0.55)
    assert pos is None or pos.side == "flat"


def test_closes_on_target_hit(tmp_path):
    svc, fills = _svc(tmp_path)
    fills.price = 100.0
    svc.on_signal("TCS.NS", prob_up=0.70, threshold=0.55, qty=10,
                  stop_pct=0.02, target_pct=0.04)

    # price jumps above target (+4% → 104) — mark-to-market should close it
    fills.price = 105.0
    closed = svc.mark_to_market()
    assert len(closed) == 1
    fill = closed[0]
    assert fill.exit_price == 105.0
    assert fill.gross_pnl == pytest.approx((105.0 - 100.0) * 10, abs=1e-6)
    # Net < gross because costs are deducted
    assert fill.net_pnl < fill.gross_pnl


def test_closes_on_stop_hit(tmp_path):
    svc, fills = _svc(tmp_path)
    fills.price = 100.0
    svc.on_signal("TCS.NS", prob_up=0.70, threshold=0.55, qty=10,
                  stop_pct=0.02, target_pct=0.04)

    fills.price = 97.0  # below 2% stop
    closed = svc.mark_to_market()
    assert len(closed) == 1
    assert closed[0].reason_exit == "stop_hit"
    assert closed[0].gross_pnl < 0


def test_signal_flip_closes_position(tmp_path):
    svc, fills = _svc(tmp_path)
    fills.price = 100.0
    svc.on_signal("TCS.NS", prob_up=0.70, threshold=0.55, qty=10)

    # Prob drops below threshold → should close
    fills.price = 101.0
    svc.on_signal("TCS.NS", prob_up=0.30, threshold=0.55)

    state = svc.book_state()
    assert state.n_open == 0
    assert state.n_fills == 1


def test_book_state_tracks_equity(tmp_path):
    svc, fills = _svc(tmp_path, starting_cash=100_000.0)

    # No trades yet → cash = starting, equity = starting
    state = svc.book_state()
    assert state.cash == 100_000.0
    assert state.realised_pnl == 0.0
    assert state.equity == 100_000.0

    # Open a long, mark price higher → unrealised > 0
    fills.price = 100.0
    svc.on_signal("ABC.NS", prob_up=0.70, threshold=0.55, qty=5)
    fills.price = 102.0
    state = svc.book_state()
    assert state.unrealised_pnl == pytest.approx((102.0 - 100.0) * 5, abs=1e-6)
    assert state.n_open == 1


def test_recent_fills_filters_by_ticker(tmp_path):
    svc, fills = _svc(tmp_path)
    fills.price = 50.0
    svc.on_signal("AAA.NS", prob_up=0.70, threshold=0.55)
    svc.on_signal("BBB.NS", prob_up=0.70, threshold=0.55)
    fills.price = 52.0
    svc.mark_to_market()  # may or may not close (0.04 target from 50 = 52)

    aaa = svc.recent_fills(ticker="AAA.NS")
    assert all(f.ticker == "AAA.NS" for f in aaa)


def test_fill_cost_matches_nse_model(tmp_path):
    """Verify closed-fill costs are non-zero and use DELIVERY notional-scaled model."""
    svc, fills = _svc(tmp_path)
    fills.price = 1000.0  # high-price stock → large notional
    svc.on_signal("X.NS", prob_up=0.70, threshold=0.55, qty=100)

    fills.price = 1050.0  # target hit (+5%)
    closed = svc.mark_to_market()
    assert len(closed) == 1
    fill = closed[0]
    # notional = 100_000 → ~14-16 bps ≈ ₹140-160 in costs
    assert 100 < fill.costs < 300
