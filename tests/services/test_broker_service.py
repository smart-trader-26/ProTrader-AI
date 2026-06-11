"""Tests for the P5 broker adapters (no kiteconnect, no network)."""

from __future__ import annotations

from services.broker_service import DryRunBroker, KiteBroker, get_broker


def test_dry_run_broker_simulates_order():
    res = DryRunBroker().place_buy_with_protection(
        "TCS.NS", qty=10, last_price=2172.5, stop_price=1955.25, target_price=None
    )
    assert res.dry_run is True
    assert res.broker == "dry_run"
    assert res.order_id.startswith("DRY-")
    assert res.gtt_id and res.gtt_id.startswith("DRY-GTT-")


def test_dry_run_no_gtt_without_stop():
    res = DryRunBroker().place_buy_with_protection(
        "TCS.NS", qty=10, last_price=2172.5, stop_price=None, target_price=None
    )
    assert res.gtt_id is None


def test_get_broker_defaults_to_dry_run():
    broker, live = get_broker()
    assert broker.name == "dry_run"
    assert live is False


def test_kite_unconfigured_without_keys():
    assert KiteBroker(api_key="", access_token="").is_configured() is False
    assert KiteBroker(api_key="k", access_token="t").is_configured() is True


def test_kite_symbol_mapping():
    assert KiteBroker._split_symbol("TCS.NS") == ("NSE", "TCS")
    assert KiteBroker._split_symbol("reliance.ns") == ("NSE", "RELIANCE")
    assert KiteBroker._split_symbol("SENSEXSTK.BO") == ("BSE", "SENSEXSTK")
    assert KiteBroker._split_symbol("INFY") == ("NSE", "INFY")
