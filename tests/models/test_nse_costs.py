"""
Tests for A8.1 NSE cost model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.nse_costs import (
    DELIVERY,
    INTRADAY,
    NSECosts,
    apply_costs_to_returns,
    cost_breakdown,
    round_trip_cost_fraction,
    round_trip_cost_rupees,
)


def test_delivery_cost_at_100k_is_reasonable():
    # Zerodha public schedule ≈ 12-16 bps round-trip at retail notional
    frac = round_trip_cost_fraction(notional=100_000.0, costs=DELIVERY)
    assert 0.001 <= frac <= 0.002
    rupees = round_trip_cost_rupees(100_000.0, DELIVERY)
    assert 100 <= rupees <= 200


def test_intraday_stt_is_higher_than_delivery():
    """Intraday STT is 0.1% (4× delivery's 0.025%) — total cost must reflect that."""
    delivery = round_trip_cost_fraction(100_000.0, DELIVERY)
    intraday = round_trip_cost_fraction(100_000.0, INTRADAY)
    assert intraday > delivery


def test_cost_breakdown_lines_sum_to_total():
    bd = cost_breakdown(100_000.0, DELIVERY)
    # Sum must match the 'total' line (within float noise)
    parts = bd["brokerage"] + bd["stt"] + bd["exchange_txn"] + bd["sebi"] + bd["stamp_duty"] + bd["gst"] + bd["slippage"]
    assert parts == pytest.approx(bd["total"], abs=0.01)


def test_brokerage_flat_cap_kicks_in_for_large_trades():
    """Larger notional → per-trade cost drops *as fraction* because brokerage is capped at ₹20."""
    small = round_trip_cost_fraction(10_000.0, DELIVERY)
    large = round_trip_cost_fraction(1_000_000.0, DELIVERY)
    assert large < small


def test_apply_costs_only_charges_on_signal_flips():
    # signal: 0,0,1,1,1,0,0,-1,-1 → two entries at index 2 and 7
    signals = pd.Series([0, 0, 1, 1, 1, 0, 0, -1, -1])
    returns = pd.Series([0.0] * 9)

    adj = apply_costs_to_returns(returns, signals, notional=100_000, costs=DELIVERY)
    # Exactly two bars should be non-zero (the two entries)
    assert (adj != 0).sum() == 2
    # Both should be negative (costs reduce returns)
    assert (adj[adj != 0] < 0).all()


def test_custom_cost_overrides_change_total():
    custom = NSECosts(brokerage_pct=0.002, brokerage_flat_cap=1_000_000)  # no cap
    default_frac = round_trip_cost_fraction(100_000, DELIVERY)
    custom_frac = round_trip_cost_fraction(100_000, custom)
    assert custom_frac > default_frac  # much higher brokerage
