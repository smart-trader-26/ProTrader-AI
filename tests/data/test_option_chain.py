"""
Unit tests for A4 option-chain feature extraction.

Feature math is tested against a handcrafted chain; the NSE network path is
only smoke-tested behind the `network` marker because the live API gates
requests by cookie + geo and flakes in CI.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from data.option_chain import (
    OptionChainSnapshot,
    _normalize_symbol,
    extract_option_features,
    fetch_option_chain,
)


def test_normalize_symbol_strips_ns_suffix():
    assert _normalize_symbol("RELIANCE.NS") == "RELIANCE"
    assert _normalize_symbol("tcs.ns") == "TCS"
    assert _normalize_symbol("INFY") == "INFY"
    assert _normalize_symbol("  HDFC.BO ") == "HDFC"


def test_extract_features_returns_zeros_when_snap_none():
    feats = extract_option_features(None)
    # Sentinel values — neutral PCR=1, everything else zero
    assert feats["put_call_ratio"] == 1.0
    assert feats["put_call_chg_ratio"] == 1.0
    assert feats["max_pain_distance"] == 0.0
    assert feats["atm_iv"] == 0.0
    assert feats["iv_skew"] == 0.0
    assert feats["weighted_iv"] == 0.0


def test_extract_features_pcr_and_iv_skew():
    # Spot = 100, five strikes: 90, 95, 100, 105, 110
    # PE-heavy book (bearish positioning) → PCR > 1
    # PE IV > CE IV at ATM → positive skew
    chain = pd.DataFrame(
        {
            "strike":     [90.0, 95.0, 100.0, 105.0, 110.0],
            "CE_OI":      [100,  200,  500,   400,   300],
            "CE_Chng_OI": [10,   20,   50,    40,    30],
            "CE_IV":      [25,   23,   20,    22,    25],
            "CE_LTP":     [12,    8,    4,     2,     1],
            "PE_OI":      [400,  600,  800,   300,   200],
            "PE_Chng_OI": [40,   60,   80,    30,    20],
            "PE_IV":      [30,   27,   24,    23,    25],
            "PE_LTP":     [1,     2,    4,     8,    12],
        }
    )
    snap = OptionChainSnapshot(
        symbol="TEST", underlying_price=100.0,
        expiry="25-Apr-2026", fetched_at=date(2026, 4, 10), chain=chain,
    )

    feats = extract_option_features(snap)

    # PCR = sum(PE_OI)/sum(CE_OI) = 2300 / 1500
    assert feats["put_call_ratio"] == pytest.approx(2300 / 1500, abs=1e-6)
    assert feats["put_call_chg_ratio"] == pytest.approx(230 / 150, abs=1e-6)

    # ATM = 100 → PE_IV(24) - CE_IV(20) = 4 (put skew)
    assert feats["iv_skew"] == pytest.approx(4.0, abs=1e-6)
    assert feats["atm_iv"] == pytest.approx((24 + 20) / 2, abs=1e-6)

    # Call wall = strike with max CE_OI = 100; distance = 0
    assert feats["call_wall_distance"] == pytest.approx(0.0, abs=1e-6)
    # Put wall = strike with max PE_OI = 100; distance = 0
    assert feats["put_wall_distance"] == pytest.approx(0.0, abs=1e-6)

    # OI concentration: top-3 CE = 500+400+300 = 1200, total 1500 → 0.8
    assert feats["oi_call_concentration"] == pytest.approx(0.8, abs=1e-6)
    # Top-3 PE = 800+600+400 = 1800, total 2300
    assert feats["oi_put_concentration"] == pytest.approx(1800 / 2300, abs=1e-6)


def test_extract_features_weighted_iv_is_oi_weighted():
    chain = pd.DataFrame(
        {
            "strike":     [100.0, 110.0],
            "CE_OI":      [100, 0],
            "CE_Chng_OI": [0, 0],
            "CE_IV":      [20, 30],
            "CE_LTP":     [5, 1],
            "PE_OI":      [0, 100],
            "PE_Chng_OI": [0, 0],
            "PE_IV":      [25, 35],
            "PE_LTP":     [1, 5],
        }
    )
    snap = OptionChainSnapshot(
        symbol="T", underlying_price=105.0, expiry="x",
        fetched_at=date(2026, 4, 10), chain=chain,
    )
    feats = extract_option_features(snap)
    # Weighted IV = (100*20 + 100*35) / 200 = (2000 + 3500)/200 = 27.5
    assert feats["weighted_iv"] == pytest.approx(27.5, abs=1e-6)


@pytest.mark.network
@pytest.mark.slow
def test_fetch_option_chain_smoke():
    """Real NSE call — tolerant of rate limits / geo blocks (returns None)."""
    snap = fetch_option_chain("RELIANCE")
    if snap is None:
        pytest.skip("NSE rate-limited or unreachable")
    assert snap.underlying_price > 0
    assert not snap.chain.empty
    assert {"strike", "CE_OI", "PE_OI"}.issubset(snap.chain.columns)
