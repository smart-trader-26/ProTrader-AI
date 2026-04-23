"""
Tests for scripts.populate_instruments — offline, no network.
"""

from __future__ import annotations

import json

import pytest

from scripts.populate_instruments import build_instrument_map, write_map


# ── Sample data matching Upstox BOD JSON format ──

SAMPLE_INSTRUMENTS = [
    {
        "segment": "NSE_EQ",
        "name": "RELIANCE INDUSTRIES",
        "exchange": "NSE",
        "isin": "INE002A01018",
        "instrument_type": "EQ",
        "instrument_key": "NSE_EQ|INE002A01018",
        "lot_size": 1,
        "exchange_token": "2885",
        "tick_size": 5.0,
        "trading_symbol": "RELIANCE",
        "short_name": "RELIANCE",
    },
    {
        "segment": "NSE_EQ",
        "name": "TATA CONSULTANCY SVCS",
        "exchange": "NSE",
        "isin": "INE467B01029",
        "instrument_type": "EQ",
        "instrument_key": "NSE_EQ|INE467B01029",
        "lot_size": 1,
        "exchange_token": "11536",
        "tick_size": 5.0,
        "trading_symbol": "TCS",
        "short_name": "TCS",
    },
    # Should be excluded: wrong segment
    {
        "segment": "NSE_FO",
        "name": "RELIANCE FUT",
        "exchange": "NSE",
        "instrument_type": "FUT",
        "instrument_key": "NSE_FO|36702",
        "lot_size": 250,
        "trading_symbol": "RELIANCE",
    },
    # Should be excluded: wrong instrument type
    {
        "segment": "NSE_EQ",
        "name": "JOCIL LIMITED",
        "exchange": "NSE",
        "isin": "INE839G01010",
        "instrument_type": "BE",  # not EQ
        "instrument_key": "NSE_EQ|INE839G01010",
        "lot_size": 1,
        "trading_symbol": "JOCIL",
    },
    # Should be excluded: missing trading_symbol
    {
        "segment": "NSE_EQ",
        "name": "BLANK",
        "exchange": "NSE",
        "instrument_type": "EQ",
        "instrument_key": "NSE_EQ|INE000X00000",
        "trading_symbol": "",
    },
]


def test_build_instrument_map_filters_eq_only():
    result = build_instrument_map(SAMPLE_INSTRUMENTS)
    assert "RELIANCE.NS" in result
    assert "TCS.NS" in result
    # FUT, BE, and blank should be excluded
    assert len(result) == 2


def test_build_instrument_map_correct_keys():
    result = build_instrument_map(SAMPLE_INSTRUMENTS)
    assert result["RELIANCE.NS"] == "NSE_EQ|INE002A01018"
    assert result["TCS.NS"] == "NSE_EQ|INE467B01029"


def test_build_instrument_map_empty_input():
    assert build_instrument_map([]) == {}


def test_write_map_creates_valid_json(tmp_path):
    mapping = {"RELIANCE.NS": "NSE_EQ|INE002A01018", "TCS.NS": "NSE_EQ|INE467B01029"}
    out = tmp_path / "instruments.json"
    write_map(mapping, out)

    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == mapping


def test_write_map_creates_parent_dirs(tmp_path):
    out = tmp_path / "sub" / "dir" / "instruments.json"
    write_map({"A.NS": "NSE_EQ|X"}, out)
    assert out.exists()
