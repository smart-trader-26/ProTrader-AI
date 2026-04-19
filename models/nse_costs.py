"""
NSE round-trip transaction cost model (A8.1).

The existing backtester used a flat 0.1% per trade. That's wrong for Indian
equities — real costs are a stack of fixed + proportional + taxed fees:

    Brokerage:            0.03% or ₹20 per side, whichever is lower  (discount broker)
    STT (sell side only): 0.025% on sell turnover (delivery) or 0.1% (intraday)
    Exchange txn charge:  0.00325% per side (NSE) or 0.00375% (BSE)
    SEBI turnover fee:    0.0001% per side
    Stamp duty (buy):     0.015% (delivery) or 0.003% (intraday)
    GST:                  18% on (brokerage + exchange txn + SEBI fee)
    Slippage:             5 bps assumed for liquid NSE names

This module exposes:
    round_trip_cost_fraction(...)        — returns fraction (e.g. 0.00123)
    round_trip_cost_rupees(...)          — returns absolute ₹ for a given notional
    apply_costs_to_returns(returns, ...) — subtract per-trade costs from a return series

Defaults match Zerodha's public schedule (the de-facto retail standard).
Users running with a different broker can override every rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NSECosts:
    """Per-side and per-turnover rates for a typical discount broker."""

    brokerage_pct: float = 0.0003          # 0.03% per side
    brokerage_flat_cap: float = 20.0       # ₹20 flat cap per side
    stt_sell_pct: float = 0.00025          # 0.025% on sell notional (delivery)
    exchange_txn_pct: float = 0.0000325    # NSE 0.00325 bps
    sebi_pct: float = 0.000001             # 0.0001 bps
    stamp_duty_buy_pct: float = 0.00015    # 0.015% on buy notional (delivery)
    gst_pct: float = 0.18                  # 18% on services (brokerage + exchange + sebi)
    slippage_bps: float = 5.0              # 5 bps slippage per trade

    @property
    def slippage_pct(self) -> float:
        return self.slippage_bps / 10000.0


INTRADAY = NSECosts(
    stt_sell_pct=0.001,          # 0.1% intraday STT on sell notional
    stamp_duty_buy_pct=0.00003,  # 0.003% intraday stamp duty
)

DELIVERY = NSECosts()  # defaults above


def round_trip_cost_fraction(
    notional: float = 100_000.0,
    costs: NSECosts = DELIVERY,
) -> float:
    """
    Return the round-trip (buy + sell) cost as a fraction of notional.

    Modelled for a single trade of `notional` rupees entering and exiting
    at the same price. The per-side brokerage flat-cap makes this cost
    *lower* for larger trade sizes, so the number depends on `notional`.

    Example at ₹100,000 notional delivery:
        brokerage  = min(0.0003 * 1e5, 20) × 2 = 40          (₹20 cap kicks in)
        stt_sell   = 0.00025 * 1e5             = 25
        exchange   = 2 × 0.0000325 * 1e5       = 6.5
        sebi       = 2 × 1e-6 * 1e5            = 0.2
        stamp      = 0.00015 * 1e5             = 15
        gst        = 0.18 × (40 + 6.5 + 0.2)   = 8.406
        slippage   = 0.0005 * 1e5              = 50
        total      = ~145 / 100000             = 0.00145 (≈ 14.5 bps)
    """
    notional = max(1.0, float(notional))

    brokerage_per_side = min(costs.brokerage_pct * notional, costs.brokerage_flat_cap)
    brokerage = 2 * brokerage_per_side

    stt = costs.stt_sell_pct * notional  # sell side only
    exchange = 2 * costs.exchange_txn_pct * notional
    sebi = 2 * costs.sebi_pct * notional
    stamp = costs.stamp_duty_buy_pct * notional  # buy side only
    gst = costs.gst_pct * (brokerage + exchange + sebi)
    slippage = costs.slippage_pct * notional * 2  # both sides slipped

    total = brokerage + stt + exchange + sebi + stamp + gst + slippage
    return float(total / notional)


def round_trip_cost_rupees(
    notional: float,
    costs: NSECosts = DELIVERY,
) -> float:
    return round_trip_cost_fraction(notional, costs) * notional


def apply_costs_to_returns(
    returns: pd.Series,
    signals: pd.Series,
    notional: float = 100_000.0,
    costs: NSECosts = DELIVERY,
) -> pd.Series:
    """
    Subtract the round-trip cost on every bar where `signals` flips to a
    new non-zero value (trade entry). Matches the semantics of the existing
    `VectorizedBacktester` but with a realistic cost rate.
    """
    rt = round_trip_cost_fraction(notional, costs)
    prev = signals.shift(1).fillna(0)
    entries = ((signals != prev) & (signals != 0)).astype(float)
    return (returns - entries * rt).astype(float)


def cost_breakdown(notional: float, costs: NSECosts = DELIVERY) -> dict[str, float]:
    """Line-item breakdown in rupees — useful for the UI cost-audit expander."""
    brokerage_per_side = min(costs.brokerage_pct * notional, costs.brokerage_flat_cap)
    brokerage = 2 * brokerage_per_side
    stt = costs.stt_sell_pct * notional
    exchange = 2 * costs.exchange_txn_pct * notional
    sebi = 2 * costs.sebi_pct * notional
    stamp = costs.stamp_duty_buy_pct * notional
    gst = costs.gst_pct * (brokerage + exchange + sebi)
    slippage = costs.slippage_pct * notional * 2
    return {
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 2),
        "exchange_txn": round(exchange, 2),
        "sebi": round(sebi, 2),
        "stamp_duty": round(stamp, 2),
        "gst": round(gst, 2),
        "slippage": round(slippage, 2),
        "total": round(brokerage + stt + exchange + sebi + stamp + gst + slippage, 2),
        "pct_of_notional": round(
            np.nan if notional == 0 else 100 * (brokerage + stt + exchange + sebi + stamp + gst + slippage) / notional,
            4,
        ),
    }
