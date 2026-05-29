"""
Transaction cost model with square-root market impact (Phase 3 — P3-7, P3-8).

Extends `models.nse_costs` (which already models brokerage + STT + slippage)
with a **square-root market impact** component and an **alpha gate** that
rejects signals whose expected alpha is less than 2× estimated round-trip cost.

Theory
------
The square-root market impact model (Almgren & Chriss 2001, Grinold & Kahn 1999)
is the standard institutional cost model for equity execution:

    impact(side) = k × σ_daily × sqrt(order_size / ADV)

where:
    k      ≈ 0.10–0.50 (market impact coefficient; 0.20 for liquid NSE F&O names)
    σ      = realised daily volatility of the stock
    ADV    = average daily volume (in rupees)
    order_size = notional value of the order

Round-trip impact (buy + sell) = 2 × one-side impact.

Total round-trip cost fraction:
    total = nse_base_cost + 2 × k × σ × sqrt(size_fraction)
where:
    size_fraction = order_notional / ADV_notional

Alpha gate (P3-8):
    Only emit a trade signal when:
        alpha_estimate > _ALPHA_COST_MULTIPLIER × total_cost
    Default multiplier = 2.0 (need at least 2× cost coverage).

Usage
-----
    from models.transaction_costs import TransactionCostModel, AlphaGate

    # Cost estimation
    model = TransactionCostModel()
    cost = model.estimate(
        ticker      = "RELIANCE.NS",
        sigma_daily = 0.015,          # 1.5% daily vol
        order_notional = 500_000,     # ₹5 lakh order
        adv_notional   = 2_000_000,   # ₹20 lakh ADV (illustrative)
    )
    print(f"Round-trip cost: {cost.total_bps:.1f} bps")

    # Alpha gate
    gate = AlphaGate()
    if gate.should_trade(alpha_signal=0.02, cost=cost):
        execute_trade()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from models.nse_costs import DELIVERY, NSECosts, round_trip_cost_fraction

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# Market impact coefficient for liquid NSE large-cap names.
# 0.20 corresponds to about 10 bps impact on a 1% ADV order with 1.5% daily vol.
# More liquid names (e.g. HDFC Bank, Reliance) can use 0.10–0.15.
_DEFAULT_IMPACT_K      = 0.20

# Impact is bounded: no name should have >100 bps one-way impact in our model.
_MAX_ONE_SIDE_IMPACT   = 0.0100   # 100 bps

# Alpha gate: emit trade only if alpha > multiplier × total_cost.
_ALPHA_COST_MULTIPLIER = 2.0

# Default fallback daily vol if not provided (NSE large-cap typical)
_DEFAULT_SIGMA_DAILY   = 0.015    # 1.5%

# Default fallback participation fraction if ADV not known
_DEFAULT_SIZE_FRACTION = 0.005    # 0.5% of ADV


# ── Cost result ───────────────────────────────────────────────────────────────

@dataclass
class CostEstimate:
    """Full breakdown of estimated round-trip transaction costs."""

    ticker: str
    order_notional: float

    # Components (all as fractions, not bps)
    base_fraction:   float = 0.0    # brokerage + STT + exchange + stamp + slippage
    impact_fraction: float = 0.0    # sqrt-model one-way × 2
    total_fraction:  float = 0.0    # base + impact

    # Convenience in basis points
    @property
    def base_bps(self) -> float:
        return self.base_fraction * 10_000

    @property
    def impact_bps(self) -> float:
        return self.impact_fraction * 10_000

    @property
    def total_bps(self) -> float:
        return self.total_fraction * 10_000

    @property
    def total_rupees(self) -> float:
        return self.total_fraction * self.order_notional

    def __repr__(self) -> str:
        return (
            f"CostEstimate({self.ticker}, "
            f"notional=₹{self.order_notional:,.0f}, "
            f"base={self.base_bps:.1f}bps, "
            f"impact={self.impact_bps:.1f}bps, "
            f"total={self.total_bps:.1f}bps)"
        )


# ── Core model ────────────────────────────────────────────────────────────────

class TransactionCostModel:
    """
    Round-trip transaction cost estimator combining:
      1. Statutory costs (NSECosts: brokerage, STT, exchange fees, stamp, slippage)
      2. Square-root market impact (Almgren-Chriss one-way model)

    The model is ticker-agnostic. Callers supply sigma_daily and ADV from
    their data pipeline; the model does the arithmetic.

    Parameters
    ----------
    impact_k : float
        Market impact coefficient. Default 0.20 is conservative for NSE
        large-caps. Set lower (0.10) for the most liquid names (HDFC Bank,
        Reliance) or higher (0.40–0.50) for mid-cap / less liquid names.
    nse_costs : NSECosts
        Statutory cost schedule. Defaults to DELIVERY (Zerodha rates).
    alpha_cost_multiplier : float
        The alpha gate factor. A signal is emitted only if:
            |alpha| > alpha_cost_multiplier × total_cost_fraction.
        Default 2.0 (require 2:1 alpha/cost ratio).
    """

    def __init__(
        self,
        impact_k: float               = _DEFAULT_IMPACT_K,
        nse_costs: NSECosts           = DELIVERY,
        alpha_cost_multiplier: float  = _ALPHA_COST_MULTIPLIER,
    ) -> None:
        self.impact_k              = float(impact_k)
        self.nse_costs             = nse_costs
        self.alpha_cost_multiplier = float(alpha_cost_multiplier)

    def estimate(
        self,
        ticker: str,
        order_notional: float,
        sigma_daily: float           = _DEFAULT_SIGMA_DAILY,
        adv_notional: Optional[float]= None,
        size_fraction: Optional[float]= None,
    ) -> CostEstimate:
        """
        Estimate round-trip transaction costs for a single trade.

        Parameters
        ----------
        ticker : str
            NSE ticker (used only for logging).
        order_notional : float
            Trade size in rupees (one side).
        sigma_daily : float
            Realised daily volatility (e.g. 0.015 for 1.5%). If 0, defaults
            to _DEFAULT_SIGMA_DAILY.
        adv_notional : float | None
            Average Daily Volume in rupees. When provided, `size_fraction`
            is computed as order_notional / adv_notional. If None, falls
            back to `size_fraction` or the default 0.5%.
        size_fraction : float | None
            Override: explicit participation rate (order / ADV). Used when
            ADV is not known in rupee terms but the fraction is known.

        Returns
        -------
        CostEstimate
        """
        order_notional = max(1.0, float(order_notional))

        # ── Statutory base cost (from nse_costs) ─────────────────────────────
        base_frac = round_trip_cost_fraction(order_notional, self.nse_costs)

        # ── Square-root market impact ─────────────────────────────────────────
        if sigma_daily <= 0.0:
            sigma_daily = _DEFAULT_SIGMA_DAILY

        if size_fraction is not None:
            sf = float(np.clip(size_fraction, 0.0, 1.0))
        elif adv_notional is not None and adv_notional > 0.0:
            sf = float(np.clip(order_notional / adv_notional, 0.0, 1.0))
        else:
            sf = _DEFAULT_SIZE_FRACTION

        # One-side impact = k × σ × √(participation_rate)
        one_side_impact = float(
            self.impact_k * sigma_daily * np.sqrt(sf)
        )
        one_side_impact = float(min(one_side_impact, _MAX_ONE_SIDE_IMPACT))

        # Round-trip = enter + exit
        impact_frac = 2.0 * one_side_impact

        total_frac = float(base_frac + impact_frac)

        return CostEstimate(
            ticker          = ticker,
            order_notional  = order_notional,
            base_fraction   = float(base_frac),
            impact_fraction = float(impact_frac),
            total_fraction  = float(total_frac),
        )

    def estimate_from_price(
        self,
        ticker: str,
        price: float,
        shares: int,
        sigma_daily: float            = _DEFAULT_SIGMA_DAILY,
        adv_shares: Optional[int]     = None,
    ) -> CostEstimate:
        """
        Convenience wrapper: compute notional from (price × shares) and
        ADV from (price × adv_shares).
        """
        notional = float(price * shares)
        adv_notional = float(price * adv_shares) if adv_shares else None
        return self.estimate(ticker, notional, sigma_daily, adv_notional)


# ── Alpha gate ────────────────────────────────────────────────────────────────

class AlphaGate:
    """
    Filters trading signals by requiring alpha > multiplier × cost.

    Design rationale:
        A signal with edge 5 bps on a stock with 15 bps round-trip cost has a
        negative expected return net of costs. The 2× multiplier (default) adds
        a safety margin for estimation error in both the alpha and cost models.

    Usage
    -----
        gate   = AlphaGate()
        cost   = cost_model.estimate("RELIANCE.NS", 500_000, sigma_daily=0.015)
        passed = gate.should_trade(alpha_signal=0.004, cost=cost)
        # alpha_signal in fraction: 0.004 = 40 bps expected gross alpha
    """

    def __init__(
        self,
        cost_model: Optional[TransactionCostModel] = None,
        multiplier: float = _ALPHA_COST_MULTIPLIER,
    ) -> None:
        self._cost_model = cost_model or TransactionCostModel()
        self.multiplier  = float(multiplier)

    def should_trade(
        self,
        alpha_signal: float,
        cost: Optional[CostEstimate] = None,
        *,
        ticker: str = "",
        order_notional: float = 100_000.0,
        sigma_daily: float    = _DEFAULT_SIGMA_DAILY,
        adv_notional: Optional[float] = None,
    ) -> bool:
        """
        Return True iff |alpha_signal| > multiplier × total_cost_fraction.

        You can either:
          (a) pass a pre-computed `CostEstimate` object, or
          (b) provide ticker/order_notional/sigma_daily and the gate will
              call the cost model internally.

        Parameters
        ----------
        alpha_signal : float
            Expected gross alpha as a fraction (e.g. 0.005 = 50 bps).
        cost : CostEstimate | None
            Pre-computed cost estimate. If None, computed from other args.
        """
        if cost is None:
            cost = self._cost_model.estimate(
                ticker         = ticker,
                order_notional = order_notional,
                sigma_daily    = sigma_daily,
                adv_notional   = adv_notional,
            )

        threshold  = self.multiplier * cost.total_fraction
        abs_alpha  = abs(float(alpha_signal))
        passes     = abs_alpha >= threshold

        if not passes:
            logger.debug(
                "AlphaGate: %s REJECTED — |alpha|=%.2fbps < threshold=%.2fbps "
                "(cost=%.2fbps × %.1f×)",
                cost.ticker or ticker,
                abs_alpha * 10_000,
                threshold * 10_000,
                cost.total_bps,
                self.multiplier,
            )

        return passes

    def filter_signals(
        self,
        signals: dict[str, float],
        cost_params: Optional[dict[str, dict]] = None,
    ) -> dict[str, float]:
        """
        Filter a dict of (ticker → alpha_signal) values using the alpha gate.

        `cost_params` is an optional dict mapping ticker → kwargs for
        `TransactionCostModel.estimate`. Defaults are used for absent tickers.

        Returns a filtered dict containing only signals that pass the gate.
        """
        cost_params = cost_params or {}
        passed: dict[str, float] = {}
        rejected_count = 0

        for ticker, alpha in signals.items():
            params   = cost_params.get(ticker, {})
            notional = float(params.get("order_notional", 100_000.0))
            sigma    = float(params.get("sigma_daily", _DEFAULT_SIGMA_DAILY))
            adv      = params.get("adv_notional")

            cost = self._cost_model.estimate(ticker, notional, sigma, adv)
            if self.should_trade(alpha, cost, ticker=ticker):
                passed[ticker] = alpha
            else:
                rejected_count += 1

        logger.info(
            "AlphaGate.filter_signals: %d/%d signals passed (%d rejected by cost gate)",
            len(passed), len(signals), rejected_count,
        )
        return passed


# ── Convenience: apply to return series ──────────────────────────────────────

def adjust_returns_for_costs(
    returns: "pd.Series",
    signals: "pd.Series",
    cost_model: Optional[TransactionCostModel] = None,
    ticker: str = "",
    order_notional: float = 100_000.0,
    sigma_daily: float    = _DEFAULT_SIGMA_DAILY,
    adv_notional: Optional[float] = None,
) -> "pd.Series":
    """
    Subtract the round-trip cost on every bar where `signals` changes to a
    new non-zero value (trade entry or direction flip).

    Combines the statutory cost (brokerage etc.) with the market impact
    model — more realistic than the original flat 0.1% in the backtester.

    Parameters
    ----------
    returns : pd.Series
        Strategy gross returns (one per bar, float).
    signals : pd.Series
        Position direction aligned to `returns`: +1 = long, −1 = short, 0 = flat.
    """
    import pandas as pd

    model = cost_model or TransactionCostModel()
    cost  = model.estimate(ticker, order_notional, sigma_daily, adv_notional)

    prev     = signals.shift(1).fillna(0)
    entries  = ((signals != prev) & (signals != 0)).astype(float)
    net_rets = returns - entries * cost.total_fraction
    return net_rets.astype(float)
