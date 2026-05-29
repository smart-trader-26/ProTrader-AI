"""
Portfolio constructor: fractional Kelly sizing with per-name and sector caps (Phase 3 — P3-6).

Theory
------
The Kelly criterion (Kelly 1956) provides the bet size that maximises long-run
log-wealth. For continuous returns with signal strength s and model IC:

    f* = IC × s / σ²_return

where σ²_return is the variance of the forecast return distribution.

In practice we use a *fractional* Kelly (0.25× of full Kelly) for two reasons:
  1. Kelly is derived under log-utility; most institutional investors have less
     aggressive utility, equivalent to quarter-Kelly or less.
  2. IC and σ are estimated with error. Over-sizing relative to a noisy estimate
     produces ruin risk that the Kelly formula does not account for.

Position caps:
  • Per-name cap:   5% of portfolio (prevents concentration in any single name)
  • Sector cap:    30% of portfolio across all names in the same sector
    (defined by TICKER_SECTOR in cross_sectional_trainer.py)

The portfolio is long-only (no short positions). After capping, weights are
renormalised to sum to 1.0.

Risk budget:
  Before sizing, gross notional is scaled so that the portfolio's expected daily
  volatility equals the *volatility target* (default 15% annualised = ~0.95% daily).
  This is the standard "risk parity / vol targeting" approach used by most
  systematic equity funds.

Usage
-----
    from models.portfolio_constructor import PortfolioConstructor

    constructor = PortfolioConstructor(
        kelly_fraction       = 0.25,
        per_name_cap         = 0.05,
        sector_cap           = 0.30,
        annual_vol_target    = 0.15,
    )

    # combined_signals: dict[ticker → composite_signal (−1..+1)]
    # sigma_dict:       dict[ticker → daily_vol (fraction)]
    # sector_map:       dict[ticker → sector_id (int)]
    weights = constructor.construct(
        combined_signals = {"RELIANCE.NS": 0.45, "HDFCBANK.NS": 0.30, ...},
        sigma_dict       = {"RELIANCE.NS": 0.015, "HDFCBANK.NS": 0.012, ...},
        sector_map       = {"RELIANCE.NS": 0, "HDFCBANK.NS": 1, ...},
    )
    # weights: dict[ticker → portfolio_weight (0..1, sum ≤ 1)]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
_DEFAULT_KELLY_FRACTION   = 0.25    # fractional Kelly
_DEFAULT_PER_NAME_CAP     = 0.05    # 5% max per stock
_DEFAULT_SECTOR_CAP       = 0.30    # 30% max per sector
_DEFAULT_ANNUAL_VOL_TGT   = 0.15    # 15% annualised vol target
_TRADING_DAYS             = 252     # trading days per year
_DAILY_VOL_TGT            = _DEFAULT_ANNUAL_VOL_TGT / np.sqrt(_TRADING_DAYS)

# Model IC: expected rank-correlation between signal and forward return.
# Used to convert signal magnitude to Kelly position size.
_DEFAULT_IC               = 0.030   # conservative prior

# Minimum signal magnitude to include a name in the portfolio
_MIN_SIGNAL_THRESHOLD     = 0.10    # signals below this are treated as zero

# Max number of names in the portfolio (avoid spreading too thin)
_MAX_PORTFOLIO_NAMES      = 20
_MIN_PORTFOLIO_NAMES      = 3       # don't construct if fewer than 3 names pass filter


# ── DTOs ─────────────────────────────────────────────────────────────────────

@dataclass
class PortfolioWeights:
    """Result of portfolio construction."""

    weights: dict[str, float]                  # ticker → weight (0..1, sum ≤ 1.0)
    cash_weight: float = 0.0                   # 1 − sum(weights): uninvested

    # Diagnostics
    n_names:          int   = 0
    pre_cap_weights:  dict[str, float] = field(default_factory=dict)
    sector_weights:   dict[int, float] = field(default_factory=dict)
    total_gross:      float = 0.0             # sum before normalisation
    applied_vol_scale: float = 1.0            # volatility scaling factor

    # Constraint breach summary
    per_name_cap_applied:  bool = False
    sector_cap_applied:    bool = False

    def __repr__(self) -> str:
        top5 = sorted(self.weights.items(), key=lambda x: -x[1])[:5]
        top5_str = ", ".join(f"{t}={w:.1%}" for t, w in top5)
        return (
            f"PortfolioWeights(n={self.n_names}, "
            f"invested={sum(self.weights.values()):.1%}, "
            f"top5=[{top5_str}])"
        )


# ── Core constructor ──────────────────────────────────────────────────────────

class PortfolioConstructor:
    """
    Converts alpha signal scores into portfolio weights using fractional
    Kelly sizing, per-name caps, sector caps, and volatility targeting.

    Parameters
    ----------
    kelly_fraction : float
        Fraction of full Kelly to use. Default 0.25 (quarter-Kelly).
    per_name_cap : float
        Maximum portfolio weight per individual name. Default 0.05 (5%).
    sector_cap : float
        Maximum total weight in any single GICS sector. Default 0.30 (30%).
    annual_vol_target : float
        Target annualised portfolio volatility. Default 0.15 (15%).
        The portfolio is scaled so its expected vol matches this target.
    ic : float
        Expected IC of the combined signal. Default 0.030.
    max_names : int
        Maximum number of names in the portfolio. Default 20.
    """

    def __init__(
        self,
        kelly_fraction:    float = _DEFAULT_KELLY_FRACTION,
        per_name_cap:      float = _DEFAULT_PER_NAME_CAP,
        sector_cap:        float = _DEFAULT_SECTOR_CAP,
        annual_vol_target: float = _DEFAULT_ANNUAL_VOL_TGT,
        ic:                float = _DEFAULT_IC,
        max_names:         int   = _MAX_PORTFOLIO_NAMES,
    ) -> None:
        self.kelly_fraction    = float(kelly_fraction)
        self.per_name_cap      = float(per_name_cap)
        self.sector_cap        = float(sector_cap)
        self.annual_vol_target = float(annual_vol_target)
        self.daily_vol_target  = self.annual_vol_target / np.sqrt(_TRADING_DAYS)
        self.ic                = float(ic)
        self.max_names         = int(max_names)

    def construct(
        self,
        combined_signals: dict[str, float],
        sigma_dict: Optional[dict[str, float]] = None,
        sector_map: Optional[dict[str, int]]   = None,
    ) -> PortfolioWeights:
        """
        Build portfolio weights from combined alpha signals.

        Parameters
        ----------
        combined_signals : dict[str, float]
            Maps NSE ticker → composite signal (−1..+1). Only positive signals
            produce long positions (long-only portfolio).
        sigma_dict : dict[str, float] | None
            Maps ticker → realised daily volatility (e.g. 0.015 for 1.5%).
            If None, the default _DAILY_VOL_TGT / 3 is used (conservative).
        sector_map : dict[str, int] | None
            Maps ticker → sector ID (0–9 as per cross_sectional_trainer.py).
            If None, all names are treated as sector 0 (one mega-sector).

        Returns
        -------
        PortfolioWeights
        """
        sigma_dict = sigma_dict or {}
        sector_map = sector_map or {}

        # ── Step 1: filter to positive signals above minimum threshold ────────
        eligible = {
            t: s for t, s in combined_signals.items()
            if s > _MIN_SIGNAL_THRESHOLD
        }
        if len(eligible) < _MIN_PORTFOLIO_NAMES:
            logger.info(
                "PortfolioConstructor: only %d names above threshold %.2f — "
                "returning empty portfolio",
                len(eligible), _MIN_SIGNAL_THRESHOLD,
            )
            return PortfolioWeights(weights={}, n_names=0, cash_weight=1.0)

        # Take top max_names by signal strength
        sorted_names = sorted(eligible.items(), key=lambda x: -x[1])
        top_names    = dict(sorted_names[: self.max_names])

        # ── Step 2: Kelly sizing per name ─────────────────────────────────────
        raw_weights: dict[str, float] = {}
        for ticker, signal in top_names.items():
            # σ²_return for Kelly denominator.
            # We use the name's realised daily vol if available.
            sigma_d = float(sigma_dict.get(ticker, self.daily_vol_target))
            if sigma_d <= 0.0:
                sigma_d = self.daily_vol_target

            # Kelly: f = IC × signal / σ²
            # `signal` is already in normalised z-score space (≈ unit variance
            # cross-sectionally), so it directly represents the alpha contribution.
            full_kelly = float(self.ic * abs(signal) / (sigma_d ** 2 + 1e-10))
            # Apply fractional Kelly and bound at 1.0 per name
            raw_weight = float(min(self.kelly_fraction * full_kelly, 1.0))
            raw_weights[ticker] = raw_weight

        # ── Step 3: Per-name cap ──────────────────────────────────────────────
        per_name_cap_applied = False
        capped_weights = {}
        for ticker, w in raw_weights.items():
            if w > self.per_name_cap:
                per_name_cap_applied = True
                capped_weights[ticker] = self.per_name_cap
            else:
                capped_weights[ticker] = w

        # ── Step 4: Sector cap ────────────────────────────────────────────────
        sector_cap_applied = False
        capped_weights = self._apply_sector_cap(
            capped_weights, sector_map,
            self.sector_cap,
        )
        # Check if sector capping changed any weights
        if any(capped_weights.get(t, 0) < raw_weights.get(t, 0) for t in raw_weights):
            sector_cap_applied = True

        # ── Step 5: Volatility targeting ─────────────────────────────────────
        vol_scale = self._vol_scale(capped_weights, sigma_dict)
        scaled_weights = {t: w * vol_scale for t, w in capped_weights.items()}

        # ── Step 6: Final normalisation ───────────────────────────────────────
        total = float(sum(scaled_weights.values()))
        if total < 1e-10:
            return PortfolioWeights(weights={}, n_names=0, cash_weight=1.0)

        # Cap total gross at 1.0 (long-only, fully invested maximum)
        if total > 1.0:
            final_weights = {t: w / total for t, w in scaled_weights.items()}
        else:
            final_weights = dict(scaled_weights)

        cash = float(max(0.0, 1.0 - sum(final_weights.values())))

        # ── Sector weights for diagnostics ───────────────────────────────────
        sector_w: dict[int, float] = {}
        for t, w in final_weights.items():
            sid = sector_map.get(t, -1)
            sector_w[sid] = sector_w.get(sid, 0.0) + w

        logger.info(
            "PortfolioConstructor: %d names, invested=%.1f%%, cash=%.1f%%, "
            "vol_scale=%.2f, per_name_cap=%s, sector_cap=%s",
            len(final_weights),
            sum(final_weights.values()) * 100,
            cash * 100,
            vol_scale,
            per_name_cap_applied,
            sector_cap_applied,
        )

        return PortfolioWeights(
            weights               = final_weights,
            cash_weight           = cash,
            n_names               = len(final_weights),
            pre_cap_weights       = dict(raw_weights),
            sector_weights        = sector_w,
            total_gross           = float(sum(raw_weights.values())),
            applied_vol_scale     = vol_scale,
            per_name_cap_applied  = per_name_cap_applied,
            sector_cap_applied    = sector_cap_applied,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _apply_sector_cap(
        self,
        weights: dict[str, float],
        sector_map: dict[str, int],
        sector_cap: float,
    ) -> dict[str, float]:
        """
        Iteratively scale down sectors that breach the cap.

        Within a breaching sector, all names are scaled proportionally so their
        relative weights are preserved. The algorithm converges in one pass
        because we process sectors independently.
        """
        # Compute sector totals
        sector_buckets: dict[int, list[str]] = {}
        for ticker in weights:
            sid = sector_map.get(ticker, -1)
            sector_buckets.setdefault(sid, []).append(ticker)

        result = dict(weights)
        for sid, tickers in sector_buckets.items():
            sector_total = sum(result.get(t, 0.0) for t in tickers)
            if sector_total > sector_cap + 1e-10:
                scale = sector_cap / sector_total
                for t in tickers:
                    result[t] = result.get(t, 0.0) * scale

        return result

    def _vol_scale(
        self,
        weights: dict[str, float],
        sigma_dict: dict[str, float],
    ) -> float:
        """
        Compute a volatility scaling factor so the portfolio's expected
        annualised volatility equals `annual_vol_target`.

        Assumes zero pairwise correlation (conservative upper bound on
        portfolio vol; in practice correlations reduce portfolio vol).

        Returns a scale ∈ (0, 2] applied uniformly to all weights.
        """
        if not weights:
            return 1.0

        # Portfolio daily variance under zero-correlation assumption
        port_var = sum(
            (w * sigma_dict.get(t, self.daily_vol_target)) ** 2
            for t, w in weights.items()
        )
        if port_var <= 0.0:
            return 1.0

        port_vol_daily  = float(np.sqrt(port_var))
        port_vol_annual = float(port_vol_daily * np.sqrt(_TRADING_DAYS))

        if port_vol_annual <= 0.0:
            return 1.0

        # Scale = target / current
        scale = float(np.clip(self.annual_vol_target / port_vol_annual, 0.1, 2.0))
        return scale


# ── Convenience: merge with cross-sectional signals ──────────────────────────

def build_portfolio_from_xs_ranks(
    xs_signals: "pd.DataFrame",
    sigma_dict: Optional[dict[str, float]] = None,
    sector_map: Optional[dict[str, int]]   = None,
    constructor: Optional[PortfolioConstructor] = None,
    date_col: str = "date",
    ticker_col: str = "ticker",
    signal_col: str = "Combined_ZScore",
) -> PortfolioWeights:
    """
    Build a portfolio from the cross-sectional signal DataFrame output by
    `alpha_signals.compute_universe_signals`.

    Uses the most recent date in `xs_signals` as the current date.

    Parameters
    ----------
    xs_signals : pd.DataFrame
        Output of compute_universe_signals() with columns: ticker, date,
        Combined_ZScore (and others).
    sigma_dict : dict[str, float] | None
        Per-ticker realised daily vol.
    sector_map : dict[str, int] | None
        Per-ticker sector ID.
    constructor : PortfolioConstructor | None
        Re-use a pre-configured constructor or create a default one.

    Returns
    -------
    PortfolioWeights
    """
    import pandas as pd

    if xs_signals.empty:
        return PortfolioWeights(weights={}, n_names=0, cash_weight=1.0)

    if constructor is None:
        constructor = PortfolioConstructor()

    # Take the most recent date
    latest_date = xs_signals[date_col].max()
    latest      = xs_signals[xs_signals[date_col] == latest_date]

    combined_signals = dict(zip(latest[ticker_col], latest[signal_col].astype(float)))

    return constructor.construct(combined_signals, sigma_dict, sector_map)
