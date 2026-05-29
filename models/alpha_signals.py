"""
Alpha signal library (Phase 3 — P3-1, P3-2).

Two pure price-based alpha signals for cross-sectional equity selection on the
Indian equity market (Nifty 500 universe):

P3-1  Cross-sectional momentum (12-1 month, Jegadeesh & Titman 1993)
      Signal = log(P[t−skip] / P[t−lookback])
             = 12-month log-return *excluding* the most recent month.
      Why skip the last month: the 1-month reversal effect contaminates a raw
      12-month window. Excluding it gives a cleaner trend-continuation signal.
      Cross-sectional: rank each ticker in the universe on each date (long the
      top quintile, short the bottom quintile). The cross-sectional z-score is
      used in the signal combiner.

P3-2  Short-term reversal (5-day, Jegadeesh 1990, Lo & MacKinlay 1990)
      Signal = −log(P[t] / P[t−5])
      Negative sign: recent losers outperform over the next 5 trading days.
      The mechanism is liquidity provision — market makers demand a price concession
      from impatient sellers; 5-day reversion captures their profit.

Both signals are:
  • Computed as raw log-return windows.
  • Standardised cross-sectionally (z-score or percentile rank per date).
  • Winsorised at ±3σ to limit influence of outlier days.

Usage — offline (universe mode):
------
    from models.alpha_signals import compute_universe_signals
    price_pivot = pd.DataFrame({ticker: series, ...})   # index=date, col=ticker
    signals_df  = compute_universe_signals(price_pivot)
    # signals_df columns: ticker, date, Mom_Raw, Rev_Raw,
    #                     Mom_ZScore, Rev_ZScore, Mom_Rank, Rev_Rank,
    #                     Combined_ZScore

Usage — live (single ticker, inference time):
------
    from models.alpha_signals import compute_single_ticker_signals
    info = compute_single_ticker_signals(close_series, ticker="RELIANCE.NS")
    # info["mom_signal"]  — float or None
    # info["rev_signal"]  — float or None
    # info["combined"]    — equal-weighted blend
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Window constants ──────────────────────────────────────────────────────────
_MOM_LOOKBACK  = 252   # 12 months in trading days
_MOM_SKIP      = 21    # skip last month to avoid 1-month reversal noise
_REV_LOOKBACK  = 5     # 5 trading days

# Cross-sectional winsorisation clip (z-scores)
_ZSCORE_CLIP   = 3.0

# Equal-weight blend (can be overridden by SignalCombiner)
_MOM_BLEND_WT  = 0.60
_REV_BLEND_WT  = 0.40


# ── Single-ticker scorers ─────────────────────────────────────────────────────

@dataclass
class MomentumSignal:
    """
    12-1 month cross-sectional momentum scorer.

    Raw signal = log( P[t − skip] / P[t − lookback] )
               = 12-month log-return excluding the most recent 21 days.

    Returns a float in roughly (−1.5, +1.5) log-return space.
    Positive → strong upward trend continuation expected.
    Negative → downward trend continuation expected.
    """

    lookback: int = _MOM_LOOKBACK
    skip: int     = _MOM_SKIP

    def score(self, close: pd.Series) -> Optional[float]:
        """Latest momentum score for a single ticker. Returns None if insufficient data."""
        n = len(close)
        if n <= self.lookback:
            return None

        p_skip  = float(close.iloc[-(self.skip + 1)])
        p_lb    = float(close.iloc[-(self.lookback + 1)])

        if p_lb <= 0.0 or p_skip <= 0.0:
            return None

        raw = float(np.log(p_skip / p_lb))
        return float(np.clip(raw, -1.5, 1.5))

    def score_series(self, close: pd.Series) -> pd.Series:
        """
        Rolling 12-1 month momentum aligned to `close.index`.

        NaN before index position `lookback`. No look-ahead: uses shift(),
        so the value at date t uses data available up to and including t.
        """
        p_skip = close.shift(self.skip)
        p_lb   = close.shift(self.lookback)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.log(p_skip / p_lb)
        return raw.clip(-1.5, 1.5)


@dataclass
class ReversalSignal:
    """
    5-day short-term reversal scorer.

    Raw signal = −log( P[t] / P[t−lookback] )

    Positive → recent loser, rebound expected.
    Negative → recent winner, pullback expected.

    The negation makes the signal directionally consistent with momentum:
    both signals are positive when the model expects future outperformance.
    """

    lookback: int = _REV_LOOKBACK

    def score(self, close: pd.Series) -> Optional[float]:
        """Latest reversal score for a single ticker. Returns None if insufficient data."""
        n = len(close)
        if n <= self.lookback:
            return None

        p_t  = float(close.iloc[-1])
        p_lb = float(close.iloc[-(self.lookback + 1)])

        if p_lb <= 0.0:
            return None

        raw = float(-np.log(p_t / p_lb))
        return float(np.clip(raw, -0.5, 0.5))

    def score_series(self, close: pd.Series) -> pd.Series:
        """
        Rolling 5-day reversal aligned to `close.index`.

        Negate sign: raw return is negated so positive = expected rebound.
        """
        p_lb = close.shift(self.lookback)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = -np.log(close / p_lb)
        return raw.clip(-0.5, 0.5)


# ── Cross-sectional standardisation ──────────────────────────────────────────

def _cross_sectional_zscore(
    long_df: pd.DataFrame,
    col: str,
    date_col: str = "_date_idx",
) -> pd.Series:
    """
    Compute cross-sectional z-score of `col` per date in `long_df`.

    `long_df` is in long format with `date_col` holding the date value.
    Returns a pd.Series aligned to long_df.index, clipped at ±_ZSCORE_CLIP.
    """
    result = pd.Series(np.nan, index=long_df.index, dtype=float)

    for dt, grp in long_df.groupby(date_col):
        vals = grp[col]
        valid = vals.dropna()
        if len(valid) < 2:
            continue
        mu   = float(valid.mean())
        sigma = float(valid.std(ddof=0)) + 1e-10
        zs   = ((vals - mu) / sigma).clip(-_ZSCORE_CLIP, _ZSCORE_CLIP)
        result.loc[grp.index] = zs

    return result


def _cross_sectional_rank(
    long_df: pd.DataFrame,
    col: str,
    date_col: str = "_date_idx",
) -> pd.Series:
    """
    Percentile rank [0, 1] within the cross-section on each date,
    then shift to [−0.5, +0.5] so the mean is 0.

    Rankings are stable even when many tickers share the same value.
    """
    result = pd.Series(np.nan, index=long_df.index, dtype=float)

    for dt, grp in long_df.groupby(date_col):
        ranked = grp[col].rank(pct=True, method="average") - 0.5
        result.loc[grp.index] = ranked

    return result


# ── Universe-level computation ────────────────────────────────────────────────

def compute_universe_signals(
    price_pivot: pd.DataFrame,
    min_history_days: int = 280,
    winsorise: bool = True,
) -> pd.DataFrame:
    """
    Compute momentum and reversal alpha signals for a universe of tickers.

    Parameters
    ----------
    price_pivot : pd.DataFrame
        Wide-format price matrix: DatetimeIndex rows, ticker-symbol columns.
        Each cell contains the adjusted close price. NaN allowed for missing
        sessions (e.g. trading halts).
    min_history_days : int
        Tickers with fewer non-NaN rows are silently skipped.
    winsorise : bool
        If True, the cross-sectional z-scores are clipped at ±3σ before
        ranking. Protects the combined signal from extreme outliers.

    Returns
    -------
    pd.DataFrame
        Long format with one row per (ticker, date):
            ticker, date,
            Mom_Raw, Rev_Raw,
            Mom_ZScore, Rev_ZScore,
            Mom_Rank, Rev_Rank,
            Combined_ZScore

        Combined_ZScore = 0.6 × Mom_ZScore + 0.4 × Rev_ZScore (equal-weighted
        default; the SignalCombiner can override this in production).
    """
    if price_pivot.empty:
        return pd.DataFrame()

    mom_scorer = MomentumSignal()
    rev_scorer = ReversalSignal()

    records: list[dict] = []

    for ticker in price_pivot.columns:
        close = price_pivot[ticker].dropna()
        if len(close) < min_history_days:
            logger.debug(
                "alpha_signals: skipping %s — only %d bars (need %d)",
                ticker, len(close), min_history_days,
            )
            continue

        mom_series = mom_scorer.score_series(close)
        rev_series = rev_scorer.score_series(close)

        valid_mask = mom_series.notna() & rev_series.notna()
        if valid_mask.sum() == 0:
            logger.debug("alpha_signals: no valid rows for %s", ticker)
            continue

        for dt in close.index[valid_mask]:
            records.append({
                "ticker":  ticker,
                "date":    dt,
                "Mom_Raw": float(mom_series.at[dt]),
                "Rev_Raw": float(rev_series.at[dt]),
            })

    if not records:
        logger.warning(
            "alpha_signals.compute_universe_signals: no valid (ticker, date) pairs. "
            "Check that price_pivot has sufficient history."
        )
        return pd.DataFrame()

    out = pd.DataFrame(records)
    out["_date_idx"] = out["date"]
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)

    # ── Cross-sectional standardisation ──────────────────────────────────────
    out["Mom_ZScore"] = _cross_sectional_zscore(out, "Mom_Raw")
    out["Rev_ZScore"] = _cross_sectional_zscore(out, "Rev_Raw")
    out["Mom_Rank"]   = _cross_sectional_rank(out, "Mom_Raw")
    out["Rev_Rank"]   = _cross_sectional_rank(out, "Rev_Raw")

    # ── Composite signal (equal-weight default) ───────────────────────────────
    out["Combined_ZScore"] = (
        _MOM_BLEND_WT * out["Mom_ZScore"].fillna(0.0) +
        _REV_BLEND_WT * out["Rev_ZScore"].fillna(0.0)
    )

    out = out.drop(columns=["_date_idx"])

    n_tickers = out["ticker"].nunique()
    n_dates   = out["date"].nunique()
    logger.info(
        "alpha_signals: computed signals for %d tickers × %d dates = %d rows",
        n_tickers, n_dates, len(out),
    )
    return out


# ── Live inference (single ticker) ───────────────────────────────────────────

def compute_single_ticker_signals(
    close: pd.Series,
    ticker: str = "",
) -> dict:
    """
    Compute the latest momentum and reversal alpha scores for a single ticker.

    Intended for use at inference time inside the prediction pipeline where the
    full cross-sectional universe is not available. The returned scores are in
    raw log-return space and cannot be compared across tickers (no universe
    z-scoring). Use `compute_universe_signals` when ranking across tickers.

    Parameters
    ----------
    close : pd.Series
        Adjusted close price series (DatetimeIndex, float values).
    ticker : str
        Ticker string for logging purposes.

    Returns
    -------
    dict with keys:
        ticker         : str
        mom_signal     : float | None   — 12-1 month raw momentum
        rev_signal     : float | None   — 5-day raw reversal
        combined       : float | None   — weighted blend (60/40)
        has_full_data  : bool           — True if both signals computed
    """
    mom = MomentumSignal().score(close)
    rev = ReversalSignal().score(close)

    if mom is not None and rev is not None:
        combined = _MOM_BLEND_WT * mom + _REV_BLEND_WT * rev
        has_full  = True
    elif mom is not None:
        combined = float(mom)
        has_full  = False
    elif rev is not None:
        combined = float(rev)
        has_full  = False
    else:
        combined  = None
        has_full  = False

    return {
        "ticker":        ticker,
        "mom_signal":    mom,
        "rev_signal":    rev,
        "combined":      combined,
        "has_full_data": has_full,
    }


# ── Rolling IC tracker (for signal combiner calibration) ─────────────────────

class SignalICTracker:
    """
    Tracks the rolling Information Coefficient (IC) of each alpha signal
    against future returns. Used by the SignalCombiner to calibrate weights.

    IC = rank_corr(signal[t], return[t+horizon]) over a rolling window.
    A well-performing signal should have IC > 0.03 (3%) consistently.

    Usage
    -----
        tracker = SignalICTracker(window=60)
        tracker.update(date, mom_signal, rev_signal, future_return)
        ic_stats = tracker.rolling_ic(signal="mom")
    """

    def __init__(self, window: int = 60) -> None:
        self.window = window
        self._records: list[dict] = []

    def update(
        self,
        date,
        mom_signal: Optional[float],
        rev_signal: Optional[float],
        future_return: Optional[float],
    ) -> None:
        """Append one (date, signal_values, outcome) record."""
        if future_return is None:
            return
        self._records.append({
            "date":          date,
            "mom_signal":    mom_signal,
            "rev_signal":    rev_signal,
            "future_return": future_return,
        })

    def rolling_ic(self, signal: str = "mom") -> dict:
        """
        Compute IC statistics over the last `window` records.

        Returns dict with keys: ic_mean, ic_std, ic_t_stat, ic_ir (information ratio).
        ic_ir ≡ mean / std — annualised IC information ratio.
        """
        if len(self._records) < 10:
            return {"ic_mean": 0.0, "ic_std": 1.0, "ic_t_stat": 0.0, "ic_ir": 0.0}

        col = f"{signal}_signal"
        df  = pd.DataFrame(self._records[-self.window:])

        valid = df[[col, "future_return"]].dropna()
        if len(valid) < 5:
            return {"ic_mean": 0.0, "ic_std": 1.0, "ic_t_stat": 0.0, "ic_ir": 0.0}

        # Rank correlation (Spearman) — more robust than Pearson for fat-tailed returns
        ic_series = valid[col].rank().corr(valid["future_return"].rank())
        # Single-period IC — use rolling window to build a distribution
        # For a rolling window of n observations we compute IC per date via
        # a 20-day sub-window for better temporal resolution.
        sub_ics: list[float] = []
        sub_w = max(5, self.window // 10)
        for i in range(sub_w, len(valid) + 1, max(1, sub_w // 2)):
            chunk = valid.iloc[max(0, i - sub_w): i]
            if len(chunk) < 4:
                continue
            ic = chunk[col].rank().corr(chunk["future_return"].rank())
            if not np.isnan(ic):
                sub_ics.append(ic)

        if not sub_ics:
            return {"ic_mean": float(ic_series), "ic_std": 1.0, "ic_t_stat": 0.0, "ic_ir": 0.0}

        ic_arr  = np.array(sub_ics)
        ic_mean = float(np.mean(ic_arr))
        ic_std  = float(np.std(ic_arr, ddof=1)) + 1e-10
        ic_t    = float(ic_mean / ic_std * np.sqrt(len(ic_arr)))
        ic_ir   = float(ic_mean / ic_std * np.sqrt(252))  # annualised

        return {
            "ic_mean":   ic_mean,
            "ic_std":    ic_std,
            "ic_t_stat": ic_t,
            "ic_ir":     ic_ir,
        }

    def get_prior_ic(self, signal: str = "mom") -> float:
        """
        Return best-estimate IC. Falls back to prior if insufficient history.

        Prior ICs (Nifty universe, academic consensus):
            mom → 0.030   (momentum is well-documented in Indian equities)
            rev → 0.020   (short-term reversal is noisier)
        """
        priors = {"mom": 0.030, "rev": 0.020}
        stats  = self.rolling_ic(signal)
        if abs(stats["ic_t_stat"]) >= 1.5:
            return stats["ic_mean"]
        return priors.get(signal, 0.020)
