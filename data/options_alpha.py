"""
Options flow alpha extractor (Phase 3 — P3-3).

Extracts two well-researched options-market alpha signals for Indian equities:

1.  PCR Δ (Put-Call Ratio Delta)
    PCR = total PE open interest / total CE open interest.
    * PCR > 1 and rising  → more puts than calls, bearish hedging → bearish signal.
    * PCR < 1 and falling → call buying / put unwinding → bullish signal.
    Signal = −ΔlnPCR  (negative sign: rising PCR is bearish for the underlying).
    Range: approximately (−0.3, +0.3) in practice.

2.  IV Skew Δ (Implied Volatility Skew Delta)
    Skew = atm_put_iv − atm_call_iv (positive = put skew = market fear premium).
    Rising skew → hedging demand is accelerating → bearish signal.
    Signal = −Δskew / normalisation  (negative: rising skew = bearish).
    Range: approximately (−1, +1).

Both signals are cached per ticker so the *delta* (change from last observation)
can be computed on the next call. Cache is stored in `data/ledger/options_cache.json`
so it survives process restarts without a database.

Usage — live (called from prediction_service or hybrid_model):
------
    from data.options_alpha import fetch_options_alpha
    alpha = fetch_options_alpha("RELIANCE.NS")
    # alpha: OptionsAlphaResult with pcr_signal, iv_skew_signal, combined_signal

Usage — batch (for backtesting with historical option-chain snapshots):
------
    from data.options_alpha import score_option_features_dict
    result = score_option_features_dict(current_features, prior_features)

Note on data availability:
    Live NSE option chains are available only for F&O-eligible stocks
    (~230 names). For non-F&O stocks the function returns a neutral
    OptionsAlphaResult with all signals = 0.0 and `available = False`.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np

from data.option_chain import OptionChainSnapshot, extract_option_features, fetch_option_chain

logger = logging.getLogger(__name__)

# ── Cache ────────────────────────────────────────────────────────────────────
_CACHE_PATH = Path("data/ledger/options_cache.json")
_CACHE_LOCK = threading.Lock()

# Normalisation constants (fit on Nifty 50 historical option chains)
_PCR_NORM          = 0.15   # 1-σ daily log-PCR change ≈ 0.15
_SKEW_NORM         = 5.0    # 1-σ daily skew change ≈ 5 vol points

# Signal blend weights
_PCR_WEIGHT        = 0.55
_SKEW_WEIGHT       = 0.45

# Maximum signal values before clipping
_SIGNAL_CLIP       = 1.5


@dataclass
class OptionsAlphaResult:
    """All options-flow alpha signals for one ticker at one point in time."""

    ticker: str
    fetched_at: date

    # PCR-based signal: −Δln(PCR) / _PCR_NORM
    pcr_level: float = 1.0          # current PCR
    pcr_signal: float = 0.0         # −Δln(PCR) normalised, range ≈ (−1.5, +1.5)

    # IV skew-based signal: −Δskew / _SKEW_NORM
    iv_skew_level: float = 0.0      # current IV skew (put_iv − call_iv, vol points)
    iv_skew_signal: float = 0.0     # −Δskew normalised

    # Weighted combination
    combined_signal: float = 0.0    # pcr_weight × pcr_signal + skew_weight × iv_skew_signal

    # Diagnostics
    atm_iv: float = 0.0
    weighted_iv: float = 0.0
    call_wall_distance: float = 0.0
    put_wall_distance: float = 0.0
    max_pain_distance: float = 0.0

    # Quality flag
    available: bool = False         # False when no option chain data exists


# ── Cache I/O ─────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    with _CACHE_LOCK:
        if not _CACHE_PATH.exists():
            return {}
        try:
            with _CACHE_PATH.open("r") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return {}


def _save_cache(cache: dict) -> None:
    with _CACHE_LOCK:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with _CACHE_PATH.open("w") as fh:
                json.dump(cache, fh, indent=2, default=str)
        except OSError as exc:
            logger.warning("options_alpha: failed to write cache: %s", exc)


def _ticker_cache_key(ticker: str) -> str:
    return ticker.upper().replace(".NS", "").replace(".BO", "")


# ── Core signal computation ───────────────────────────────────────────────────

def score_option_features_dict(
    current: dict,
    prior: Optional[dict],
) -> dict:
    """
    Compute PCR and IV skew alpha signals given current and prior option feature
    dicts (as returned by `extract_option_features`).

    When `prior` is None (first observation), signals default to 0.0.

    Returns a dict with pcr_signal, iv_skew_signal, combined_signal plus
    diagnostic fields (atm_iv, weighted_iv, etc.).
    """
    pcr_now    = float(current.get("put_call_ratio", 1.0))
    skew_now   = float(current.get("iv_skew", 0.0))

    if prior is None:
        # First observation: no delta available — return neutral signal.
        pcr_signal    = 0.0
        iv_skew_signal = 0.0
    else:
        pcr_prior  = float(prior.get("put_call_ratio", 1.0))
        skew_prior = float(prior.get("iv_skew", 0.0))

        # PCR signal: −Δln(PCR). Rising PCR (more puts) → bearish → negative signal.
        # Guard against zero PCR (market closed / no open interest).
        if pcr_prior > 0.01 and pcr_now > 0.01:
            delta_ln_pcr  = float(np.log(pcr_now / pcr_prior))
            pcr_signal    = float(np.clip(-delta_ln_pcr / _PCR_NORM, -_SIGNAL_CLIP, _SIGNAL_CLIP))
        else:
            pcr_signal = 0.0

        # IV skew signal: −Δskew / normalisation.
        delta_skew     = skew_now - skew_prior
        iv_skew_signal = float(np.clip(-delta_skew / _SKEW_NORM, -_SIGNAL_CLIP, _SIGNAL_CLIP))

    combined = float(
        _PCR_WEIGHT * pcr_signal + _SKEW_WEIGHT * iv_skew_signal
    )

    return {
        "pcr_level":         pcr_now,
        "pcr_signal":        pcr_signal,
        "iv_skew_level":     skew_now,
        "iv_skew_signal":    iv_skew_signal,
        "combined_signal":   combined,
        "atm_iv":            float(current.get("atm_iv", 0.0)),
        "weighted_iv":       float(current.get("weighted_iv", 0.0)),
        "call_wall_distance": float(current.get("call_wall_distance", 0.0)),
        "put_wall_distance": float(current.get("put_wall_distance", 0.0)),
        "max_pain_distance": float(current.get("max_pain_distance", 0.0)),
    }


def _neutral_result(ticker: str) -> OptionsAlphaResult:
    return OptionsAlphaResult(ticker=ticker, fetched_at=date.today(), available=False)


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_options_alpha(
    ticker: str,
    session=None,
    force_refresh: bool = False,
) -> OptionsAlphaResult:
    """
    Fetch the live NSE option chain for `ticker`, compute PCR Δ and IV skew Δ
    signals, and return an `OptionsAlphaResult`.

    Caches the latest snapshot so the delta can be computed on the next call.
    If the chain fetch fails (NSE endpoint down, ticker not F&O eligible, market
    closed), returns a neutral `OptionsAlphaResult` with `available=False`.

    Parameters
    ----------
    ticker : str
        NSE ticker, e.g. "RELIANCE.NS". The `.NS` suffix is stripped internally.
    session : requests.Session | None
        Optional pre-warmed HTTP session to reuse across calls. Useful when
        fetching option chains for many tickers back-to-back.
    force_refresh : bool
        If True, ignore the 1-hour intraday staleness guard and always re-fetch.

    Returns
    -------
    OptionsAlphaResult
    """
    cache_key = _ticker_cache_key(ticker)
    cache     = _load_cache()
    prior_rec = cache.get(cache_key)  # may be None

    # ── Fetch live option chain ───────────────────────────────────────────────
    snap: Optional[OptionChainSnapshot] = fetch_option_chain(ticker, session=session)
    if snap is None:
        logger.debug("options_alpha: no option chain for %s (not F&O or NSE down)", ticker)
        return _neutral_result(ticker)

    current_features = extract_option_features(snap)

    # Extract prior feature dict from cache (or None for first call)
    prior_features: Optional[dict] = prior_rec.get("features") if prior_rec else None

    # ── Compute signals ───────────────────────────────────────────────────────
    scored = score_option_features_dict(current_features, prior_features)

    # ── Update cache ─────────────────────────────────────────────────────────
    cache[cache_key] = {
        "ticker":     ticker,
        "updated_at": date.today().isoformat(),
        "features":   current_features,
    }
    _save_cache(cache)

    result = OptionsAlphaResult(
        ticker               = ticker,
        fetched_at           = date.today(),
        pcr_level            = scored["pcr_level"],
        pcr_signal           = scored["pcr_signal"],
        iv_skew_level        = scored["iv_skew_level"],
        iv_skew_signal       = scored["iv_skew_signal"],
        combined_signal      = scored["combined_signal"],
        atm_iv               = scored["atm_iv"],
        weighted_iv          = scored["weighted_iv"],
        call_wall_distance   = scored["call_wall_distance"],
        put_wall_distance    = scored["put_wall_distance"],
        max_pain_distance    = scored["max_pain_distance"],
        available            = True,
    )

    logger.info(
        "options_alpha: %s  PCR=%.2f (Δsig=%.3f)  IVskew=%.1f (Δsig=%.3f)  combined=%.3f",
        ticker,
        result.pcr_level,
        result.pcr_signal,
        result.iv_skew_level,
        result.iv_skew_signal,
        result.combined_signal,
    )
    return result


def batch_fetch_options_alpha(
    tickers: list[str],
    max_workers: int = 4,
) -> dict[str, OptionsAlphaResult]:
    """
    Fetch options alpha for a list of tickers in parallel using a shared
    HTTP session to amortise the NSE cookie dance across requests.

    Returns a dict mapping ticker → OptionsAlphaResult.
    Failed / unavailable tickers map to OptionsAlphaResult(available=False).
    """
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Each worker gets its own session (NSE cookies are per-session)
    def _fetch_one(ticker: str) -> tuple[str, OptionsAlphaResult]:
        sess = requests.Session()
        try:
            return ticker, fetch_options_alpha(ticker, session=sess)
        except Exception as exc:
            logger.warning("options_alpha batch: %s failed: %s", ticker, exc)
            return ticker, _neutral_result(ticker)

    results: dict[str, OptionsAlphaResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tickers}
        for fut in as_completed(futures):
            ticker, result = fut.result()
            results[ticker] = result

    n_ok   = sum(1 for r in results.values() if r.available)
    n_fail = len(results) - n_ok
    logger.info(
        "options_alpha.batch_fetch: %d/%d tickers returned valid chains (%d failed)",
        n_ok, len(results), n_fail,
    )
    return results


def options_alpha_to_feature_dict(result: OptionsAlphaResult) -> dict[str, float]:
    """
    Convert an OptionsAlphaResult to a flat feature dict for model ingestion.

    Handles the `available=False` case by returning zeros — the hybrid model
    treats missing option data as a neutral (uninformative) state.
    """
    if not result.available:
        return {
            "Options_PCR":           1.0,
            "Options_PCR_Signal":    0.0,
            "Options_IVSkew":        0.0,
            "Options_IVSkew_Signal": 0.0,
            "Options_Combined":      0.0,
            "Options_ATM_IV":        0.0,
            "Options_Weighted_IV":   0.0,
            "Options_MaxPain_Dist":  0.0,
            "Options_Available":     0.0,
        }

    return {
        "Options_PCR":           result.pcr_level,
        "Options_PCR_Signal":    result.pcr_signal,
        "Options_IVSkew":        result.iv_skew_level,
        "Options_IVSkew_Signal": result.iv_skew_signal,
        "Options_Combined":      result.combined_signal,
        "Options_ATM_IV":        result.atm_iv,
        "Options_Weighted_IV":   result.weighted_iv,
        "Options_MaxPain_Dist":  result.max_pain_distance,
        "Options_Available":     1.0,
    }
