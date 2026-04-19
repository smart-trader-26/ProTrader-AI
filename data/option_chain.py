"""
NSE option-chain fetcher (A4.1) + feature extractor (A4.2).

NSE's JSON endpoint rejects clients that don't look like a browser session.
Workflow to bypass this without a paid data vendor:

    1. Hit `https://www.nseindia.com/option-chain` with a browser User-Agent
       and a fresh cookie jar. NSE sets ~5 cookies we must echo back.
    2. Hit the JSON endpoint `/api/option-chain-equities?symbol=<SYM>` with
       the same session. Every request renews the cookies.
    3. Strip the `.NS` suffix — NSE's API uses the raw underlying symbol
       (e.g. `RELIANCE`, not `RELIANCE.NS`).

No paid keys, no rate limits for modest use (one call per ticker per
refresh). If the endpoint returns non-200 (common during market-closed
pre-open), callers should fall back to last-known values.

All network calls have a short timeout and return `None` on any failure —
callers in the hybrid model treat "no options data" as a valid state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import requests

_NSE_HOME = "https://www.nseindia.com/option-chain"
_NSE_API = "https://www.nseindia.com/api/option-chain-equities"

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain",
    "Connection": "keep-alive",
}


@dataclass
class OptionChainSnapshot:
    """One NSE option-chain snapshot for a single underlying."""

    symbol: str
    underlying_price: float
    expiry: str
    fetched_at: date
    # Long-format chain: strike × {CE_OI, CE_IV, CE_Chng_OI, PE_OI, PE_IV, PE_Chng_OI}
    chain: pd.DataFrame


def _normalize_symbol(ticker: str) -> str:
    """`RELIANCE.NS` → `RELIANCE`. NSE option API doesn't take the `.NS` suffix."""
    base = ticker.upper().strip()
    for suffix in (".NS", ".BO"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


def fetch_option_chain(
    ticker: str, timeout: float = 8.0, session: requests.Session | None = None
) -> OptionChainSnapshot | None:
    """
    Fetch the live NSE option chain for `ticker`. Returns None on any error.

    Handles the cookie dance: priming GET to /option-chain, then the API
    call. Callers cache this — it's ~2s per call and unchanged intraday
    within a minute.
    """
    symbol = _normalize_symbol(ticker)
    try:
        sess = session or requests.Session()
        sess.headers.update(_BROWSER_HEADERS)

        # Cookie priming — NSE sets ak_bmsc + bm_sv + nsit etc. on this call.
        primer = sess.get(_NSE_HOME, timeout=timeout)
        if primer.status_code != 200:
            return None

        resp = sess.get(
            _NSE_API, params={"symbol": symbol}, timeout=timeout
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
    except (requests.RequestException, ValueError):
        return None

    records = (data or {}).get("records") or {}
    rows = records.get("data") or []
    if not rows:
        return None

    expiries = records.get("expiryDates") or []
    expiry = expiries[0] if expiries else records.get("expiryDate", "")
    underlying_value = float(records.get("underlyingValue") or 0.0)

    # Keep only rows for the nearest expiry to avoid double-counting strikes
    chain_rows = []
    for r in rows:
        if r.get("expiryDate") != expiry:
            continue
        strike = r.get("strikePrice")
        if strike is None:
            continue
        ce = r.get("CE") or {}
        pe = r.get("PE") or {}
        chain_rows.append(
            {
                "strike": float(strike),
                "CE_OI": float(ce.get("openInterest") or 0),
                "CE_Chng_OI": float(ce.get("changeinOpenInterest") or 0),
                "CE_IV": float(ce.get("impliedVolatility") or 0),
                "CE_LTP": float(ce.get("lastPrice") or 0),
                "PE_OI": float(pe.get("openInterest") or 0),
                "PE_Chng_OI": float(pe.get("changeinOpenInterest") or 0),
                "PE_IV": float(pe.get("impliedVolatility") or 0),
                "PE_LTP": float(pe.get("lastPrice") or 0),
            }
        )

    if not chain_rows:
        return None

    chain_df = pd.DataFrame(chain_rows).sort_values("strike").reset_index(drop=True)
    return OptionChainSnapshot(
        symbol=symbol,
        underlying_price=underlying_value,
        expiry=str(expiry),
        fetched_at=date.today(),
        chain=chain_df,
    )


def extract_option_features(snap: OptionChainSnapshot | None) -> dict[str, float]:
    """
    Reduce an option-chain snapshot to a dict of scalar features (A4.2).

    Output (always a non-empty dict — zeros when `snap` is None):
        put_call_ratio:      sum(PE_OI) / sum(CE_OI)
        put_call_chg_ratio:  sum(PE_Chng_OI) / sum(CE_Chng_OI)
        max_pain_distance:   (spot - max_pain_strike) / spot  — signed
        atm_iv:              average IV at the nearest strike
        iv_skew:             PE_IV(ATM) - CE_IV(ATM) — positive = put skew (bearish)
        oi_call_concentration: pct of CE OI in top-3 strikes
        oi_put_concentration:  pct of PE OI in top-3 strikes
        call_wall_distance:   (call_wall - spot) / spot (highest CE OI strike)
        put_wall_distance:    (put_wall  - spot) / spot (highest PE OI strike)
        weighted_iv:          OI-weighted average of CE/PE IVs

    Safe to call with `None` — returns zeros so downstream feature merging
    stays simple.
    """
    zero = {
        "put_call_ratio": 1.0,
        "put_call_chg_ratio": 1.0,
        "max_pain_distance": 0.0,
        "atm_iv": 0.0,
        "iv_skew": 0.0,
        "oi_call_concentration": 0.0,
        "oi_put_concentration": 0.0,
        "call_wall_distance": 0.0,
        "put_wall_distance": 0.0,
        "weighted_iv": 0.0,
    }
    if snap is None or snap.chain.empty or snap.underlying_price <= 0:
        return zero

    df = snap.chain
    spot = float(snap.underlying_price)

    ce_oi_sum = df["CE_OI"].sum()
    pe_oi_sum = df["PE_OI"].sum()
    ce_chg_sum = df["CE_Chng_OI"].sum()
    pe_chg_sum = df["PE_Chng_OI"].sum()

    pcr = float(pe_oi_sum / ce_oi_sum) if ce_oi_sum > 0 else 1.0
    pcr_chg = (
        float(pe_chg_sum / ce_chg_sum)
        if abs(ce_chg_sum) > 0
        else 1.0
    )

    # Max pain — strike where total pain (call writers + put writers) is minimised
    strikes = df["strike"].to_numpy()
    call_pain = np.array(
        [np.sum(np.maximum(spot_k - strikes, 0) * df["CE_OI"].to_numpy()) for spot_k in strikes]
    )
    put_pain = np.array(
        [np.sum(np.maximum(strikes - spot_k, 0) * df["PE_OI"].to_numpy()) for spot_k in strikes]
    )
    total_pain = call_pain + put_pain
    max_pain_strike = float(strikes[int(np.argmin(total_pain))])
    max_pain_distance = float((spot - max_pain_strike) / spot)

    # ATM row (strike nearest spot)
    atm_idx = int(np.argmin(np.abs(strikes - spot)))
    atm_ce_iv = float(df["CE_IV"].iloc[atm_idx])
    atm_pe_iv = float(df["PE_IV"].iloc[atm_idx])
    atm_iv = float((atm_ce_iv + atm_pe_iv) / 2) if (atm_ce_iv + atm_pe_iv) > 0 else 0.0
    iv_skew = float(atm_pe_iv - atm_ce_iv)

    # OI concentration in top-3 strikes
    top3_ce = df.nlargest(3, "CE_OI")["CE_OI"].sum()
    top3_pe = df.nlargest(3, "PE_OI")["PE_OI"].sum()
    oi_call_conc = float(top3_ce / ce_oi_sum) if ce_oi_sum > 0 else 0.0
    oi_put_conc = float(top3_pe / pe_oi_sum) if pe_oi_sum > 0 else 0.0

    # Call wall = strike with max CE OI (resistance); Put wall = max PE OI (support)
    call_wall = float(df.loc[df["CE_OI"].idxmax(), "strike"])
    put_wall = float(df.loc[df["PE_OI"].idxmax(), "strike"])
    call_wall_dist = float((call_wall - spot) / spot)
    put_wall_dist = float((put_wall - spot) / spot)

    # Weighted IV across all strikes
    all_oi = (df["CE_OI"] + df["PE_OI"]).to_numpy()
    all_iv = ((df["CE_OI"] * df["CE_IV"]) + (df["PE_OI"] * df["PE_IV"])).to_numpy()
    tot_oi = all_oi.sum()
    weighted_iv = float(all_iv.sum() / tot_oi) if tot_oi > 0 else 0.0

    return {
        "put_call_ratio": pcr,
        "put_call_chg_ratio": pcr_chg,
        "max_pain_distance": max_pain_distance,
        "atm_iv": atm_iv,
        "iv_skew": iv_skew,
        "oi_call_concentration": oi_call_conc,
        "oi_put_concentration": oi_put_conc,
        "call_wall_distance": call_wall_dist,
        "put_wall_distance": put_wall_dist,
        "weighted_iv": weighted_iv,
    }


def get_option_features(ticker: str) -> dict[str, float]:
    """One-shot convenience: fetch + extract. Zeros on failure."""
    snap = fetch_option_chain(ticker)
    return extract_option_features(snap)
