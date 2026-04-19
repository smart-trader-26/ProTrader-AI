"""
Macro features (A5.1) — global/cross-asset drivers of Indian equities.

All sources are free yfinance tickers:

    INR=X    USD/INR — rupee weakness is a headwind for importers, tailwind for IT/pharma
    CL=F     Crude oil — inverse to energy-importing sectors (OMCs, aviation, paints)
    ^TNX     US 10-year yield — risk-free rate, FII flow driver
    GC=F     Gold — safe-haven, inverse to equity risk-on
    ^GSPC    S&P 500 — global risk benchmark; Nifty tends to co-move
    ^VIX     US VIX — global fear gauge (India VIX already separate in hybrid_model)

Features are return-based (1-day, 5-day pct changes) so they're stationary
and comparable across instruments. The DataFrame indexed by date carries
all macro factors side-by-side and is joined into `create_hybrid_model`'s
feature matrix exactly like FII/DII and VIX already are.
"""

from __future__ import annotations

import pandas as pd

MACRO_TICKERS = {
    "USDINR": "INR=X",
    "Crude":  "CL=F",
    "US10Y":  "^TNX",
    "Gold":   "GC=F",
    "SP500":  "^GSPC",
    "USVIX":  "^VIX",
}


def fetch_macro_series(start, end, tickers: dict[str, str] | None = None) -> pd.DataFrame:
    """
    Pull close-price history for each macro ticker; return a wide DataFrame.

    Missing series are simply omitted — the caller's feature extractor treats
    any missing column as "macro factor unavailable" and zero-fills. This
    lets the function degrade gracefully when yfinance blocks one ticker.

    Args:
        start, end: yfinance-compatible date bounds
        tickers:    {label: yf_symbol} override (defaults to MACRO_TICKERS)

    Returns:
        DataFrame indexed by date, one column per label: `<label>_Close`.
        Empty DataFrame on complete failure.
    """
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()

    tickers = tickers or MACRO_TICKERS
    frames: list[pd.DataFrame] = []
    for label, symbol in tickers.items():
        try:
            raw = yf.Ticker(symbol).history(start=start, end=end)
        except Exception:
            continue
        if raw is None or raw.empty or "Close" not in raw.columns:
            continue
        series = raw[["Close"]].copy()
        series.columns = [f"{label}_Close"]
        if series.index.tz is not None:
            series.index = series.index.tz_localize(None)
        frames.append(series)

    if not frames:
        return pd.DataFrame()

    # Outer join keeps the full calendar; callers forward-fill on join.
    macro = frames[0]
    for f in frames[1:]:
        macro = macro.join(f, how="outer")
    macro = macro.sort_index()
    return macro


def extract_macro_features(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw close prices to stationary return features.

    For each label column `X_Close` produces:
        X_1d_chg:  1-day log return
        X_5d_chg:  5-day log return

    Returns an empty DataFrame when input is empty so the caller can safely
    do a no-op join.
    """
    if macro_df is None or macro_df.empty:
        return pd.DataFrame()

    import numpy as np

    out = pd.DataFrame(index=macro_df.index)
    for col in macro_df.columns:
        if not col.endswith("_Close"):
            continue
        label = col.replace("_Close", "")
        series = macro_df[col]
        out[f"{label}_1d_chg"] = np.log(series / series.shift(1)).clip(-0.2, 0.2)
        out[f"{label}_5d_chg"] = np.log(series / series.shift(5)).clip(-0.4, 0.4)
    return out.fillna(0.0)


def get_macro_features(start, end) -> pd.DataFrame:
    """One-shot convenience: fetch + transform."""
    return extract_macro_features(fetch_macro_series(start, end))
