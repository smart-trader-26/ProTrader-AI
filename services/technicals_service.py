"""
Technicals + chart-pattern + FII/DII services.

Thin wrappers that translate legacy DataFrame/dict shapes into the typed DTOs
in `schemas.technicals`. Keeps FastAPI routers dumb and the Streamlit path
untouched.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from schemas.technicals import (
    DetectedPattern,
    FiiDiiBundle,
    FiiDiiRow,
    PatternBundle,
    PatternKeypoint,
    SupportResistance,
    TechnicalSnapshot,
)
from services.stock_service import get_history_df
from utils.technical_indicators import calculate_technical_indicators


def get_technicals(ticker: str, lookback_days: int = 180) -> TechnicalSnapshot:
    end = date.today()
    start = end - timedelta(days=lookback_days)
    df = get_history_df(ticker, start, end)
    if df is None or df.empty:
        return TechnicalSnapshot(ticker=ticker)

    enriched = calculate_technical_indicators(df)
    if enriched.empty:
        return TechnicalSnapshot(ticker=ticker)

    last = enriched.iloc[-1]
    as_of = pd.to_datetime(enriched.index[-1]).date()
    return TechnicalSnapshot(
        ticker=ticker,
        as_of=as_of,
        last_close=_f(last.get("Close")),
        rsi_14=_f(last.get("RSI")),
        macd=_f(last.get("MACD")),
        macd_signal=_f(last.get("MACD_Signal")),
        macd_histogram=_f(last.get("MACD_Histogram")),
        atr_14=_f(last.get("ATR")),
        obv=_f(last.get("OBV")),
        ma_5=_f(last.get("MA5")),
        ma_20=_f(last.get("MA20")),
        ma_50=_f(last.get("MA50")),
        ma_200=_f(last.get("MA200")),
        volatility_20d=_f(last.get("Volatility_20D")),
        volume_ratio=_f(last.get("Volume_Ratio")),
        pivot_point=_f(last.get("Pivot_Point")),
        support_1=_f(last.get("S1")),
        resistance_1=_f(last.get("R1")),
        price_vs_ma20=_f(last.get("Price_vs_MA20")),
        price_vs_ma50=_f(last.get("Price_vs_MA50")),
    )


def get_patterns(ticker: str, lookback_days: int = 365) -> PatternBundle:
    from models.visual_analyst import PatternAnalyst

    end = date.today()
    start = end - timedelta(days=lookback_days)
    df = get_history_df(ticker, start, end)
    if df is None or df.empty:
        return PatternBundle(ticker=ticker)

    analyst = PatternAnalyst(order=5)
    try:
        result = analyst.analyze_all_patterns(df)
    except Exception:
        return PatternBundle(ticker=ticker)

    raw_patterns = result.get("patterns") or []
    detected: list[DetectedPattern] = []
    for p in raw_patterns:
        if not isinstance(p, dict):
            continue
        try:
            raw_kp = p.get("keypoints") or []
            keypoints = []
            for kp in raw_kp:
                try:
                    keypoints.append(PatternKeypoint(
                        date=str(kp["date"]),
                        price=float(kp["price"]),
                        label=str(kp["label"]),
                    ))
                except Exception:
                    pass
            detected.append(
                DetectedPattern(
                    name=str(p.get("Pattern") or "Unknown"),
                    type=_s(p.get("Type")),
                    confidence=max(0.0, min(100.0, float(p.get("Confidence", 0.0)))),
                    volume_confirmed=bool(p.get("volume_confirmed", False)),
                    timeframe_confluence=bool(p.get("timeframe_confluence", False)),
                    neckline=_f(p.get("Neckline")),
                    target=_f(p.get("Target")),
                    keypoints=keypoints,
                )
            )
        except (TypeError, ValueError):
            continue

    sr_raw = result.get("support_resistance") or {}
    sr = SupportResistance(
        nearest_support=_f(sr_raw.get("Nearest_Support")),
        nearest_resistance=_f(sr_raw.get("Nearest_Resistance")),
        strong_supports=_float_list(sr_raw.get("Support_Levels")),
        strong_resistances=_float_list(sr_raw.get("Resistance_Levels")),
    )

    trend = result.get("trend") or {}
    return PatternBundle(
        ticker=ticker,
        trend=_s(trend.get("Trend") if isinstance(trend, dict) else trend),
        market_character=_s(result.get("market_character")),
        hurst_exponent=_f(result.get("hurst_exponent")),
        bias=_s(result.get("overall_bias")),
        patterns=detected,
        support_resistance=sr,
    )


def get_fii_dii(lookback_days: int = 30) -> FiiDiiBundle:
    from data.fii_dii import extract_fii_dii_features, get_fii_dii_data

    end = date.today()
    start = end - timedelta(days=lookback_days)
    try:
        df = get_fii_dii_data(start_date=start, end_date=end)
    except Exception:
        df = None

    if df is None or df.empty:
        return FiiDiiBundle()

    rows: list[FiiDiiRow] = []
    for ts, r in df.iterrows():
        try:
            rows.append(
                FiiDiiRow(
                    date=pd.to_datetime(ts).date(),
                    fii_buy=_f(r.get("FII_Buy") or r.get("fii_buy")),
                    fii_sell=_f(r.get("FII_Sell") or r.get("fii_sell")),
                    fii_net=_f(r.get("FII_Net") or r.get("fii_net")),
                    dii_buy=_f(r.get("DII_Buy") or r.get("dii_buy")),
                    dii_sell=_f(r.get("DII_Sell") or r.get("dii_sell")),
                    dii_net=_f(r.get("DII_Net") or r.get("dii_net")),
                )
            )
        except (TypeError, ValueError):
            continue

    feats: dict = {}
    try:
        feats = extract_fii_dii_features(df, lookback=5) or {}
    except Exception:
        feats = {}

    return FiiDiiBundle(
        rows=rows,
        fii_net_5d=_f(feats.get("fii_net_5d") or feats.get("FII_Net_5D")),
        dii_net_5d=_f(feats.get("dii_net_5d") or feats.get("DII_Net_5D")),
        fii_net_streak=_opt_int(feats.get("fii_streak") or feats.get("FII_Streak")),
        dii_net_streak=_opt_int(feats.get("dii_streak") or feats.get("DII_Streak")),
    )


def _f(v) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)
        if f != f:
            return None
        return f
    except (TypeError, ValueError):
        return None


def _s(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _opt_int(v) -> int | None:
    try:
        if v is None:
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _float_list(v) -> list[float]:
    if not isinstance(v, (list, tuple)):
        return []
    out: list[float] = []
    for x in v:
        f = _f(x)
        if f is not None:
            out.append(f)
    return out
