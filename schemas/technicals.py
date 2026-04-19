"""Technical-indicator + chart-pattern + FII/DII DTOs."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class TechnicalSnapshot(BaseModel):
    """Latest-bar readings for the indicator panel."""

    ticker: str
    as_of: date | None = None
    last_close: float | None = None
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    atr_14: float | None = None
    obv: float | None = None
    ma_5: float | None = None
    ma_20: float | None = None
    ma_50: float | None = None
    ma_200: float | None = None
    volatility_20d: float | None = None
    volume_ratio: float | None = None
    pivot_point: float | None = None
    support_1: float | None = None
    resistance_1: float | None = None
    price_vs_ma20: float | None = None
    price_vs_ma50: float | None = None


class DetectedPattern(BaseModel):
    """One chart pattern picked up by `PatternAnalyst`."""

    name: str
    type: str | None = None  # "Bullish Reversal", "Bearish Continuation", ...
    confidence: float = Field(ge=0.0, le=100.0)
    volume_confirmed: bool = False
    timeframe_confluence: bool = False
    neckline: float | None = None
    target: float | None = None


class SupportResistance(BaseModel):
    nearest_support: float | None = None
    nearest_resistance: float | None = None
    strong_supports: list[float] = Field(default_factory=list)
    strong_resistances: list[float] = Field(default_factory=list)


class PatternBundle(BaseModel):
    """Complete chart-analysis payload for Tab 8."""

    ticker: str
    trend: str | None = None
    market_character: str | None = None  # Trending / Mean-Reverting / Random Walk
    hurst_exponent: float | None = None
    bias: str | None = None
    patterns: list[DetectedPattern] = Field(default_factory=list)
    support_resistance: SupportResistance | None = None


class FiiDiiRow(BaseModel):
    date: date
    fii_buy: float | None = None
    fii_sell: float | None = None
    fii_net: float | None = None
    dii_buy: float | None = None
    dii_sell: float | None = None
    dii_net: float | None = None


class FiiDiiBundle(BaseModel):
    """Tab 7 payload: recent flows + aggregates."""

    rows: list[FiiDiiRow] = Field(default_factory=list)
    fii_net_5d: float | None = None
    dii_net_5d: float | None = None
    fii_net_streak: int | None = None  # consecutive positive/negative days
    dii_net_streak: int | None = None
