"""
Pattern / technical-indicator service.

Thin facade over `utils.technical_indicators` so both Streamlit and
FastAPI consume the same surface. Returns enriched DataFrames for now —
once a `Pattern` DTO is needed (e.g. for the Next.js `/patterns` endpoint),
promote it into `schemas/`.
"""

from __future__ import annotations

import pandas as pd

from utils.technical_indicators import calculate_technical_indicators


def enrich_with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` augmented with RSI, MACD, Bollinger bands, etc."""
    if df is None or df.empty:
        return df
    return calculate_technical_indicators(df)
