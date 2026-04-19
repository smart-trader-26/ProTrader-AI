"""
Stocks router (B1.2).

  GET /api/v1/stocks?q=                       → ticker search (substring, case-insensitive)
  GET /api/v1/stocks/{ticker}/ohlcv           → typed StockHistory
  GET /api/v1/stocks/{ticker}/fundamentals    → typed Fundamentals
  GET /api/v1/stocks/{ticker}/info            → raw info dict (parity with Streamlit panel)

Defaults: 6-month OHLCV window if start/end omitted.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Query

from schemas.stock import Fundamentals, StockHistory
from schemas.technicals import FiiDiiBundle, PatternBundle, TechnicalSnapshot
from services import stock_service, technicals_service

router = APIRouter(prefix="/stocks", tags=["stocks"])


@lru_cache(maxsize=1)
def _all_symbols() -> list[str]:
    """Load the Indian stock universe once per process. Local CSV — no network."""
    try:
        import pandas as pd

        from config.settings import DataConfig

        df = pd.read_csv(DataConfig.INDIAN_STOCKS_FILE, encoding="utf-8")
        df.columns = df.columns.str.strip()
        if "SYMBOL" in df.columns:
            return [str(s).strip() for s in df["SYMBOL"].dropna().tolist()]
    except Exception:
        pass
    from config.settings import DataConfig

    return list(DataConfig.DEFAULT_STOCKS)


@router.get("", summary="Search the Indian stock universe")
def search_stocks(
    q: str | None = Query(default=None, description="Substring filter, case-insensitive"),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict:
    symbols = _all_symbols()
    if q:
        needle = q.strip().upper()
        symbols = [s for s in symbols if needle in s.upper()]
    return {"count": len(symbols), "results": symbols[:limit]}


@router.get(
    "/{ticker}/ohlcv",
    response_model=StockHistory,
    summary="OHLCV history",
)
def get_ohlcv(
    ticker: str,
    start: date | None = Query(default=None),
    end: date | None = Query(default=None),
) -> StockHistory:
    end = end or date.today()
    start = start or (end - timedelta(days=180))
    if start >= end:
        raise HTTPException(status_code=422, detail="start must be < end")
    try:
        return stock_service.get_history(ticker, start, end)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {e}") from e


@router.get(
    "/{ticker}/fundamentals",
    response_model=Fundamentals,
    summary="Fundamental ratios",
)
def get_fundamentals(ticker: str) -> Fundamentals:
    try:
        return stock_service.get_fundamentals(ticker)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {e}") from e


@router.get("/{ticker}/info", summary="Raw info dict (Streamlit parity)")
def get_info(ticker: str) -> dict:
    try:
        return stock_service.get_info(ticker)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {e}") from e


@router.get(
    "/{ticker}/technicals",
    response_model=TechnicalSnapshot,
    summary="Latest-bar technical indicators",
)
def get_technicals(
    ticker: str,
    lookback_days: int = Query(default=180, ge=30, le=1825),
) -> TechnicalSnapshot:
    try:
        return technicals_service.get_technicals(ticker, lookback_days=lookback_days)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {e}") from e


@router.get(
    "/{ticker}/patterns",
    response_model=PatternBundle,
    summary="Detected chart patterns + support/resistance",
)
def get_patterns(
    ticker: str,
    lookback_days: int = Query(default=365, ge=90, le=1825),
) -> PatternBundle:
    try:
        return technicals_service.get_patterns(ticker, lookback_days=lookback_days)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {e}") from e


@router.get(
    "/fii-dii",
    response_model=FiiDiiBundle,
    summary="Recent FII / DII flows (market-wide, not per-ticker)",
)
def get_fii_dii(
    lookback_days: int = Query(default=30, ge=5, le=365),
) -> FiiDiiBundle:
    try:
        return technicals_service.get_fii_dii(lookback_days=lookback_days)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {e}") from e
