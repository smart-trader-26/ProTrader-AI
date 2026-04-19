"""
Sentiment router (B1.3).

  GET /api/v1/stocks/{ticker}/sentiment       → SentimentAggregate (FinBERT + 6-cat)
  POST /api/v1/sentiment/score                → SentimentScore for arbitrary text
  GET /api/v1/stocks/{ticker}/sentiment/v2    → V2EnsemblePrediction (HF gated)

The v2 endpoint requires `HF_TOKEN`; without it we 503 cleanly so the
frontend can fall back to the v1 sentiment block.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from config.settings import HF_TOKEN
from schemas.sentiment import SentimentAggregate, SentimentScore
from schemas.v2_ensemble import V2EnsemblePrediction
from services import sentiment_service

router = APIRouter(tags=["sentiment"])


class ScoreRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@router.post("/sentiment/score", response_model=SentimentScore, summary="Score arbitrary text")
def score_text(body: ScoreRequest) -> SentimentScore:
    return sentiment_service.score_text(body.text)


@router.get(
    "/stocks/{ticker}/sentiment",
    response_model=SentimentAggregate,
    summary="Headline sentiment for a ticker",
)
def ticker_sentiment(
    ticker: str,
    company_name: str | None = Query(default=None),
    max_headlines: int = Query(default=25, ge=1, le=100),
) -> SentimentAggregate:
    try:
        return sentiment_service.analyze_ticker(
            ticker, company_name=company_name, max_headlines=max_headlines
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"sentiment fetch failed: {e}") from e


@router.get(
    "/stocks/{ticker}/sentiment/v2",
    response_model=V2EnsemblePrediction,
    summary="V2 ensemble prediction (gated behind HF_TOKEN)",
)
def ticker_sentiment_v2(
    ticker: str,
    max_headlines: int = Query(default=25, ge=1, le=100),
) -> V2EnsemblePrediction:
    if not HF_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="v2 ensemble requires HF_TOKEN — see docs/API_KEYS.md",
        )
    # Late import: pulling joblib + huggingface_hub at module load slows the
    # whole API boot. Import inside the handler so cold-start stays fast.
    from data.news_sentiment import get_news as _get_news
    from services import v2_ensemble_service

    try:
        articles = _get_news(ticker) or []
        headlines = [{"title": a.get("title", "")} for a in articles[:max_headlines] if a.get("title")]
        if not headlines:
            raise HTTPException(status_code=404, detail="no headlines available for v2 ensemble")
        return v2_ensemble_service.predict_v2(ticker, headlines)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"v2 prediction failed: {e}") from e
