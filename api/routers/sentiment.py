"""
Sentiment router (B1.3).

  GET /api/v1/stocks/{ticker}/sentiment           → SentimentAggregate (FinBERT + 6-cat)
  POST /api/v1/sentiment/score                    → SentimentScore for arbitrary text
  GET /api/v1/stocks/{ticker}/sentiment/v2        → V2EnsemblePrediction (HF gated)
  GET /api/v1/stocks/{ticker}/sentiment/llm-alpha → LLM novelty×materiality alpha (P3-4)

The v2 endpoint requires `HF_TOKEN`; without it we 503 cleanly so the
frontend can fall back to the v1 sentiment block.
The llm-alpha endpoint requires `ANTHROPIC_API_KEY`; without it the response
includes `needs_manual_input=True` and a copyable `manual_prompt` string.
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


class ManualLlmAlphaRequest(BaseModel):
    manual_response: str = Field(min_length=2, max_length=20000)


class LlmAlphaResponse(BaseModel):
    ticker: str
    available: bool
    needs_manual_input: bool
    manual_prompt: str
    combined_signal: float
    mean_novelty: float
    mean_materiality: float
    mean_sentiment: float
    top_headline: str
    top_alpha: float
    headlines_scored: int
    used_fallback: bool
    scores: list[dict]


@router.get(
    "/stocks/{ticker}/sentiment/llm-alpha",
    response_model=LlmAlphaResponse,
    summary="LLM novelty×materiality alpha signal (P3-4)",
)
def ticker_llm_alpha(
    ticker: str,
    max_headlines: int = Query(default=10, ge=1, le=20),
) -> LlmAlphaResponse:
    """
    Scores recent headlines with Claude Haiku on novelty, materiality, and
    sentiment. Returns combined alpha = sentiment × novelty × materiality.
    When ANTHROPIC_API_KEY is not set, returns needs_manual_input=True with
    a copyable prompt for manual entry via claude.ai.
    """
    from data.news_sentiment import get_news as _get_news
    from services.llm_sentiment_alpha import LLMSentimentAlpha

    try:
        articles = _get_news(ticker) or []
        headlines = [a.get("title", "") for a in articles[:max_headlines] if a.get("title")]
        scorer = LLMSentimentAlpha()
        result = scorer.score_ticker(ticker, headlines)
        return _result_to_response(result)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM alpha failed: {e}") from e


def _result_to_response(result) -> LlmAlphaResponse:
    return LlmAlphaResponse(
        ticker=result.ticker,
        available=result.available,
        needs_manual_input=result.needs_manual_input,
        manual_prompt=result.manual_prompt,
        combined_signal=result.combined_signal,
        mean_novelty=result.mean_novelty,
        mean_materiality=result.mean_materiality,
        mean_sentiment=result.mean_sentiment,
        top_headline=result.top_headline,
        top_alpha=result.top_alpha,
        headlines_scored=result.headlines_scored,
        used_fallback=result.used_fallback,
        scores=[
            {
                "headline": s.headline,
                "novelty": s.novelty,
                "materiality": s.materiality,
                "sentiment": s.sentiment,
                "alpha_signal": s.alpha_signal,
            }
            for s in result.individual_scores
        ],
    )


@router.post(
    "/stocks/{ticker}/sentiment/llm-alpha/manual",
    response_model=LlmAlphaResponse,
    summary="Parse user-pasted Claude response for LLM alpha (P3-4 manual mode)",
)
def ticker_llm_alpha_manual(ticker: str, body: ManualLlmAlphaRequest) -> LlmAlphaResponse:
    """
    Accepts the JSON response the user pasted from claude.ai and converts it
    into a scored TickerSentimentResult. Used when ANTHROPIC_API_KEY is absent.
    """
    from data.news_sentiment import get_news as _get_news
    from services.llm_sentiment_alpha import LLMSentimentAlpha

    try:
        articles = _get_news(ticker) or []
        headlines = [a.get("title", "") for a in articles[:10] if a.get("title")]
        scorer = LLMSentimentAlpha()
        result = scorer.score_ticker_from_manual_response(ticker, headlines, body.manual_response)
        return _result_to_response(result)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Manual parse failed: {e}") from e
