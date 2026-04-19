"""Sentiment DTOs."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SentimentLabel = Literal["positive", "negative", "neutral"]


class SentimentScore(BaseModel):
    label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)


class HeadlineSentiment(BaseModel):
    title: str
    source: str
    url: str | None = None
    published_at: datetime | None = None
    category: str = "Other"
    score: SentimentScore


class SentimentAggregate(BaseModel):
    ticker: str
    window_start: datetime
    window_end: datetime
    n_headlines: int
    mean_score: float = Field(description="Signed: +1 bullish, -1 bearish, 0 neutral")
    pos_count: int = 0
    neg_count: int = 0
    neu_count: int = 0
    headlines: list[HeadlineSentiment] = Field(default_factory=list)
