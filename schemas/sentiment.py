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


class SourceBreakdown(BaseModel):
    """Per-source aggregate: headline count + signed mean score for the agreement check."""

    source: str
    n_headlines: int
    mean_score: float
    label: SentimentLabel


class EventBreakdownItem(BaseModel):
    """One row of the news event-category pie chart."""

    category: str
    count: int
    share: float = Field(ge=0.0, le=1.0)


class SentimentAggregate(BaseModel):
    ticker: str
    window_start: datetime
    window_end: datetime
    n_headlines: int
    mean_score: float = Field(description="Signed: +1 bullish, -1 bearish, 0 neutral")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Mean per-headline confidence")
    overall_label: SentimentLabel = "neutral"
    pos_count: int = 0
    neg_count: int = 0
    neu_count: int = 0
    headlines: list[HeadlineSentiment] = Field(default_factory=list)
    source_breakdown: list[SourceBreakdown] = Field(default_factory=list)
    source_agreement: float | None = Field(
        default=None,
        description="0–1; 1 = all sources agree on direction, 0 = perfectly split",
    )
    source_agreement_label: str | None = None
    event_breakdown: list[EventBreakdownItem] = Field(default_factory=list)
