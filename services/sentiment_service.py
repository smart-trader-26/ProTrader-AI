"""
Sentiment service — headlines → classified, scored, aggregated.

Wraps `data.news_sentiment` (FinBERT + keyword overrides) and returns
typed DTOs from `schemas.sentiment`. All FinBERT calls flow through the
lazy pipeline in `news_sentiment._get_sentiment_pipeline`, which enforces
`framework="pt"` (hard invariant #1).
"""

from __future__ import annotations

from datetime import UTC, datetime

from data.news_sentiment import (
    analyze_sentiment as _analyze_raw,
)
from data.news_sentiment import (
    categorize_headline as _categorize,
)
from data.news_sentiment import (
    filter_relevant_news as _filter,
)
from data.news_sentiment import (
    get_news as _get_news_raw,
)
from schemas.sentiment import HeadlineSentiment, SentimentAggregate, SentimentScore


def score_text(text: str) -> SentimentScore:
    label, conf = _analyze_raw(text or "")
    return SentimentScore(label=label, confidence=float(conf))


def analyze_ticker(
    ticker: str,
    company_name: str | None = None,
    max_headlines: int = 25,
) -> SentimentAggregate:
    """Fetch recent headlines and return an aggregated sentiment snapshot."""
    raw = _get_news_raw(ticker) or []
    relevant = _filter(raw, company_name or ticker)[:max_headlines]

    scored: list[HeadlineSentiment] = []
    pos = neg = neu = 0
    signed_total = 0.0

    for art in relevant:
        title = art.get("title") or ""
        if not title:
            continue
        score = score_text(title)
        pub = _parse_dt(art.get("publishedAt"))
        scored.append(
            HeadlineSentiment(
                title=title,
                source=(art.get("source") or {}).get("name", "unknown"),
                url=art.get("url"),
                published_at=pub,
                category=_categorize(title),
                score=score,
            )
        )
        if score.label == "positive":
            pos += 1
            signed_total += score.confidence
        elif score.label == "negative":
            neg += 1
            signed_total -= score.confidence
        else:
            neu += 1

    n = len(scored)
    mean = (signed_total / n) if n else 0.0
    now = datetime.now(UTC)
    return SentimentAggregate(
        ticker=ticker,
        window_start=now,
        window_end=now,
        n_headlines=n,
        mean_score=mean,
        pos_count=pos,
        neg_count=neg,
        neu_count=neu,
        headlines=scored,
    )


def _parse_dt(val) -> datetime | None:
    if not val:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
