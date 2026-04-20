"""
Sentiment service — headlines → classified, scored, aggregated.

Wraps `data.news_sentiment` (FinBERT + keyword overrides) and returns
typed DTOs from `schemas.sentiment`. All FinBERT calls flow through the
lazy pipeline in `news_sentiment._get_sentiment_pipeline`, which enforces
`framework="pt"` (hard invariant #1).
"""

from __future__ import annotations

from collections import defaultdict
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
from schemas.sentiment import (
    EventBreakdownItem,
    HeadlineSentiment,
    SentimentAggregate,
    SentimentScore,
    SourceBreakdown,
)


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
    conf_total = 0.0

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
        conf_total += score.confidence
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
    mean_conf = (conf_total / n) if n else 0.0
    overall_label = "positive" if mean > 0.1 else "negative" if mean < -0.1 else "neutral"

    breakdown = _per_source_breakdown(scored)
    agreement, agree_label = _agreement_score(breakdown)
    events = _event_breakdown(scored)

    now = datetime.now(UTC)
    return SentimentAggregate(
        ticker=ticker,
        window_start=now,
        window_end=now,
        n_headlines=n,
        mean_score=mean,
        confidence=mean_conf,
        overall_label=overall_label,
        pos_count=pos,
        neg_count=neg,
        neu_count=neu,
        headlines=scored,
        source_breakdown=breakdown,
        source_agreement=agreement,
        source_agreement_label=agree_label,
        event_breakdown=events,
    )


def _per_source_breakdown(headlines: list[HeadlineSentiment]) -> list[SourceBreakdown]:
    """Group headlines by source, compute signed mean per source."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for h in headlines:
        s = h.source or "unknown"
        if h.score.label == "positive":
            buckets[s].append(h.score.confidence)
        elif h.score.label == "negative":
            buckets[s].append(-h.score.confidence)
        else:
            buckets[s].append(0.0)
    out: list[SourceBreakdown] = []
    for src, vals in buckets.items():
        if not vals:
            continue
        m = sum(vals) / len(vals)
        lbl = "positive" if m > 0.1 else "negative" if m < -0.1 else "neutral"
        out.append(
            SourceBreakdown(source=src, n_headlines=len(vals), mean_score=m, label=lbl)
        )
    out.sort(key=lambda x: x.n_headlines, reverse=True)
    return out


def _agreement_score(breakdown: list[SourceBreakdown]) -> tuple[float | None, str | None]:
    """1.0 = all sources agree on direction, 0.0 = perfectly split.

    Score is the maximum share of sources sharing the dominant label, mapped
    onto [0, 1]. With only one source the agreement is trivially 1.0 — we
    return None in that case so the UI can label it "single source".
    """
    if not breakdown:
        return None, None
    if len(breakdown) == 1:
        return None, "Single source"
    counts: dict[str, int] = defaultdict(int)
    for s in breakdown:
        counts[s.label] += 1
    top = max(counts.values())
    score = top / len(breakdown)
    if score >= 0.8:
        label = "Sources Agree"
    elif score >= 0.6:
        label = "Mostly Agree"
    elif score >= 0.4:
        label = "Split View"
    else:
        label = "Sources Disagree"
    return float(score), label


def _event_breakdown(headlines: list[HeadlineSentiment]) -> list[EventBreakdownItem]:
    """Tally per-category counts + share for the news-event pie chart."""
    if not headlines:
        return []
    counts: dict[str, int] = defaultdict(int)
    for h in headlines:
        counts[(h.category or "Other").lower()] += 1
    total = sum(counts.values()) or 1
    out = [
        EventBreakdownItem(category=cat, count=n, share=n / total)
        for cat, n in counts.items()
    ]
    out.sort(key=lambda x: x.count, reverse=True)
    return out


def _parse_dt(val) -> datetime | None:
    if not val:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
