"""
Smoke tests for sentiment_service.

FinBERT is heavy (~500MB model, 30s cold load). These tests are marked `slow`
so local dev can `pytest -m 'not slow'` and CI can run the full suite.
No mocking — CLAUDE.md invariant #4.
"""

from __future__ import annotations

import pytest

from schemas.sentiment import SentimentScore
from services import sentiment_service


@pytest.mark.slow
def test_score_text_returns_typed_score():
    # Non-network: FinBERT loads from local HF cache (or downloads once, then cached)
    score = sentiment_service.score_text("Reliance beats earnings estimates sharply")
    assert isinstance(score, SentimentScore)
    assert score.label in ("positive", "negative", "neutral")
    assert 0.0 <= score.confidence <= 1.0


@pytest.mark.slow
def test_score_text_handles_empty():
    score = sentiment_service.score_text("")
    assert score.label == "neutral"
    assert score.confidence == 0.0
