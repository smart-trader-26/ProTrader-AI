"""
Tests for the A2.4/A2.5 late-blend helper in `services.prediction_service`.

We exercise `_maybe_blend_v2` directly — the full `predict()` path needs
live market data, which is out of scope for unit tests.

Hard invariant: the blend must never break a prediction. Every failure
mode returns a `used=False` info record and the caller keeps the
stacker's probability unchanged.
"""

from __future__ import annotations

import types

from services import prediction_service as ps


def test_blend_disabled_returns_stacker_only(monkeypatch):
    # `use_v2_blend=False` must short-circuit before any import / fetch.
    out = ps._maybe_blend_v2("X.NS", stacker_prob=0.62, use_v2_blend=False, weight_override=None)
    assert out.used is False
    assert out.blended_prob == 0.62
    assert out.stacker_prob == 0.62
    assert out.v2_prob is None
    assert out.weight_v2 == 0.0


def test_blend_skipped_when_v2_not_configured(monkeypatch):
    from services import v2_ensemble_service as v2svc
    monkeypatch.setattr(v2svc, "is_configured", lambda: False)
    out = ps._maybe_blend_v2("X.NS", stacker_prob=0.40, use_v2_blend=None, weight_override=None)
    assert out.used is False
    assert out.blended_prob == 0.40


def test_blend_skipped_when_no_headlines(monkeypatch):
    from services import v2_ensemble_service as v2svc
    monkeypatch.setattr(v2svc, "is_configured", lambda: True)
    # Simulate a ticker with no fetched news.
    monkeypatch.setattr("data.news_sentiment.get_news", lambda t: [])
    monkeypatch.setattr("data.news_sentiment.filter_relevant_news", lambda raw, name: [])
    out = ps._maybe_blend_v2("X.NS", stacker_prob=0.55, use_v2_blend=None, weight_override=None)
    assert out.used is False
    assert out.blended_prob == 0.55


def test_blend_combines_stacker_and_v2_with_weight(monkeypatch):
    from services import v2_ensemble_service as v2svc
    monkeypatch.setattr(v2svc, "is_configured", lambda: True)

    fake_articles = [{"title": "Reliance beats earnings", "description": ""}]
    monkeypatch.setattr("data.news_sentiment.get_news", lambda t: fake_articles)
    monkeypatch.setattr(
        "data.news_sentiment.filter_relevant_news", lambda raw, name: fake_articles
    )

    # Stub the v2 predictor — we only care about the blend arithmetic here.
    def fake_predict_v2(ticker, headlines):
        ns = types.SimpleNamespace(
            prob_up=0.80,
            n_headlines=len(headlines),
            stacker_available=True,
        )
        return ns

    monkeypatch.setattr(v2svc, "predict_v2", fake_predict_v2)

    # weight_v2 = 0.25 → blended = 0.75 * 0.40 + 0.25 * 0.80 = 0.50
    out = ps._maybe_blend_v2("X.NS", stacker_prob=0.40, use_v2_blend=True, weight_override=0.25)
    assert out.used is True
    assert out.n_headlines == 1
    assert out.stacker_available is True
    assert out.stacker_prob == 0.40
    assert out.v2_prob == 0.80
    assert out.weight_v2 == 0.25
    assert abs(out.blended_prob - 0.50) < 1e-9


def test_blend_weight_is_clamped(monkeypatch):
    from services import v2_ensemble_service as v2svc
    monkeypatch.setattr(v2svc, "is_configured", lambda: True)
    arts = [{"title": "news", "description": ""}]
    monkeypatch.setattr("data.news_sentiment.get_news", lambda t: arts)
    monkeypatch.setattr("data.news_sentiment.filter_relevant_news", lambda raw, name: arts)
    monkeypatch.setattr(
        v2svc,
        "predict_v2",
        lambda ticker, h: types.SimpleNamespace(prob_up=0.9, n_headlines=1, stacker_available=False),
    )

    out_hi = ps._maybe_blend_v2("X.NS", 0.4, use_v2_blend=True, weight_override=5.0)
    assert out_hi.weight_v2 == 1.0  # clamped from above
    assert out_hi.blended_prob == 0.9  # fully v2

    out_lo = ps._maybe_blend_v2("X.NS", 0.4, use_v2_blend=True, weight_override=-1.0)
    assert out_lo.weight_v2 == 0.0  # clamped from below
    assert out_lo.blended_prob == 0.4  # fully stacker


def test_blend_swallows_v2_exceptions(monkeypatch):
    from services import v2_ensemble_service as v2svc
    monkeypatch.setattr(v2svc, "is_configured", lambda: True)
    arts = [{"title": "news", "description": ""}]
    monkeypatch.setattr("data.news_sentiment.get_news", lambda t: arts)
    monkeypatch.setattr("data.news_sentiment.filter_relevant_news", lambda raw, name: arts)

    def raiser(*a, **k):
        raise RuntimeError("HF 500")

    monkeypatch.setattr(v2svc, "predict_v2", raiser)

    out = ps._maybe_blend_v2("X.NS", 0.55, use_v2_blend=True, weight_override=0.3)
    assert out.used is False
    assert out.blended_prob == 0.55


