"""
Smoke tests for v2_ensemble_service.

Two behaviours we can verify cheaply without downloading 340 MB of model:

1. `is_configured()` reflects HF_TOKEN presence.
2. `_build_feature_row()` produces the exact 12-column schema in training order
   (regression guard — if these columns ever drift out of order, every base
   learner silently miscomputes).

The full `predict_v2()` path is marked `network + slow` — it pulls ~340 MB from
HuggingFace on first call and spins up a PyTorch FinBERT pipeline.
"""

from __future__ import annotations

import pytest

from services import v2_ensemble_service as v2


def test_is_configured_reflects_env(monkeypatch):
    # Patch at the module level (settings already resolved at import time)
    monkeypatch.setattr(v2, "HF_TOKEN", "", raising=False)
    assert v2.is_configured() is False
    monkeypatch.setattr(v2, "HF_TOKEN", "hf_fake", raising=False)
    monkeypatch.setattr(v2, "HF_REPO_ID", "user/repo", raising=False)
    assert v2.is_configured() is True


def test_feature_row_has_exact_training_columns():
    row = v2._build_feature_row("Market Action", 0.42)
    assert list(row.columns) == v2.ALL_FEATURES
    assert row.shape == (1, 12)
    # Only the Market Action cells should be non-zero.
    assert row["Sentiment_Market Action"].iloc[0] == pytest.approx(0.42)
    assert row["Count_Market Action"].iloc[0] == 1
    for cat in v2.VALID_CATEGORIES:
        if cat == "Market Action":
            continue
        assert row[f"Sentiment_{cat}"].iloc[0] == 0.0
        assert row[f"Count_{cat}"].iloc[0] == 0


def test_feature_row_dtype_is_float32():
    row = v2._build_feature_row("Other", -0.3)
    assert str(row.dtypes.iloc[0]) == "float32"


@pytest.mark.network
@pytest.mark.slow
def test_predict_v2_end_to_end():
    if not v2.is_configured():
        pytest.skip("HF_TOKEN not set — skipping end-to-end v2 ensemble test")
    headlines = [
        {"title": "Reliance beats Q2 earnings, profit jumps 18%"},
        {"title": "Goldman Sachs upgrades Reliance to Buy, target raised"},
    ]
    out = v2.predict_v2("RELIANCE.NS", headlines)
    assert 0.0 <= out.prob_up <= 1.0
    assert out.n_headlines == 2
    assert out.top_category in v2.VALID_CATEGORIES
    assert -1.0 <= out.weighted_sentiment <= 1.0
    # Base learners should all have produced something.
    assert 0.0 <= out.model_breakdown.logreg <= 1.0
    assert 0.0 <= out.model_breakdown.xgboost <= 1.0


def test_predict_v2_rejects_empty_headlines():
    with pytest.raises(ValueError):
        v2.predict_v2("X", [])
