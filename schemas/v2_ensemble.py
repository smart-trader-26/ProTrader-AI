"""
V2 sentiment ensemble DTOs.

The v2 model (from sentiment_analysis_v2.py, hosted on HuggingFace at
`EnteiTiger3/protrader-sentiment-v2`) is a 4-learner stack:
  LogReg + RandomForest + XGBoost + LightGBM → LogReg meta-stacker

Base features per headline: sentiment value (-1/0/+1) × category (6-way).
The ensemble outputs a single bullish probability [0, 1].
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class V2ModelBreakdown(BaseModel):
    """Per-base-learner bullish probability for diagnostics / UI."""

    logreg: float = Field(ge=0.0, le=1.0)
    random_forest: float = Field(ge=0.0, le=1.0)
    xgboost: float = Field(ge=0.0, le=1.0)
    lightgbm: float = Field(ge=0.0, le=1.0)


class V2EnsemblePrediction(BaseModel):
    """Output of `v2_ensemble_service.predict_v2`."""

    ticker: str
    made_at: datetime
    n_headlines: int
    top_category: str
    weighted_sentiment: float = Field(
        ge=-1.0, le=1.0,
        description="Confidence-weighted mean of per-headline sentiment in [-1, +1]",
    )
    category_counts: dict[str, int]
    prob_up: float = Field(ge=0.0, le=1.0, description="Stacked consensus probability")
    model_breakdown: V2ModelBreakdown
    stacker_available: bool = Field(
        description="False if stacker.pkl missing — prob_up is a weighted average fallback"
    )
    model_version: str = "v2-finbert-4lrn-stack"
