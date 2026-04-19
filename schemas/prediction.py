"""Prediction + calibration DTOs."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

Direction = Literal["up", "down", "flat"]


class PredictionPoint(BaseModel):
    target_date: date
    pred_price: float
    ci_low: float | None = None
    ci_high: float | None = None
    confidence_level: float = 0.90
    direction: Direction
    prob_up: float = Field(ge=0.0, le=1.0)


class CalibrationReport(BaseModel):
    """Holdout calibration diagnostics for the directional classifier."""

    n_samples: int
    ece: float = Field(ge=0.0, le=1.0, description="Expected Calibration Error")
    brier_score: float = Field(ge=0.0, le=1.0)
    bin_edges: list[float]
    bin_predicted: list[float]
    bin_actual: list[float]
    bin_counts: list[int]


class AccuracyBadge(BaseModel):
    """Inline historical accuracy attached to every predict() response (A7.5)."""

    window_days: int
    n_resolved: int
    directional_accuracy: float | None = None  # 0..1
    brier_score: float | None = None


class V2BlendInfo(BaseModel):
    """A2.4/A2.5 late-blend diagnostics — which probabilities got combined and how."""

    stacker_prob: float = Field(ge=0.0, le=1.0)
    v2_prob: float | None = Field(default=None, ge=0.0, le=1.0)
    blended_prob: float = Field(ge=0.0, le=1.0)
    weight_v2: float = Field(ge=0.0, le=1.0, description="Convex weight applied to v2")
    n_headlines: int = 0
    stacker_available: bool = False
    used: bool = False


class ShapFeature(BaseModel):
    """Single (feature, importance) pair — sign carries bull/bear direction."""

    feature: str
    importance: float


class ThresholdTuning(BaseModel):
    """Per-ticker decision threshold picked on holdout (A1.5)."""

    tau_star: float = Field(ge=0.0, le=1.0, description="Youden-J optimal threshold")
    auc: float | None = None
    accuracy_at_tau: float | None = None


class ModelRmseBreakdown(BaseModel):
    """Test-fold RMSE per base learner + the stacked meta."""

    xgb: float | None = None
    lgbm: float | None = None
    catboost: float | None = None
    rnn: float | None = None
    stacked: float | None = None


class WalkforwardSummary(BaseModel):
    """Aggregate directional accuracy across walk-forward folds."""

    accuracy: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    n_windows: int | None = None


class PredictionBundle(BaseModel):
    ticker: str
    made_at: datetime
    model_version: str
    horizon_days: int
    points: list[PredictionPoint]

    # Anchor + aggregate metrics (set on every response)
    anchor_price: float | None = None

    # Regime / dynamics
    regime: str | None = None
    regime_detail: str | None = None
    hurst_exponent: float | None = None

    # Probability aggregates
    avg_directional_prob: float | None = None
    last_directional_prob: float | None = None

    # Calibration + threshold tuning (A1.2 / A1.5)
    calibration: CalibrationReport | None = None
    threshold_tuning: ThresholdTuning | None = None

    # Interval metadata (A6)
    confidence_level: float = 0.90
    conformal_halfwidth: float | None = None  # ± log-return half-width (None if not fit)

    # Explainability + per-learner diagnostics
    shap_top_features: list[ShapFeature] = Field(default_factory=list)
    shap_method: str | None = None
    rmse_breakdown: ModelRmseBreakdown | None = None
    walkforward: WalkforwardSummary | None = None

    # A7.5 — "model was right X% of the time on this ticker over 30d"
    accuracy_30d: AccuracyBadge | None = None
    # A2.4/A2.5 — v2 ensemble late-blend diagnostics (None when v2 wasn't consulted).
    v2_blend: V2BlendInfo | None = None
