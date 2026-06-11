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
    p25_price: float | None = None
    p75_price: float | None = None
    confidence_level: float = 0.90
    direction: Direction
    prob_up: float | None = Field(default=None, ge=0.0, le=1.0)


class TestSeriesPoint(BaseModel):
    """One row from the held-out test fold — actual vs predicted (returns + prices)."""

    date: date
    actual_return: float
    predicted_return: float
    actual_price: float | None = None
    predicted_price: float | None = None


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


class ConvictionPrecision(BaseModel):
    """Precision restricted to high-conviction calls (Phase 3B)."""

    threshold: float = 0.60
    n_high_bull: int = 0
    n_high_bear: int = 0
    bull_precision: float | None = None
    bear_precision: float | None = None
    conviction_rate: float = 0.0


class SwingSignal(BaseModel):
    """
    P4-1 — honest, conviction-gated multi-day (≈1-2 month) directional signal.

    Unlike `last_directional_prob` (next-day, ~50/50 noise), this targets a
    ~30-trading-day horizon where equity drift + a small genuine selective edge
    give a measurable, *out-of-sample-validated* hit-rate. Long/neutral only —
    confident single-name shorts are unreliable at this horizon (drift). Every
    number here is backed by the persisted walk-forward statistics, not an
    in-sample fit.
    """

    horizon_days: int
    signal: Literal["UP", "NEUTRAL"] = "NEUTRAL"
    prob_up: float = Field(ge=0.0, le=1.0, description="Calibrated P(up over horizon)")
    conviction: float = Field(default=0.0, ge=0.0, le=1.0)
    trained: bool = False
    # Honesty fields — measured out-of-sample, displayed verbatim in the UI.
    expected_hit_rate: float | None = None   # OOS precision of the fired UP bucket
    coverage: float | None = None            # fraction of history the signal fires
    base_rate: float | None = None           # naive always-up rate at this horizon
    tau_star: float | None = None            # conviction threshold for an UP call
    note: str = ""


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
    conformal_halfwidth: float | None = None  # ±INR price half-width (None if not fit)
    conformal_halfwidth_return: float | None = None  # raw log-return halfwidth (diagnostic)

    # Explainability + per-learner diagnostics
    shap_top_features: list[ShapFeature] = Field(default_factory=list)
    shap_method: str | None = None
    rmse_breakdown: ModelRmseBreakdown | None = None
    walkforward: WalkforwardSummary | None = None

    # A7.5 — "model was right X% of the time on this ticker over 30d"
    accuracy_30d: AccuracyBadge | None = None
    # A2.4/A2.5 — v2 ensemble late-blend diagnostics (None when v2 wasn't consulted).
    v2_blend: V2BlendInfo | None = None
    # Test-fold series for "Predicted vs Actual" / "Model Accuracy" charts.
    test_predictions: list[TestSeriesPoint] = Field(default_factory=list)
    # 3A — naive baselines (model must beat these)
    naive_up_rate: float | None = None
    naive_accuracy: float | None = None
    naive_brier: float | None = None
    # 3B — conviction-precision gate
    conviction_precision: ConvictionPrecision | None = None

    # P1-5 — ECE calibration gate flag
    calibration_gated: bool = False

    # P3-11 — trading suspension flag (set by LiveMonitor when 60d Sharpe < 0)
    trading_suspended: bool = False

    # P4-1 — honest 1-2 month swing-direction signal (None when model untrained).
    swing_signal: SwingSignal | None = None

