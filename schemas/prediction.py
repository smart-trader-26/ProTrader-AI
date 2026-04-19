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


class PredictionBundle(BaseModel):
    ticker: str
    made_at: datetime
    model_version: str
    horizon_days: int
    points: list[PredictionPoint]
    regime: str | None = None
    hurst_exponent: float | None = None
    avg_directional_prob: float | None = None
    calibration: CalibrationReport | None = None
    # A6.3 — interval metadata carried on the bundle (points already hold the band)
    confidence_level: float = 0.90
    conformal_halfwidth: float | None = None  # ± log-return half-width (None if not fit)
    # A7.5 — "model was right X% of the time on this ticker over 30d"
    accuracy_30d: AccuracyBadge | None = None
