"""
Prediction service — thin seam over `models.hybrid_model`.

Exposes `predict()` returning a typed `PredictionBundle`. The notebook /
FastAPI contract is:

    from services.prediction_service import predict
    bundle = predict("RELIANCE.NS", horizon_days=10)

Heavy lifting (XGB + LGBM + CatBoost + GRU + Ridge meta-stacker) still
lives in `models/hybrid_model.py` — this file is deliberately thin so the
refactor stays reversible and the Streamlit path keeps working unchanged.

A7 wiring: every call appends to the SQLite ledger and attaches the
rolling 30-day accuracy badge to the response (`bundle.accuracy_30d`).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd

from models.hybrid_model import create_hybrid_model, hybrid_predict_prices
from schemas.prediction import (
    AccuracyBadge,
    Direction,
    PredictionBundle,
    PredictionPoint,
)
from services import ledger_service
from services.stock_service import get_history_df

MODEL_VERSION = "hybrid-v1"


def predict(
    ticker: str,
    horizon_days: int = 10,
    start: date | None = None,
    end: date | None = None,
    sentiment_features: dict | None = None,
    fii_dii_data: pd.DataFrame | None = None,
    vix_data: pd.DataFrame | None = None,
    multi_source_sentiment: dict | None = None,
    enable_automl: bool = False,
    n_paths: int = 200,
    log_to_ledger: bool = True,
) -> PredictionBundle:
    """Fit the hybrid model on recent history and return forecast points."""
    end = end or date.today()
    start = start or (end - timedelta(days=365 * 5))
    df = get_history_df(ticker, start, end)
    if df is None or df.empty or len(df) < 60:
        raise ValueError(
            f"Insufficient history for {ticker}: {0 if df is None else len(df)} bars"
        )

    df_proc, _results_df, models, scaler, features, metrics = create_hybrid_model(
        df,
        sentiment_features or {},
        fii_dii_data=fii_dii_data,
        vix_data=vix_data,
        multi_source_sentiment=multi_source_sentiment,
        enable_automl=enable_automl,
    )

    future = hybrid_predict_prices(
        models,
        scaler,
        df_proc.iloc[-60:],
        features,
        days=horizon_days,
        df_proc_full=df_proc,
        directional_prob=metrics.get("last_directional_prob", 50.0) / 100.0,
        regime=metrics.get("regime", "normal"),
        n_paths=n_paths,
    )

    last_close = float(df["Close"].iloc[-1])
    prob_up = float(metrics.get("last_directional_prob", 50.0)) / 100.0
    points = _to_points(future, last_close, prob_up)

    # A6 — conformal half-width, if the model computed it during training.
    conformal_hw = _opt(metrics.get("conformal_halfwidth"))

    bundle = PredictionBundle(
        ticker=ticker,
        made_at=datetime.now(UTC),
        model_version=MODEL_VERSION,
        horizon_days=horizon_days,
        points=points,
        regime=metrics.get("regime_detail"),
        hurst_exponent=_opt(metrics.get("hurst_exponent")),
        avg_directional_prob=_opt(metrics.get("avg_directional_prob")),
        confidence_level=0.90,
        conformal_halfwidth=conformal_hw,
    )

    # A7.2 — append to the ledger so actuals can be resolved later.
    if log_to_ledger:
        try:
            ledger_service.log_prediction(bundle, anchor_price=last_close)
        except Exception:
            # Never let a ledger failure break a prediction — they're independent.
            pass

    # A7.5 — attach the rolling 30-day accuracy for this ticker, if resolved
    # rows exist. For a cold ticker this is `n_resolved=0` and the UI hides it.
    try:
        win = ledger_service.accuracy_window(ticker, days=30)
        bundle.accuracy_30d = AccuracyBadge(
            window_days=win.window_days,
            n_resolved=win.n_resolved,
            directional_accuracy=win.directional_accuracy,
            brier_score=win.brier_score,
        )
    except Exception:
        pass

    return bundle


def _to_points(future, last_close: float, prob_up: float) -> list[PredictionPoint]:
    if future is None or len(future) == 0:
        return []

    if isinstance(future, pd.DataFrame):
        if "Predicted Price" in future.columns:
            prices = future["Predicted Price"].tolist()
        elif "pred_price" in future.columns:
            prices = future["pred_price"].tolist()
        else:
            prices = future.iloc[:, 0].tolist()
        dates = [
            (d.date() if hasattr(d, "date") else d) for d in future.index
        ]
        ci_lo_col = "P5" if "P5" in future.columns else "ci_low"
        ci_hi_col = "P95" if "P95" in future.columns else "ci_high"
        ci_lo = future[ci_lo_col].tolist() if ci_lo_col in future.columns else [None] * len(prices)
        ci_hi = future[ci_hi_col].tolist() if ci_hi_col in future.columns else [None] * len(prices)
    else:
        prices = list(future)
        today = date.today()
        dates = [today + timedelta(days=i + 1) for i in range(len(prices))]
        ci_lo = [None] * len(prices)
        ci_hi = [None] * len(prices)

    out: list[PredictionPoint] = []
    prev = last_close
    for i, (d, p) in enumerate(zip(dates, prices)):
        direction: Direction = "flat"
        if p > prev * 1.001:
            direction = "up"
        elif p < prev * 0.999:
            direction = "down"
        out.append(
            PredictionPoint(
                target_date=d if isinstance(d, date) else pd.to_datetime(d).date(),
                pred_price=float(p),
                ci_low=_opt(ci_lo[i]),
                ci_high=_opt(ci_hi[i]),
                direction=direction,
                prob_up=max(0.0, min(1.0, prob_up)),
            )
        )
        prev = p
    return out


def _opt(v):
    try:
        if v is None:
            return None
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None
