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

import logging
from datetime import UTC, date, datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

from models.hybrid_model import create_hybrid_model, hybrid_predict_prices
from schemas.prediction import (
    AccuracyBadge,
    CalibrationReport,
    Direction,
    ModelRmseBreakdown,
    PredictionBundle,
    PredictionPoint,
    ShapFeature,
    TestSeriesPoint,
    ThresholdTuning,
    V2BlendInfo,
    WalkforwardSummary,
)
from services import ledger_service
from services.stock_service import get_history_df

MODEL_VERSION = "hybrid-v1-blend"

# A2.4/A2.5 — default convex weight applied to v2's prob_up in the late blend.
# Stacker gets (1 - weight). Overridable via V2_BLEND_WEIGHT env / call arg.
# Kept conservative since the weight is fixed, not learned — the ledger will
# either validate or kill this default over a 30-day window.
_DEFAULT_V2_BLEND_WEIGHT = 0.3


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
    use_v2_blend: bool | None = None,
    v2_blend_weight: float | None = None,
) -> PredictionBundle:
    """Fit the hybrid model on recent history and return forecast points."""
    end = end or date.today()
    start = start or (end - timedelta(days=365 * 5))
    df = get_history_df(ticker, start, end)
    if df is None or df.empty or len(df) < 60:
        raise ValueError(
            f"Insufficient history for {ticker}: {0 if df is None else len(df)} bars"
        )

    df_proc, results_df, models, scaler, features, metrics = create_hybrid_model(
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
    stacker_prob = float(metrics.get("last_directional_prob", 50.0)) / 100.0

    # Dampen sentiment when patterns are strongly bearish so structural evidence
    # isn't overridden by a positive news cycle.
    pattern_conviction = _compute_pattern_conviction(df)
    effective_v2_weight = _adjust_v2_weight_for_patterns(
        v2_blend_weight, pattern_conviction
    )
    blend_info = _maybe_blend_v2(ticker, stacker_prob, use_v2_blend, effective_v2_weight)
    prob_up = blend_info.blended_prob
    points = _to_points(future, last_close, prob_up)

    # A6 — conformal half-width (return space → price space).
    # The model computes halfwidth as |y_true - y_pred| quantile in log-return
    # units. Convert to an INR price band: price ± price * exp(hw) - price.
    # This makes the UI stat readable as an actual rupee range, not a 0.01-ish
    # decimal fraction that displays as zero.
    conformal_hw_ret = _opt(metrics.get("conformal_halfwidth"))
    conformal_hw = None
    if conformal_hw_ret is not None:
        import math
        conformal_hw = last_close * (math.exp(conformal_hw_ret) - 1.0)

    bundle = PredictionBundle(
        ticker=ticker,
        made_at=datetime.now(UTC),
        model_version=MODEL_VERSION,
        horizon_days=horizon_days,
        points=points,
        anchor_price=last_close,
        regime=metrics.get("regime"),
        regime_detail=metrics.get("regime_detail"),
        hurst_exponent=_opt(metrics.get("hurst_exponent")),
        avg_directional_prob=_opt(metrics.get("avg_directional_prob")),
        last_directional_prob=_opt(metrics.get("last_directional_prob")),
        calibration=_to_calibration(metrics.get("calibration")),
        threshold_tuning=_to_threshold_tuning(metrics.get("threshold_tuning")),
        confidence_level=0.90,
        conformal_halfwidth=conformal_hw,
        shap_top_features=_to_shap(metrics.get("shap_top_features")),
        shap_method=metrics.get("shap_method"),
        rmse_breakdown=_to_rmse(metrics),
        walkforward=_to_walkforward(metrics),
        v2_blend=blend_info if blend_info.used else None,
        test_predictions=_to_test_series(results_df, df),
        conformal_halfwidth_return=conformal_hw_ret,  # kept for diagnostics
        naive_up_rate=_opt(metrics.get("naive_up_rate")),
        naive_accuracy=_opt(metrics.get("naive_accuracy")),
        naive_brier=_opt(metrics.get("naive_brier")),
        conviction_precision=_to_conviction(metrics.get("conviction_precision")),
    )

    # A7.2 — append to the ledger so actuals can be resolved later.
    if log_to_ledger:
        try:
            ledger_service.log_prediction(bundle, anchor_price=last_close)
        except Exception as _exc:  # Bug O: log instead of silently swallow
            logger.warning("Ledger write failed for %s: %s: %s",
                           ticker, type(_exc).__name__, _exc)

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
        p25 = future["P25"].tolist() if "P25" in future.columns else [None] * len(prices)
        p75 = future["P75"].tolist() if "P75" in future.columns else [None] * len(prices)
        # Per-day P(up) from MC paths: fraction of paths ending above anchor.
        # Day-0 overridden with the blended calibrated prob (more precise);
        # days 1-N use the MC-derived value which varies with the fan width.
        mc_prob_up = future["Prob_Up"].tolist() if "Prob_Up" in future.columns else [None] * len(prices)
    else:
        prices = list(future)
        today = date.today()
        dates = [today + timedelta(days=i + 1) for i in range(len(prices))]
        ci_lo = [None] * len(prices)
        ci_hi = [None] * len(prices)
        p25 = [None] * len(prices)
        p75 = [None] * len(prices)
        mc_prob_up = [None] * len(prices)

    out: list[PredictionPoint] = []
    for i, (d, p) in enumerate(zip(dates, prices)):
        # Compare every day's predicted price against the ORIGINAL anchor (last
        # known close), not against the rolling previous predicted price. The
        # ledger resolves `hit` using `actual vs anchor_price`, so direction must
        # use the same baseline. The old `prev = p` caused day-2+ directions to
        # be computed relative to a drifting predicted price, systematically
        # mis-marking hits in the accuracy ledger.
        direction: Direction = "flat"
        if p > last_close * 1.001:
            direction = "up"
        elif p < last_close * 0.999:
            direction = "down"

        # Day-0: use the blended calibrated probability (most precise).
        # Days 1-N: use MC-path probability — varies per horizon so the UI
        #   shows a genuine spread rather than one number repeated ten times.
        #   NOT logged to the ledger for calibration (only day-0 is a real call).
        if i == 0:
            day_prob: float | None = max(0.0, min(1.0, prob_up))
        else:
            raw_mc = mc_prob_up[i]
            day_prob = float(max(0.0, min(1.0, raw_mc))) if raw_mc is not None else None

        out.append(
            PredictionPoint(
                target_date=d if isinstance(d, date) else pd.to_datetime(d).date(),
                pred_price=float(p),
                ci_low=_opt(ci_lo[i]),
                ci_high=_opt(ci_hi[i]),
                p25_price=_opt(p25[i]),
                p75_price=_opt(p75[i]),
                direction=direction,
                prob_up=day_prob,
            )
        )
    return out


def _to_test_series(results_df, df) -> list[TestSeriesPoint]:
    """Pull (date, actual_return, predicted_return, actual_price, predicted_price)
    rows from the held-out test fold so the UI can render the Streamlit-style
    'Predicted vs Actual Returns' and 'Model Accuracy' charts.

    Anchors predicted price to the previous *actual* close (same approach as
    `ui/charts.create_accuracy_comparison_chart`) — keeps the dotted line from
    drifting on cumulative-return error.
    """
    if results_df is None or len(results_df) == 0:
        return []
    try:
        import math

        actual_ret = results_df.get("Actual_Return")
        pred_ret = results_df.get("Predicted_Return")
        if actual_ret is None or pred_ret is None:
            return []

        # Resolve actual close prices for each test date from the source df.
        close_series = df["Close"] if "Close" in df.columns else None
        out: list[TestSeriesPoint] = []
        for i, idx in enumerate(results_df.index):
            d_obj = idx.date() if hasattr(idx, "date") else idx
            ar = float(actual_ret.iloc[i])
            pr = float(pred_ret.iloc[i])
            actual_price: float | None = None
            predicted_price: float | None = None
            if close_series is not None and idx in close_series.index:
                actual_price = float(close_series.loc[idx])
                # Anchor on previous actual close to avoid cumulative drift.
                pos = close_series.index.get_loc(idx)
                if isinstance(pos, int) and pos > 0:
                    prev_actual = float(close_series.iloc[pos - 1])
                    predicted_price = prev_actual * math.exp(pr)
            out.append(
                TestSeriesPoint(
                    date=d_obj if isinstance(d_obj, date) else pd.to_datetime(d_obj).date(),
                    actual_return=ar,
                    predicted_return=pr,
                    actual_price=actual_price,
                    predicted_price=predicted_price,
                )
            )
        return out
    except Exception:
        return []


def _maybe_blend_v2(
    ticker: str,
    stacker_prob: float,
    use_v2_blend: bool | None,
    weight_override: float | None,
) -> V2BlendInfo:
    """
    Late-blend the Ridge stacker's prob_up with v2 ensemble's prob_up.

    V2 blend runs automatically as part of every prediction when headlines are
    available. It does NOT require a separate user action.

    Skips cleanly — and returns a `used=False` info record — when:
      • caller passed `use_v2_blend=False`
      • v2 service module can't be imported
      • headline fetch fails or returns nothing
      • v2 model download / inference raises for any reason

    Any failure falls back to the stacker-only probability, so this path can
    never break a prediction.
    """
    no_blend = V2BlendInfo(
        stacker_prob=stacker_prob,
        blended_prob=stacker_prob,
        weight_v2=0.0,
        used=False,
    )
    if use_v2_blend is False:
        return no_blend

    try:
        from services import v2_ensemble_service as v2svc
    except Exception:
        return no_blend

    # v2 is now always attempted — no HF_TOKEN gate. The load() call will
    # fail gracefully if the model isn't available and we fall back to no_blend.

    # Resolve weight: call arg > settings (st.secrets → .env → env) > default.
    weight = weight_override
    if weight is None:
        try:
            from config.settings import V2_BLEND_WEIGHT as _W
            weight = _W
        except Exception:
            weight = _DEFAULT_V2_BLEND_WEIGHT
    weight = max(0.0, min(1.0, float(weight)))

    try:
        from data.news_sentiment import filter_relevant_news, get_news
        raw = get_news(ticker) or []
        relevant = filter_relevant_news(raw, ticker)[:25]
        headlines = [
            {"title": a.get("title", ""), "description": a.get("description", "") or ""}
            for a in relevant
            if (a.get("title") or "").strip()
        ]
        if not headlines:
            return no_blend

        v2_pred = v2svc.predict_v2(ticker, headlines)
        blended = (1.0 - weight) * stacker_prob + weight * float(v2_pred.prob_up)
        blended = max(0.0, min(1.0, blended))
        return V2BlendInfo(
            stacker_prob=stacker_prob,
            v2_prob=float(v2_pred.prob_up),
            blended_prob=blended,
            weight_v2=weight,
            n_headlines=int(v2_pred.n_headlines),
            stacker_available=bool(v2_pred.stacker_available),
            used=True,
        )
    except Exception:
        return no_blend


def _to_calibration(raw) -> CalibrationReport | None:
    if not isinstance(raw, dict):
        return None
    try:
        return CalibrationReport(
            n_samples=int(raw.get("n_samples", 0)),
            ece=float(raw.get("ece", 0.0)),
            brier_score=float(raw.get("brier_score", 0.0)),
            bin_edges=[float(x) for x in raw.get("bin_edges", [])],
            bin_predicted=[float(x) for x in raw.get("bin_predicted", [])],
            bin_actual=[float(x) for x in raw.get("bin_actual", [])],
            bin_counts=[int(x) for x in raw.get("bin_counts", [])],
        )
    except (TypeError, ValueError):
        return None


def _to_threshold_tuning(raw) -> ThresholdTuning | None:
    if not isinstance(raw, dict):
        return None
    tau = _opt(raw.get("tau_star"))
    if tau is None:
        return None
    return ThresholdTuning(
        tau_star=max(0.0, min(1.0, tau)),
        auc=_opt(raw.get("auc")),
        accuracy_at_tau=_opt(raw.get("accuracy_at_tau")),
    )


def _to_shap(raw) -> list[ShapFeature]:
    out: list[ShapFeature] = []
    if not raw:
        return out
    # Accept either a list of (name, importance) tuples, list of dicts, or dict.
    if isinstance(raw, dict):
        iterable = raw.items()
    else:
        iterable = raw
    for entry in iterable:
        try:
            if isinstance(entry, tuple) and len(entry) >= 2:
                name, imp = entry[0], entry[1]
            elif isinstance(entry, dict):
                name, imp = entry.get("feature"), entry.get("importance")
            elif isinstance(entry, (list,)) and len(entry) >= 2:
                name, imp = entry[0], entry[1]
            else:
                continue
            if name is None or imp is None:
                continue
            out.append(ShapFeature(feature=str(name), importance=float(imp)))
        except (TypeError, ValueError):
            continue
    return out[:12]


def _to_rmse(metrics: dict) -> ModelRmseBreakdown:
    return ModelRmseBreakdown(
        xgb=_opt(metrics.get("xgb_rmse")),
        lgbm=_opt(metrics.get("lgbm_rmse")),
        catboost=_opt(metrics.get("catboost_rmse")),
        rnn=_opt(metrics.get("rnn_rmse")),
        stacked=_opt(metrics.get("stacked_rmse")),
    )


def _to_walkforward(metrics: dict) -> WalkforwardSummary | None:
    acc = _opt(metrics.get("walkforward_accuracy"))
    if acc is None:
        return None
    windows = metrics.get("walkforward_windows")
    n = len(windows) if isinstance(windows, (list, tuple)) else None
    return WalkforwardSummary(
        accuracy=acc,
        std=_opt(metrics.get("walkforward_std")),
        min=_opt(metrics.get("walkforward_min")),
        max=_opt(metrics.get("walkforward_max")),
        n_windows=n,
    )


def _compute_pattern_conviction(df: "pd.DataFrame") -> float:
    """Return a signal in [-1, 1]: positive = net bearish patterns, negative = bullish.
    Used to dampen v2 blend weight when confirmed bearish patterns dominate.
    Returns 0 on any failure so prediction degrades gracefully."""
    try:
        from models.visual_analyst import PatternAnalyst
        analyst = PatternAnalyst()
        result = analyst.analyze_all_patterns(df)
        patterns = result.get("patterns", [])
        score = 0.0
        for p in patterns:
            if str(p.get("Status", "")).upper() != "CONFIRMED":
                continue
            conf = float(p.get("Confidence", 50)) / 100.0
            ptype = str(p.get("Type", ""))
            if "Bearish" in ptype:
                score += conf
            elif "Bullish" in ptype:
                score -= conf
        return float(max(-1.0, min(1.0, score)))
    except Exception:
        return 0.0


def _adjust_v2_weight_for_patterns(
    base_weight: float | None, conviction: float
) -> float | None:
    """Scale down the v2 blend weight when confirmed bearish patterns are strong.
    conviction > 0.5  → strongly bearish → cap at 10% of base weight.
    conviction 0.25-0.5 → linearly taper.
    conviction <= 0.25  → no change."""
    if base_weight is None:
        return None
    if conviction <= 0.25:
        return base_weight
    if conviction >= 0.50:
        return base_weight * 0.33  # at most 1/3 of the normal sentiment influence
    # linear taper between 0.25 and 0.50
    scale = 1.0 - ((conviction - 0.25) / 0.25) * 0.67
    return base_weight * scale


def _to_conviction(raw) -> "ConvictionPrecision | None":
    if not isinstance(raw, dict):
        return None
    from schemas.prediction import ConvictionPrecision
    return ConvictionPrecision(
        threshold=raw.get("threshold", 0.60),
        n_high_bull=raw.get("n_high_bull", 0),
        n_high_bear=raw.get("n_high_bear", 0),
        bull_precision=raw.get("bull_precision"),
        bear_precision=raw.get("bear_precision"),
        conviction_rate=raw.get("conviction_rate", 0.0),
    )


def _opt(v):
    try:
        if v is None:
            return None
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None
