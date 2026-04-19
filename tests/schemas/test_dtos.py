"""Schema validation smoke tests — no network, no models."""

from __future__ import annotations

from datetime import UTC, date, datetime, timezone

import pytest
from pydantic import ValidationError

from schemas import (
    BacktestMetrics,
    BacktestResult,
    CalibrationReport,
    Fundamentals,
    HeadlineSentiment,
    PredictionBundle,
    PredictionPoint,
    SentimentAggregate,
    SentimentScore,
    StockBar,
    StockHistory,
)


def test_stock_history_round_trip():
    bar = StockBar(ts=datetime(2024, 1, 1), open=1, high=2, low=0.5, close=1.5, volume=1000)
    hist = StockHistory(ticker="RELIANCE.NS", start=date(2024, 1, 1), end=date(2024, 1, 2), bars=[bar])
    assert hist.n_bars == 1
    assert hist.bars[0].close == 1.5


def test_sentiment_score_bounds():
    SentimentScore(label="positive", confidence=0.9)
    with pytest.raises(ValidationError):
        SentimentScore(label="positive", confidence=1.5)


def test_sentiment_aggregate_defaults_empty():
    agg = SentimentAggregate(
        ticker="X",
        window_start=datetime.now(UTC),
        window_end=datetime.now(UTC),
        n_headlines=0,
        mean_score=0.0,
    )
    assert agg.headlines == []
    assert agg.pos_count == 0


def test_prediction_point_direction_enum():
    pt = PredictionPoint(
        target_date=date.today(),
        pred_price=100.0,
        direction="up",
        prob_up=0.6,
    )
    assert pt.direction == "up"
    with pytest.raises(ValidationError):
        PredictionPoint(
            target_date=date.today(), pred_price=100.0, direction="sideways", prob_up=0.5
        )


def test_prediction_bundle_allows_empty_points():
    pb = PredictionBundle(
        ticker="X",
        made_at=datetime.now(UTC),
        model_version="hybrid-v1",
        horizon_days=5,
        points=[],
    )
    assert pb.horizon_days == 5


def test_calibration_report_shapes():
    rep = CalibrationReport(
        n_samples=100,
        ece=0.04,
        brier_score=0.18,
        bin_edges=[0, 0.5, 1.0],
        bin_predicted=[0.25, 0.75],
        bin_actual=[0.30, 0.70],
        bin_counts=[50, 50],
    )
    assert rep.ece == 0.04


def test_backtest_metrics_win_rate_bounded():
    BacktestMetrics(
        total_return_pct=10.0, cagr_pct=5.0, sharpe=1.2, max_drawdown_pct=-15.0, win_rate_pct=55.0
    )
    with pytest.raises(ValidationError):
        BacktestMetrics(
            total_return_pct=0, cagr_pct=0, sharpe=0, max_drawdown_pct=0, win_rate_pct=150.0
        )


def test_backtest_result_composes():
    res = BacktestResult(
        ticker="X",
        start=date(2024, 1, 1),
        end=date(2024, 6, 1),
        strategy="ma_crossover",
        initial_capital=100_000,
        final_equity=112_500,
        metrics=BacktestMetrics(
            total_return_pct=12.5, cagr_pct=25.0, sharpe=1.1, max_drawdown_pct=-8.0, win_rate_pct=52.0
        ),
    )
    assert res.final_equity == 112_500


def test_headline_sentiment_optional_fields():
    hs = HeadlineSentiment(
        title="Reliance beats earnings",
        source="ET",
        score=SentimentScore(label="positive", confidence=0.9),
    )
    assert hs.url is None
    assert hs.category == "Other"


def test_fundamentals_coerces_none():
    f = Fundamentals(ticker="X", forward_pe=None, peg_ratio=22.0)
    assert f.forward_pe is None
    assert f.peg_ratio == 22.0
