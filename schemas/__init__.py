"""
Typed contracts (Pydantic DTOs) shared between services, Streamlit UI, and the
future FastAPI backend. Services accept / return these types; UI renders them.

Keep these narrow and serializable — they cross process boundaries.
"""

from schemas.backtest import BacktestMetrics, BacktestResult
from schemas.ledger import AccuracyWindow, LedgerRow
from schemas.paper_trade import PaperBookState, PaperFill, PaperPosition
from schemas.prediction import (
    CalibrationReport,
    PredictionBundle,
    PredictionPoint,
)
from schemas.sentiment import HeadlineSentiment, SentimentAggregate, SentimentScore
from schemas.stock import Fundamentals, StockBar, StockHistory
from schemas.v2_ensemble import V2EnsemblePrediction, V2ModelBreakdown

__all__ = [
    "StockBar",
    "StockHistory",
    "Fundamentals",
    "SentimentScore",
    "HeadlineSentiment",
    "SentimentAggregate",
    "PredictionPoint",
    "PredictionBundle",
    "CalibrationReport",
    "BacktestMetrics",
    "BacktestResult",
    "V2EnsemblePrediction",
    "V2ModelBreakdown",
    "LedgerRow",
    "AccuracyWindow",
    "PaperFill",
    "PaperPosition",
    "PaperBookState",
]
