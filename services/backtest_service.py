"""
Backtest service — typed wrapper over `models.backtester.VectorizedBacktester`.

The heavy compute + transaction-cost modeling stays in `models/backtester.py`;
this layer just translates input / output shapes into `schemas.backtest`.

`VectorizedBacktester.run_backtest()` returns a dict with Title-Case keys and
fractional values (0.05 = 5%). This service converts to the snake_case,
percent-denominated `BacktestMetrics` schema so the UI / FastAPI both get the
same shape. CAGR / Sortino / Calmar / Expectancy are left `None` until A8.2
extends the backtester.

Equity + benchmark curves are downsampled to at most 300 points so multi-year
horizons don't bloat the API payload.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from config.settings import TradingConfig
from models.backtester import VectorizedBacktester, ma_crossover_signals, momentum_signals
from schemas.backtest import BacktestEquityPoint, BacktestMetrics, BacktestResult
from services.stock_service import get_history_df

_MAX_CURVE_POINTS = 300


def run_backtest(
    ticker: str,
    strategy: str = "ma_crossover",
    start: date | None = None,
    end: date | None = None,
    initial_capital: float | None = None,
    include_costs: bool = True,
) -> BacktestResult:
    df = get_history_df(ticker, start, end)
    if df is None or df.empty:
        raise ValueError(f"No history for {ticker}")

    data = df.copy()
    data["Actual_Return"] = data["Close"].pct_change().fillna(0.0)

    if strategy == "momentum":
        signals = momentum_signals(data["Close"])
    elif strategy == "ma_crossover":
        signals = ma_crossover_signals(data["Close"])
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    bt = VectorizedBacktester(data, signals)
    raw = bt.run_backtest(initial_capital=initial_capital, include_costs=include_costs)

    capital = float(initial_capital or TradingConfig.DEFAULT_INITIAL_CAPITAL)
    equity_curve = _equity_to_points(raw.get("Equity Curve"))
    benchmark = _buy_and_hold_benchmark(data["Close"], capital)

    return BacktestResult(
        ticker=ticker,
        start=pd.to_datetime(start or df.index[0]).date(),
        end=pd.to_datetime(end or df.index[-1]).date(),
        strategy=strategy,
        initial_capital=capital,
        final_equity=_final_equity(raw, capital),
        metrics=_to_metrics(raw),
        equity_curve=equity_curve,
        benchmark_equity_curve=benchmark,
    )


def _final_equity(raw: dict, capital: float) -> float:
    ec = raw.get("Equity Curve")
    if ec is not None:
        arr = np.asarray(ec)
        if len(arr):
            return float(arr[-1])
    return capital


def _equity_to_points(series) -> list[BacktestEquityPoint]:
    if series is None:
        return []
    ser = pd.Series(series).dropna()
    if ser.empty:
        return []
    ser = _downsample(ser, _MAX_CURVE_POINTS)
    out: list[BacktestEquityPoint] = []
    for idx, val in ser.items():
        try:
            d = pd.to_datetime(idx).date()
            v = float(val)
        except (TypeError, ValueError):
            continue
        if v != v:  # NaN guard
            continue
        out.append(BacktestEquityPoint(date=d, equity=v))
    return out


def _buy_and_hold_benchmark(close: pd.Series, capital: float) -> list[BacktestEquityPoint]:
    px = pd.Series(close).dropna()
    if px.empty:
        return []
    equity = capital * (px / float(px.iloc[0]))
    return _equity_to_points(equity)


def _downsample(series: pd.Series, max_points: int) -> pd.Series:
    n = len(series)
    if n <= max_points:
        return series
    step = max(1, n // max_points)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return series.iloc[idx]


def _to_metrics(raw: dict) -> BacktestMetrics:
    total = float(raw.get("Total Return", 0.0))
    mdd = float(raw.get("Max Drawdown", 0.0))
    win = float(raw.get("Win Rate", 0.0))
    return BacktestMetrics(
        total_return_pct=total * 100.0,
        cagr_pct=0.0,  # Added in A8.2
        sharpe=float(raw.get("Sharpe Ratio", 0.0)),
        sortino=None,
        calmar=None,
        max_drawdown_pct=mdd * 100.0,
        profit_factor=_opt_finite(raw.get("Profit Factor")),
        win_rate_pct=max(0.0, min(100.0, win * 100.0)),
        avg_win_pct=_opt_pct(raw.get("Avg Win")),
        avg_loss_pct=_opt_pct(raw.get("Avg Loss")),
        expectancy_pct=None,
        n_trades=int(raw.get("N Trades", 0)),
        dm_pvalue=_extract_dm_pvalue(raw.get("Statistical Significance")),
    )


def _opt_pct(v) -> float | None:
    f = _opt_finite(v)
    return None if f is None else f * 100.0


def _opt_finite(v) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _extract_dm_pvalue(stat: dict | None) -> float | None:
    if not isinstance(stat, dict):
        return None
    for k in ("dm_pvalue", "p_value", "pvalue"):
        v = _opt_finite(stat.get(k))
        if v is not None:
            return v
    return None
