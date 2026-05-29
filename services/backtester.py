"""
Walk-forward backtester for the Phase 3 alpha signals (P3-9).

Runs a full walk-forward simulation on the Nifty 500 universe
(≈55 liquid names from Nifty 50 + Nifty Midcap extras) over 2014–2024.

Walk-forward design
-------------------
• Training window   : 3 years (756 bars)
• Validation window : 6 months (126 bars) — used for signal combiner IC calibration
• Test window        : 6 months (126 bars) — out-of-sample performance
• Step               : 6 months (forward-rolling)

This gives ≈ 20 independent test windows over 2014–2024, which is enough
for meaningful statistical evaluation (bootstrap confidence intervals provided).

Signals computed at each step
------------------------------
  1. Momentum (12-1 month) — from models.alpha_signals
  2. Reversal  (5-day)      — from models.alpha_signals
  (Options and LLM signals require live data; they default to 0 in backtesting)

Portfolio construction
----------------------
  • Signal combiner is fit on the training window.
  • PortfolioConstructor sizes positions with fractional Kelly + caps.
  • Positions are rebalanced monthly (every 21 bars).
  • Transaction costs are deducted at every rebalance using TransactionCostModel.

Output
------
  BacktestReport contains:
    equity_curve       : pd.Series (daily portfolio value, ₹100 = start)
    daily_returns      : pd.Series
    summary_stats      : dict  (Sharpe, CAGR, max DD, Calmar, etc.)
    walk_forward_slices: list of WalkForwardSlice (per-window stats)
    signal_attribution : dict  (average IC per signal type across windows)
    bootstrap_ci       : dict  (95% CI on Sharpe from 1000 bootstrap draws)

Usage — full backtest
---------------------
    from services.backtester import WalkForwardBacktester
    bt = WalkForwardBacktester(start_year=2014, end_year=2024)
    report = bt.run()
    print(report.summary_stats)

Usage — from saved prices (faster, no re-download)
---------------------
    from services.backtester import WalkForwardBacktester
    bt = WalkForwardBacktester.from_saved_prices("data/backtest_prices.parquet")
    report = bt.run()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from models.alpha_signals import MomentumSignal, ReversalSignal, compute_universe_signals
from models.portfolio_constructor import PortfolioConstructor
from models.signal_combiner import SignalBundle, SignalCombiner
from models.transaction_costs import TransactionCostModel, adjust_returns_for_costs

logger = logging.getLogger(__name__)

# ── Universe definition ───────────────────────────────────────────────────────
# Imported from the cross-sectional trainer to avoid duplication
from models.cross_sectional_trainer import (
    NIFTY50_TICKERS,
    NIFTY_MIDCAP_EXTRAS,
    TICKER_SECTOR,
)

_UNIVERSE       = NIFTY50_TICKERS + NIFTY_MIDCAP_EXTRAS
_NIFTY_TICKER   = "^NSEI"
_SAVED_DIR      = Path("data/backtest_cache")

# ── Walk-forward parameters ───────────────────────────────────────────────────
_TRAIN_BARS     = 756    # 3 years
_VAL_BARS       = 126    # 6 months
_TEST_BARS      = 126    # 6 months
_STEP_BARS      = 126    # roll forward by 6 months
_REBALANCE_BARS = 21     # monthly rebalancing
_MIN_TRAIN_BARS = 252    # minimum 1 year before first prediction

# ── Portfolio parameters ──────────────────────────────────────────────────────
_INITIAL_CAPITAL      = 10_000_000.0   # ₹1 crore starting capital
_AVG_ORDER_NOTIONAL   = 500_000.0      # ₹5 lakh per name (for cost calc)
_AVG_ADV_NOTIONAL     = 20_000_000.0   # ₹2 crore ADV (large-cap NSE)


# ── DTOs ─────────────────────────────────────────────────────────────────────

@dataclass
class WalkForwardSlice:
    """Performance statistics for one walk-forward test window."""

    window_id:      int
    train_start:    pd.Timestamp
    train_end:      pd.Timestamp
    test_start:     pd.Timestamp
    test_end:       pd.Timestamp

    n_trading_days: int    = 0
    n_rebalances:   int    = 0
    n_names_avg:    float  = 0.0

    # Return statistics
    total_return:   float  = 0.0   # fraction, e.g. 0.12 = 12%
    cagr:           float  = 0.0
    sharpe:         float  = 0.0
    max_drawdown:   float  = 0.0   # fraction, always negative
    calmar:         float  = 0.0
    turnover_ann:   float  = 0.0   # annualised one-way turnover fraction

    # Benchmark (Nifty 50 buy-and-hold over same period)
    bench_return:   float  = 0.0
    bench_sharpe:   float  = 0.0
    excess_return:  float  = 0.0   # total_return − bench_return

    # Signal ICs in this window
    ic_momentum:    float  = 0.0
    ic_reversal:    float  = 0.0


@dataclass
class BacktestReport:
    """Complete walk-forward backtest output."""

    # Equity curve (daily, ₹ nominal)
    equity_curve:     pd.Series = field(default_factory=pd.Series)
    daily_returns:    pd.Series = field(default_factory=pd.Series)
    nifty_curve:      pd.Series = field(default_factory=pd.Series)  # benchmark

    # Aggregate statistics
    summary_stats:    dict = field(default_factory=dict)

    # Per-window breakdown
    walk_forward_slices: list[WalkForwardSlice] = field(default_factory=list)

    # Signal attribution (average IC per signal)
    signal_attribution: dict = field(default_factory=dict)

    # Bootstrap 95% CI on Sharpe
    bootstrap_ci:     dict = field(default_factory=dict)

    # Metadata
    start_date:   str  = ""
    end_date:     str  = ""
    n_tickers:    int  = 0
    n_windows:    int  = 0

    def to_dict(self) -> dict:
        """Serialisable summary (without DataFrames)."""
        return {
            "summary_stats":      self.summary_stats,
            "signal_attribution": self.signal_attribution,
            "bootstrap_ci":       self.bootstrap_ci,
            "n_tickers":          self.n_tickers,
            "n_windows":          self.n_windows,
            "start_date":         self.start_date,
            "end_date":           self.end_date,
            "walk_forward_slices": [
                {
                    "window_id":      sl.window_id,
                    "test_start":     str(sl.test_start.date()),
                    "test_end":       str(sl.test_end.date()),
                    "sharpe":         round(sl.sharpe, 3),
                    "total_return":   round(sl.total_return, 4),
                    "bench_return":   round(sl.bench_return, 4),
                    "max_drawdown":   round(sl.max_drawdown, 4),
                    "ic_momentum":    round(sl.ic_momentum, 4),
                    "ic_reversal":    round(sl.ic_reversal, 4),
                }
                for sl in self.walk_forward_slices
            ],
        }


# ── Performance analytics ─────────────────────────────────────────────────────

def _sharpe(returns: np.ndarray, ann_factor: float = 252.0) -> float:
    """Annualised Sharpe ratio with 0% risk-free rate."""
    if len(returns) < 2:
        return 0.0
    mu  = float(np.mean(returns))
    sig = float(np.std(returns, ddof=1))
    if sig < 1e-10:
        return 0.0
    return float(mu / sig * np.sqrt(ann_factor))


def _max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a fraction (negative value)."""
    if len(equity) < 2:
        return 0.0
    peak     = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / (peak + 1e-10)
    return float(drawdown.min())


def _cagr(equity: np.ndarray, trading_days: int) -> float:
    """Compound annual growth rate from start to end of equity curve."""
    if trading_days <= 0 or equity[0] <= 0 or equity[-1] <= 0:
        return 0.0
    total_years = trading_days / 252.0
    return float((equity[-1] / equity[0]) ** (1.0 / total_years) - 1.0)


def _ic(signal_series: pd.Series, forward_return: pd.Series) -> float:
    """Rank correlation (IC) of signal with next-period return."""
    df = pd.DataFrame({"s": signal_series, "r": forward_return}).dropna()
    if len(df) < 10:
        return 0.0
    return float(df["s"].rank(pct=True).corr(df["r"].rank(pct=True)))


def _bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_boot: int = 1000,
    ci: float   = 0.95,
) -> dict:
    """
    Bootstrap 95% confidence interval on the annualised Sharpe ratio.
    Stationary bootstrap (Politis & Romano 1994) with mean block length 10.
    """
    if len(returns) < 20:
        sharpe_val = _sharpe(returns)
        return {"mean": sharpe_val, "lo": sharpe_val, "hi": sharpe_val}

    rng = np.random.default_rng(42)
    n   = len(returns)
    boot_sharpes: list[float] = []

    for _ in range(n_boot):
        # Stationary bootstrap block
        block_len = max(1, int(rng.geometric(p=0.1)))  # mean block length 10
        idxs: list[int] = []
        start = int(rng.integers(0, n))
        while len(idxs) < n:
            end = min(start + block_len, n)
            idxs.extend(range(start, end))
            start = int(rng.integers(0, n))
            block_len = max(1, int(rng.geometric(p=0.1)))
        sample = returns[idxs[:n]]
        boot_sharpes.append(_sharpe(sample))

    arr  = np.array(boot_sharpes)
    lo_p = (1.0 - ci) / 2.0
    hi_p = 1.0 - lo_p
    return {
        "mean": float(np.mean(arr)),
        "lo":   float(np.quantile(arr, lo_p)),
        "hi":   float(np.quantile(arr, hi_p)),
        "std":  float(np.std(arr, ddof=1)),
    }


# ── Main backtester ───────────────────────────────────────────────────────────

class WalkForwardBacktester:
    """
    Walk-forward equity backtest for the Phase 3 alpha pipeline.

    The backtest is:
      • Strictly out-of-sample: signal combiner is fit only on training data.
      • Transaction-cost aware: every rebalance deducts sqrt-impact costs.
      • Benchmark-aware: tracks Nifty 50 buy-and-hold over the same period.

    Parameters
    ----------
    start_year : int
        First year of data to download. Default 2014.
    end_year : int
        Last year of data to download. Default 2024.
    tickers : list[str] | None
        Universe of tickers. Default: Nifty 50 + Midcap extras.
    train_bars : int
        Training window in trading bars.
    test_bars : int
        Test window in trading bars.
    step_bars : int
        Step size (how far the window moves forward each iteration).
    rebalance_bars : int
        Portfolio rebalancing frequency in bars.
    initial_capital : float
        Starting capital in INR.
    """

    def __init__(
        self,
        start_year:      int   = 2014,
        end_year:        int   = 2024,
        tickers:         Optional[list[str]] = None,
        train_bars:      int   = _TRAIN_BARS,
        val_bars:        int   = _VAL_BARS,
        test_bars:       int   = _TEST_BARS,
        step_bars:       int   = _STEP_BARS,
        rebalance_bars:  int   = _REBALANCE_BARS,
        initial_capital: float = _INITIAL_CAPITAL,
    ) -> None:
        self.start_year      = start_year
        self.end_year        = end_year
        self.tickers         = tickers or _UNIVERSE
        self.train_bars      = train_bars
        self.val_bars        = val_bars
        self.test_bars       = test_bars
        self.step_bars       = step_bars
        self.rebalance_bars  = rebalance_bars
        self.initial_capital = initial_capital

        self._prices: Optional[pd.DataFrame]     = None  # date × ticker close prices
        self._nifty:  Optional[pd.Series]        = None  # NIFTY50 close

    # ── Data loading ──────────────────────────────────────────────────────────

    def _download_prices(self) -> None:
        """Download historical prices for all tickers + NIFTY50 index."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("WalkForwardBacktester requires: pip install yfinance")

        period_str = f"{self.end_year - self.start_year + 2}y"
        logger.info(
            "WalkForwardBacktester: downloading %d tickers × %s ...",
            len(self.tickers), period_str,
        )

        # Download NIFTY50 benchmark
        nifty_raw = yf.download(_NIFTY_TICKER, period=period_str, auto_adjust=True,
                                progress=False, timeout=30)
        if isinstance(nifty_raw.columns, pd.MultiIndex):
            nifty_raw.columns = [c[0] for c in nifty_raw.columns]
        if nifty_raw.index.tz is not None:
            nifty_raw.index = nifty_raw.index.tz_localize(None)
        self._nifty = nifty_raw["Close"].dropna()

        # Download all tickers in one batch call (yfinance is faster in batch)
        batch_raw = yf.download(
            self.tickers, period=period_str, auto_adjust=True,
            progress=False, timeout=60,
        )
        if isinstance(batch_raw.columns, pd.MultiIndex):
            # MultiIndex: (field, ticker) — extract "Close"
            if "Close" in batch_raw.columns.get_level_values(0):
                close_df = batch_raw["Close"]
            else:
                close_df = batch_raw.xs("Close", level=0, axis=1)
        else:
            close_df = batch_raw[["Close"]] if "Close" in batch_raw.columns else batch_raw

        if close_df.index.tz is not None:
            close_df.index = close_df.index.tz_localize(None)

        # Filter to start/end year
        close_df = close_df[
            (close_df.index.year >= self.start_year) &
            (close_df.index.year <= self.end_year)
        ]

        # Drop tickers with >40% missing
        ok_cols = [c for c in close_df.columns if close_df[c].notna().mean() > 0.60]
        self._prices = close_df[ok_cols].copy()

        logger.info(
            "WalkForwardBacktester: loaded %d tickers × %d bars",
            len(ok_cols), len(self._prices),
        )

    def save_prices(self, path: Optional[Path] = None) -> Path:
        """Save downloaded prices to a parquet file for faster re-runs."""
        p = Path(path or (_SAVED_DIR / "backtest_prices.parquet"))
        p.parent.mkdir(parents=True, exist_ok=True)
        if self._prices is not None:
            self._prices.to_parquet(p)
        if self._nifty is not None:
            self._nifty.to_frame("Close").to_parquet(p.with_suffix(".nifty.parquet"))
        logger.info("Prices saved to %s", p)
        return p

    @classmethod
    def from_saved_prices(cls, path: str | Path, **kwargs) -> "WalkForwardBacktester":
        """Load prices from a previously saved parquet file."""
        p = Path(path)
        bt = cls(**kwargs)
        bt._prices = pd.read_parquet(p)
        nifty_p = p.with_suffix(".nifty.parquet")
        if nifty_p.exists():
            bt._nifty = pd.read_parquet(nifty_p)["Close"]
        return bt

    # ── Backtest engine ───────────────────────────────────────────────────────

    def run(self, download_if_needed: bool = True) -> BacktestReport:
        """
        Execute the full walk-forward backtest.

        Returns
        -------
        BacktestReport
        """
        if self._prices is None:
            if download_if_needed:
                self._download_prices()
            else:
                raise RuntimeError(
                    "No price data loaded. Call download_prices() or "
                    "use from_saved_prices()."
                )

        prices  = self._prices.copy()
        n_dates = len(prices)
        dates   = prices.index

        if n_dates < self.train_bars + self.test_bars:
            raise RuntimeError(
                f"Only {n_dates} bars available — need at least "
                f"{self.train_bars + self.test_bars}."
            )

        # ── Build walk-forward windows ────────────────────────────────────────
        portfolio_daily_returns: list[float] = []
        portfolio_dates:          list[pd.Timestamp] = []
        nifty_daily_returns:      list[float] = []
        wf_slices:                list[WalkForwardSlice] = []
        all_ic_mom:               list[float] = []
        all_ic_rev:               list[float] = []

        constructors  = PortfolioConstructor()
        combiner      = SignalCombiner()
        cost_model    = TransactionCostModel()

        cursor = self.train_bars  # first bar after initial training window

        while cursor + self.test_bars <= n_dates:
            train_start_i = max(0, cursor - self.train_bars)
            train_end_i   = cursor
            test_start_i  = cursor
            test_end_i    = min(cursor + self.test_bars, n_dates)

            train_prices = prices.iloc[train_start_i: train_end_i]
            test_prices  = prices.iloc[test_start_i:  test_end_i]

            if len(test_prices) < 5:
                cursor += self.step_bars
                continue

            # ── Signal combiner fit on training window ────────────────────────
            combiner = self._fit_combiner_on_window(train_prices, combiner)

            # ── Simulate the test window ──────────────────────────────────────
            # Pass full_prices + test_start_i so signals can look back 252+ bars
            window_returns, window_dates, n_rebal, avg_names, ic_m, ic_r = (
                self._simulate_test_window(
                    test_prices, combiner, constructors, cost_model,
                    full_prices=prices, test_start_i=test_start_i,
                )
            )

            portfolio_daily_returns.extend(window_returns)
            portfolio_dates.extend(window_dates)

            # Nifty benchmark returns for the same window
            if self._nifty is not None:
                nifty_slice = self._nifty.reindex(test_prices.index, method="ffill").dropna()
                nifty_rets  = nifty_slice.pct_change().fillna(0).values.tolist()
                nifty_daily_returns.extend(nifty_rets[: len(window_returns)])

            # ── Per-slice stats ───────────────────────────────────────────────
            wr_arr = np.array(window_returns)
            slice_sharpe = _sharpe(wr_arr)
            slice_equity = np.cumprod(1.0 + wr_arr) * 100.0
            slice_mdd    = _max_drawdown(slice_equity)
            slice_cagr   = _cagr(slice_equity, len(wr_arr))
            slice_total  = float((1.0 + wr_arr).prod() - 1.0)

            # Benchmark stats
            nr_arr = np.array(nifty_daily_returns[-len(window_returns):]) if nifty_daily_returns else np.zeros_like(wr_arr)
            bench_total = float((1.0 + nr_arr).prod() - 1.0)
            bench_sharpe = _sharpe(nr_arr)

            wf_slices.append(WalkForwardSlice(
                window_id      = len(wf_slices),
                train_start    = dates[train_start_i],
                train_end      = dates[train_end_i - 1],
                test_start     = dates[test_start_i],
                test_end       = dates[test_end_i - 1],
                n_trading_days = len(wr_arr),
                n_rebalances   = n_rebal,
                n_names_avg    = avg_names,
                total_return   = slice_total,
                cagr           = slice_cagr,
                sharpe         = slice_sharpe,
                max_drawdown   = slice_mdd,
                calmar         = abs(slice_cagr / slice_mdd) if slice_mdd < -0.001 else 0.0,
                bench_return   = bench_total,
                bench_sharpe   = bench_sharpe,
                excess_return  = slice_total - bench_total,
                ic_momentum    = ic_m,
                ic_reversal    = ic_r,
            ))
            all_ic_mom.append(ic_m)
            all_ic_rev.append(ic_r)

            logger.info(
                "WF window %d: %s–%s | Sharpe=%.2f | Return=%.1f%% | Bench=%.1f%%",
                len(wf_slices) - 1,
                dates[test_start_i].date(),
                dates[test_end_i - 1].date(),
                slice_sharpe,
                slice_total * 100,
                bench_total * 100,
            )

            cursor += self.step_bars

        # ── Build equity curves ───────────────────────────────────────────────
        all_rets = np.array(portfolio_daily_returns)
        equity   = pd.Series(
            np.cumprod(np.concatenate([[1.0], 1.0 + all_rets])) * self.initial_capital,
            index   = [dates[self.train_bars - 1]] + portfolio_dates,
        )

        daily_rets_series = pd.Series(all_rets, index=portfolio_dates)

        nifty_curve = pd.Series(dtype=float)
        if nifty_daily_returns:
            nr = np.array(nifty_daily_returns[: len(all_rets)])
            nifty_curve = pd.Series(
                np.cumprod(np.concatenate([[1.0], 1.0 + nr])) * self.initial_capital,
                index = equity.index,
            )

        # ── Aggregate statistics ──────────────────────────────────────────────
        ann_sharpe   = _sharpe(all_rets)
        total_equity = equity.values
        summary = {
            "sharpe_ratio":        round(ann_sharpe, 3),
            "cagr_pct":            round(_cagr(total_equity, len(all_rets)) * 100, 2),
            "max_drawdown_pct":    round(_max_drawdown(total_equity) * 100, 2),
            "calmar_ratio":        round(
                abs(_cagr(total_equity, len(all_rets)) / (_max_drawdown(total_equity) + 1e-10)),
                3,
            ),
            "total_return_pct":    round(float((total_equity[-1] / total_equity[0] - 1)) * 100, 2),
            "ann_vol_pct":         round(float(np.std(all_rets, ddof=1) * np.sqrt(252)) * 100, 2),
            "n_windows":           len(wf_slices),
            "pct_windows_positive": round(
                sum(1 for sl in wf_slices if sl.sharpe > 0) / max(len(wf_slices), 1) * 100, 1,
            ),
            "mean_window_sharpe":  round(float(np.mean([sl.sharpe  for sl in wf_slices])), 3),
            "mean_excess_return":  round(float(np.mean([sl.excess_return for sl in wf_slices])) * 100, 2),
        }

        # ── Bootstrap CI on Sharpe ────────────────────────────────────────────
        bootstrap_ci = _bootstrap_sharpe_ci(all_rets)

        # ── Signal attribution ────────────────────────────────────────────────
        signal_attribution = {
            "momentum_ic_mean": round(float(np.mean(all_ic_mom)) if all_ic_mom else 0.0, 4),
            "reversal_ic_mean": round(float(np.mean(all_ic_rev)) if all_ic_rev else 0.0, 4),
            "momentum_ic_std":  round(float(np.std(all_ic_mom))  if all_ic_mom else 0.0, 4),
            "reversal_ic_std":  round(float(np.std(all_ic_rev))  if all_ic_rev else 0.0, 4),
        }

        logger.info(
            "WalkForwardBacktester: DONE  Sharpe=%.2f  CAGR=%.1f%%  MaxDD=%.1f%%  "
            "Bootstrap 95%%CI=[%.2f, %.2f]",
            ann_sharpe,
            summary["cagr_pct"],
            summary["max_drawdown_pct"],
            bootstrap_ci["lo"],
            bootstrap_ci["hi"],
        )

        return BacktestReport(
            equity_curve         = equity,
            daily_returns        = daily_rets_series,
            nifty_curve          = nifty_curve,
            summary_stats        = summary,
            walk_forward_slices  = wf_slices,
            signal_attribution   = signal_attribution,
            bootstrap_ci         = bootstrap_ci,
            start_date           = str(dates[self.train_bars].date()),
            end_date             = str(dates[-1].date()),
            n_tickers            = len(prices.columns),
            n_windows            = len(wf_slices),
        )

    # ── Private: combiner training on one window ──────────────────────────────

    def _fit_combiner_on_window(
        self,
        train_prices: pd.DataFrame,
        combiner: SignalCombiner,
    ) -> SignalCombiner:
        """
        Fit the SignalCombiner IC estimates on the training window.

        Builds a history DataFrame with per-ticker signal values + forward
        returns, then calls combiner.fit().
        """
        mom_scorer = MomentumSignal()
        rev_scorer = ReversalSignal()

        records: list[dict] = []
        horizon = 5  # match cross-sectional trainer target

        for ticker in train_prices.columns:
            close = train_prices[ticker].dropna()
            if len(close) < 280:
                continue

            mom_s = mom_scorer.score_series(close)
            rev_s = rev_scorer.score_series(close)
            fwd_r = np.log(close.shift(-horizon) / close).shift(0)

            valid = mom_s.notna() & rev_s.notna() & fwd_r.notna()
            for dt in close.index[valid]:
                records.append({
                    "date":             dt,
                    "ticker":           ticker,
                    "momentum_signal":  float(mom_s.at[dt]),
                    "reversal_signal":  float(rev_s.at[dt]),
                    "options_signal":   0.0,
                    "llm_signal":       0.0,
                    "future_return_5d": float(fwd_r.at[dt]),
                })

        if not records:
            return combiner

        df = pd.DataFrame(records)
        try:
            combiner.fit(df)
        except Exception as exc:
            logger.warning("_fit_combiner_on_window: fit failed: %s", exc)

        return combiner

    # ── Private: simulate one test window ────────────────────────────────────

    def _simulate_test_window(
        self,
        test_prices: pd.DataFrame,
        combiner: SignalCombiner,
        constructor: PortfolioConstructor,
        cost_model: TransactionCostModel,
        full_prices: Optional[pd.DataFrame] = None,
        test_start_i: int = 0,
    ) -> tuple[list[float], list[pd.Timestamp], int, float, float, float]:
        """
        Simulate portfolio performance over one test window.

        full_prices  : full price matrix (train + test) so signals can look
                       back 252+ bars even on the first day of the test window.
        test_start_i : index of the first test bar within full_prices.

        Returns:
          (daily_returns, dates, n_rebalances, avg_names, ic_momentum, ic_reversal)
        """
        mom_scorer  = MomentumSignal()
        rev_scorer  = ReversalSignal()

        daily_returns: list[float]         = []
        dates:         list[pd.Timestamp]  = []
        n_rebalances   = 0
        names_per_day: list[int]           = []

        # Current portfolio weights
        current_weights: dict[str, float] = {}

        # For IC calculation
        signal_records: list[dict] = []

        for i, dt in enumerate(test_prices.index):
            day_returns: list[float] = []

            # ── Rebalance? ────────────────────────────────────────────────────
            if i % self.rebalance_bars == 0:
                # Compute signals using full history up to this test bar
                # (train window + i bars into the test window)
                new_signals: dict[str, float] = {}

                for ticker in test_prices.columns:
                    # Use full_prices if available so momentum can look back 252+ bars
                    if full_prices is not None and ticker in full_prices.columns:
                        close = full_prices[ticker].iloc[: test_start_i + i + 1].dropna()
                    else:
                        close = test_prices[ticker].iloc[: i + 1].dropna()

                    # Need at least lookback+skip+1 bars for momentum; 6 for reversal
                    if len(close) < mom_scorer.lookback + mom_scorer.skip + 2:
                        continue

                    mom_score = mom_scorer.score(close)
                    rev_score = rev_scorer.score(close)

                    if mom_score is None and rev_score is None:
                        continue

                    bundle    = SignalBundle(
                        ticker           = ticker,
                        momentum_signal  = mom_score,
                        reversal_signal  = rev_score,
                    )
                    combined  = combiner.combine(bundle)
                    sig_val   = float(combined.composite_signal)

                    new_signals[ticker] = sig_val

                    # Record for IC computation
                    if mom_score is not None:
                        signal_records.append({
                            "ticker":           ticker,
                            "date":             dt,
                            "momentum_signal":  mom_score,
                            "reversal_signal":  rev_score if rev_score is not None else 0.0,
                        })

                # Volatility estimates (use full history for stable 20d vol)
                sigma_dict: dict[str, float] = {}
                for ticker in test_prices.columns:
                    if full_prices is not None and ticker in full_prices.columns:
                        close_v = full_prices[ticker].iloc[: test_start_i + i + 1].dropna()
                    else:
                        close_v = test_prices[ticker].iloc[: i + 1].dropna()
                    if len(close_v) >= 20:
                        log_rets = np.log(close_v / close_v.shift(1)).dropna()
                        sigma_dict[ticker] = float(log_rets.rolling(20).std().iloc[-1])

                # Sector map
                sector_map = {t: TICKER_SECTOR.get(t, -1) for t in new_signals}

                new_weights = constructor.construct(
                    new_signals, sigma_dict, sector_map
                ).weights

                # Compute turnover (one-way)
                if current_weights:
                    all_tickers = set(current_weights) | set(new_weights)
                    turnover = 0.5 * sum(
                        abs(new_weights.get(t, 0.0) - current_weights.get(t, 0.0))
                        for t in all_tickers
                    )
                    # Deduct cost on turnover fraction
                    cost_est = cost_model.estimate(
                        ticker         = "portfolio",
                        order_notional = turnover * self.initial_capital,
                        sigma_daily    = 0.015,
                        adv_notional   = _AVG_ADV_NOTIONAL,
                    )
                    day_returns.append(-cost_est.total_fraction * turnover)

                current_weights = new_weights
                n_rebalances   += 1

            # ── Compute day's portfolio return ────────────────────────────────
            if i > 0 and current_weights:
                port_ret = 0.0
                for ticker, weight in current_weights.items():
                    if ticker not in test_prices.columns:
                        continue
                    prev_close = test_prices[ticker].iloc[i - 1]
                    curr_close = test_prices[ticker].iloc[i]
                    if pd.isna(prev_close) or pd.isna(curr_close) or prev_close <= 0:
                        continue
                    port_ret += weight * float((curr_close - prev_close) / prev_close)
                day_returns.append(port_ret)

            total_day = float(sum(day_returns)) if day_returns else 0.0
            daily_returns.append(total_day)
            dates.append(dt)
            names_per_day.append(len(current_weights))

        # IC calculation for this window
        ic_m = ic_r = 0.0
        if signal_records and len(test_prices) > 5:
            sig_df = pd.DataFrame(signal_records)
            # Compute 5-day forward returns for IC
            fwd_rets: list[dict] = []
            for ticker in test_prices.columns:
                close = test_prices[ticker].dropna()
                for j in range(len(close) - 5):
                    fwd = float(np.log(close.iloc[j + 5] / close.iloc[j]))
                    fwd_rets.append({"ticker": ticker, "date": close.index[j], "fwd": fwd})
            if fwd_rets:
                fwd_df = pd.DataFrame(fwd_rets)
                merged = sig_df.merge(fwd_df, on=["ticker", "date"], how="inner")
                if not merged.empty:
                    ic_m = _ic(merged["momentum_signal"], merged["fwd"])
                    ic_r = _ic(merged["reversal_signal"], merged["fwd"])

        avg_names = float(np.mean(names_per_day)) if names_per_day else 0.0
        return daily_returns, dates, n_rebalances, avg_names, ic_m, ic_r
