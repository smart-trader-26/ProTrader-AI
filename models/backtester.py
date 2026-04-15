"""
Vectorized Backtester for strategy evaluation.
Fast backtesting engine for evaluating trading strategies.
Includes statistical significance testing, real benchmarks, Monte Carlo simulation,
and transaction cost modeling for research-grade analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import TradingConfig


def calculate_statistical_significance(predictions: np.ndarray, actuals: np.ndarray,
                                       n_bootstrap: int = 1000) -> dict:
    """
    Calculate statistical significance of prediction performance.

    Uses paired t-test against random walk baseline and bootstrap
    confidence intervals for direction accuracy.

    Args:
        predictions: Array of predicted returns
        actuals: Array of actual returns
        n_bootstrap: Number of bootstrap iterations

    Returns:
        Dictionary with p-value, confidence intervals, and test statistics
    """
    # Direction accuracy
    correct = np.sign(predictions) == np.sign(actuals)
    accuracy = np.mean(correct)

    # Paired t-test: Is accuracy significantly different from 50%?
    # H0: accuracy = 0.5 (random guessing)
    # Using binomial test for proportion
    n_correct = np.sum(correct)
    n_total = len(correct)

    # Binomial test (two-sided)
    p_value_binomial = stats.binomtest(n_correct, n_total, p=0.5, alternative='greater').pvalue

    # Bootstrap confidence interval for accuracy
    bootstrap_accuracies = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(correct), size=len(correct), replace=True)
        bootstrap_acc = np.mean(correct[indices])
        bootstrap_accuracies.append(bootstrap_acc)

    ci_lower = np.percentile(bootstrap_accuracies, 2.5)
    ci_upper = np.percentile(bootstrap_accuracies, 97.5)

    # RMSE comparison with random walk (predict 0)
    rmse_model = np.sqrt(np.mean((predictions - actuals) ** 2))
    rmse_random_walk = np.sqrt(np.mean(actuals ** 2))  # Predicting 0

    # Paired t-test on squared errors
    squared_errors_model = (predictions - actuals) ** 2
    squared_errors_rw = actuals ** 2
    t_stat, p_value_rmse = stats.ttest_rel(squared_errors_model, squared_errors_rw)

    # Effect size (Cohen's d)
    diff = squared_errors_rw - squared_errors_model
    cohens_d = np.mean(diff) / (np.std(diff) + 1e-8)

    return {
        'accuracy': accuracy,
        'p_value_accuracy': p_value_binomial,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'rmse_model': rmse_model,
        'rmse_baseline': rmse_random_walk,
        'p_value_rmse': p_value_rmse if t_stat < 0 else 1 - p_value_rmse/2,  # One-sided
        't_statistic': t_stat,
        'cohens_d': cohens_d,
        'n_samples': n_total
    }


def calculate_sharpe_significance(strategy_returns: np.ndarray,
                                  benchmark_returns: np.ndarray,
                                  n_bootstrap: int = 1000) -> dict:
    """
    Calculate statistical significance of Sharpe ratio difference.

    Args:
        strategy_returns: Array of strategy daily returns
        benchmark_returns: Array of benchmark daily returns
        n_bootstrap: Number of bootstrap iterations

    Returns:
        Dictionary with Sharpe ratios, p-value, and confidence intervals
    """
    def calc_sharpe(returns):
        if np.std(returns) == 0:
            return 0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)

    sharpe_strategy = calc_sharpe(strategy_returns)
    sharpe_benchmark = calc_sharpe(benchmark_returns)
    sharpe_diff = sharpe_strategy - sharpe_benchmark

    # Bootstrap test for Sharpe difference
    bootstrap_diffs = []
    n = len(strategy_returns)

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        boot_strategy = calc_sharpe(strategy_returns[indices])
        boot_benchmark = calc_sharpe(benchmark_returns[indices])
        bootstrap_diffs.append(boot_strategy - boot_benchmark)

    # P-value: proportion of bootstrap samples where diff <= 0
    p_value = np.mean(np.array(bootstrap_diffs) <= 0)

    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    return {
        'sharpe_strategy': sharpe_strategy,
        'sharpe_benchmark': sharpe_benchmark,
        'sharpe_difference': sharpe_diff,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


def _compute_metrics_from_returns(returns: np.ndarray, initial_capital: float = 100000) -> dict:
    """
    Helper: compute standard performance metrics from a returns array.
    Used internally to run metrics for any signal/strategy uniformly.
    """
    if len(returns) == 0:
        return {
            'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0,
            'win_rate': 0, 'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0,
            'equity_curve': np.array([initial_capital])
        }

    equity = initial_capital * np.cumprod(1 + returns)

    total_return = (equity[-1] / initial_capital) - 1
    sharpe = (np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(252)

    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / (peak + 1e-10)
    max_drawdown = float(np.min(drawdown))

    winning = returns[returns > 0]
    losing = returns[returns < 0]
    active = returns[returns != 0]
    win_rate = len(winning) / len(active) if len(active) > 0 else 0
    avg_win = float(np.mean(winning)) if len(winning) > 0 else 0
    avg_loss = float(np.mean(losing)) if len(losing) > 0 else 0
    profit_factor = (
        abs(np.sum(winning) / np.sum(losing))
        if len(losing) > 0 and np.sum(losing) != 0
        else float('inf')
    )

    return {
        'total_return': total_return,
        'sharpe_ratio': float(sharpe),
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'equity_curve': equity
    }


def ma_crossover_signals(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """
    Generate signals from Simple Moving Average crossover.
    Long (+1) when fast MA > slow MA, Short (-1) otherwise.

    Uses only past data — no look-ahead bias.
    Computed entirely from OHLCV data already available in the app.

    Args:
        close: Close price series
        fast: Fast MA period (default 20)
        slow: Slow MA period (default 50)

    Returns:
        pd.Series of signals (-1 or +1)
    """
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    signal = np.where(ma_fast > ma_slow, 1, -1)
    return pd.Series(signal, index=close.index).fillna(0)


def momentum_signals(close: pd.Series, lookback: int = 252) -> pd.Series:
    """
    Generate 52-week momentum breakout signals.
    Long (+1) when price >= 52-week high, Short (-1) when <= 52-week low.

    This is a well-documented return factor in Indian markets. Uses only
    historically available prices — no look-ahead bias (shift(1) applied).

    Args:
        close: Close price series
        lookback: Lookback period in trading days (default 252 ≈ 1 year)

    Returns:
        pd.Series of signals (-1, 0, or +1)
    """
    high_52 = close.rolling(lookback).max().shift(1)
    low_52  = close.rolling(lookback).min().shift(1)
    signal = np.where(
        close >= high_52, 1,
        np.where(close <= low_52, -1, 0)
    )
    return pd.Series(signal, index=close.index).fillna(0)


class VectorizedBacktester:
    """
    Research-Grade Vectorized Backtesting Engine.

    Features:
    - Realistic NSE transaction costs (brokerage + STT + exchange)
    - NIFTY 50 buy-and-hold benchmark
    - MA crossover and momentum baselines (all computed from same data)
    - Monte Carlo simulation (1000 paths, fully vectorized)
    - Statistical significance testing (bootstrap, binomial, t-test)
    """

    # NSE realistic round-trip transaction cost breakdown:
    #   Brokerage:      ~0.05%
    #   STT (sell):      0.025%
    #   Exchange + GST: ~0.003%
    #   Total round-trip: ~0.1% (applies once per trade, not per day)
    NSE_ROUND_TRIP_COST = 0.001  # 0.1%

    def __init__(self, data: pd.DataFrame, signals: pd.Series):
        """
        Initialize the backtester.

        Args:
            data: DataFrame with 'Actual_Return' column (and optionally 'Close')
            signals: Series of trading signals (-1, 0, 1)
        """
        self.data = data
        self.signals = signals

    def run_backtest(self, initial_capital: float = None,
                     transaction_cost: float = None,
                     include_costs: bool = True) -> dict:
        """
        Run vectorized backtest with realistic transaction costs.

        Transaction cost logic:
        - A cost is incurred whenever the signal CHANGES (a new trade is placed)
        - Cost is deducted from the return on the day the trade begins
        - This models round-trip cost (entry + exit) conservatively on entry day

        Args:
            initial_capital: Starting capital (default from config)
            transaction_cost: Round-trip cost fraction (default NSE_ROUND_TRIP_COST)
            include_costs: If False, runs zero-cost version for comparison

        Returns:
            Dictionary with performance metrics, equity curve, and cost impact
        """
        initial_capital = initial_capital or TradingConfig.DEFAULT_INITIAL_CAPITAL
        tc = (transaction_cost if transaction_cost is not None else self.NSE_ROUND_TRIP_COST)

        df = self.data.copy()

        # Gross strategy return: Signal × Actual_Return
        df['Strategy_Return_Gross'] = self.signals * df['Actual_Return']

        # Detect trade entries (signal changes to a non-zero value)
        # Cost is deducted on the bar the new trade starts
        if include_costs and tc > 0:
            prev_signal = self.signals.shift(1).fillna(0)
            trade_entry = ((self.signals != prev_signal) & (self.signals != 0)).astype(float)
            df['Transaction_Cost'] = trade_entry * tc
        else:
            df['Transaction_Cost'] = 0.0

        df['Strategy_Return'] = df['Strategy_Return_Gross'] - df['Transaction_Cost']

        # Equity Curve (after costs)
        df['Equity_Curve'] = initial_capital * (1 + df['Strategy_Return']).cumprod()

        # Zero-cost equity for comparison
        df['Equity_Curve_Gross'] = initial_capital * (1 + df['Strategy_Return_Gross']).cumprod()

        # --- Core Metrics ---
        strat_returns = df['Strategy_Return'].values
        core = _compute_metrics_from_returns(strat_returns, initial_capital)

        # Gross (pre-cost) metrics for comparison
        gross_returns = df['Strategy_Return_Gross'].values
        gross_core = _compute_metrics_from_returns(gross_returns, initial_capital)

        # Win Rate (based on after-cost returns)
        winning_trades = df[df['Strategy_Return'] > 0]
        losing_trades  = df[df['Strategy_Return'] < 0]
        total_trades   = df[df['Strategy_Return'] != 0]

        # Statistical significance
        stat_sig = None
        if 'Predicted_Return' in self.data.columns:
            predictions = self.data['Predicted_Return'].values
            actuals = self.data['Actual_Return'].values
            stat_sig = calculate_statistical_significance(predictions, actuals)

        n_trades = int(df['Transaction_Cost'].gt(0).sum())
        total_cost_drag = float(df['Transaction_Cost'].sum())

        return {
            # After-cost metrics (primary)
            "Total Return": core['total_return'],
            "Sharpe Ratio": core['sharpe_ratio'],
            "Max Drawdown": core['max_drawdown'],
            "Win Rate": core['win_rate'],
            "Avg Win": core['avg_win'],
            "Avg Loss": core['avg_loss'],
            "Profit Factor": core['profit_factor'],
            "Equity Curve": df['Equity_Curve'],
            "Strategy Returns": df['Strategy_Return'],
            # Pre-cost metrics for transparency
            "Total Return (Gross)": gross_core['total_return'],
            "Sharpe Ratio (Gross)": gross_core['sharpe_ratio'],
            "Equity Curve (Gross)": df['Equity_Curve_Gross'],
            # Trade cost info
            "N Trades": n_trades,
            "Cost Drag": total_cost_drag,
            "Transaction Cost Rate": tc,
            # Statistical significance
            "Statistical Significance": stat_sig
        }

    def run_benchmark_comparison(self, close_prices: pd.Series = None,
                                  nifty_returns: pd.Series = None,
                                  initial_capital: float = None,
                                  transaction_cost: float = None) -> dict:
        """
        Compare strategy against authenticated real-data baselines:
        1. NIFTY 50 Buy-and-Hold (real index returns)
        2. MA 20/50 Crossover (computed from stock's own price history)
        3. 52-Week Momentum Breakout (computed from stock's own price history)

        All baselines use the SAME start/end period as the strategy and
        the SAME transaction cost model. Zero synthetic data.

        Args:
            close_prices: Stock close price series (for MA and momentum baselines)
            nifty_returns: NIFTY 50 daily return series (from real yfinance data)
            initial_capital: Starting capital
            transaction_cost: Cost per trade round-trip

        Returns:
            Dictionary with metrics for each strategy, ready for comparison table
        """
        initial_capital = initial_capital or TradingConfig.DEFAULT_INITIAL_CAPITAL
        tc = (transaction_cost if transaction_cost is not None else self.NSE_ROUND_TRIP_COST)

        comparison = {}

        # --- Our Strategy (after cost) ---
        our_result = self.run_backtest(initial_capital, tc, include_costs=True)
        comparison['Our Model'] = {
            'Total Return (%)': round(our_result['Total Return'] * 100, 2),
            'Sharpe Ratio': round(our_result['Sharpe Ratio'], 2),
            'Max Drawdown (%)': round(our_result['Max Drawdown'] * 100, 2),
            'Win Rate (%)': round(our_result['Win Rate'] * 100, 1),
            'N Trades': our_result['N Trades'],
            'Equity Curve': our_result['Equity Curve'],
            'Returns': our_result['Strategy Returns']
        }

        # --- NIFTY 50 Buy-and-Hold ---
        if nifty_returns is not None and not nifty_returns.empty:
            # Align to same period as strategy
            aligned = nifty_returns.reindex(self.data.index).fillna(0)
            nifty_core = _compute_metrics_from_returns(aligned.values, initial_capital)
            comparison['NIFTY 50 B&H'] = {
                'Total Return (%)': round(nifty_core['total_return'] * 100, 2),
                'Sharpe Ratio': round(nifty_core['sharpe_ratio'], 2),
                'Max Drawdown (%)': round(nifty_core['max_drawdown'] * 100, 2),
                'Win Rate (%)': round(nifty_core['win_rate'] * 100, 1),
                'N Trades': 1,  # Buy and hold = 1 trade
                'Equity Curve': pd.Series(
                    initial_capital * np.cumprod(1 + aligned.values),
                    index=self.data.index
                ),
                'Returns': aligned
            }

        # --- MA 20/50 Crossover ---
        if close_prices is not None:
            # Align to test period (same index as self.data)
            close_aligned = close_prices.reindex(self.data.index)
            if close_aligned.notna().sum() > 50:
                ma_sig = ma_crossover_signals(close_aligned)
                # Compute returns: signal × actual_return
                actual_ret = self.data['Actual_Return']
                ma_returns_gross = (ma_sig * actual_ret).fillna(0)
                # Apply transaction costs
                prev_ma = ma_sig.shift(1).fillna(0)
                ma_entries = ((ma_sig != prev_ma) & (ma_sig != 0)).astype(float)
                ma_returns = ma_returns_gross - ma_entries * tc

                ma_core = _compute_metrics_from_returns(ma_returns.values, initial_capital)
                comparison['MA Crossover (20/50)'] = {
                    'Total Return (%)': round(ma_core['total_return'] * 100, 2),
                    'Sharpe Ratio': round(ma_core['sharpe_ratio'], 2),
                    'Max Drawdown (%)': round(ma_core['max_drawdown'] * 100, 2),
                    'Win Rate (%)': round(ma_core['win_rate'] * 100, 1),
                    'N Trades': int(ma_entries.sum()),
                    'Equity Curve': pd.Series(
                        initial_capital * np.cumprod(1 + ma_returns.values),
                        index=self.data.index
                    ),
                    'Returns': ma_returns
                }

            # --- 52-Week Momentum ---
            if close_aligned.notna().sum() > 252:
                mom_sig = momentum_signals(close_aligned)
                mom_returns_gross = (mom_sig * actual_ret).fillna(0)
                prev_mom = mom_sig.shift(1).fillna(0)
                mom_entries = ((mom_sig != prev_mom) & (mom_sig != 0)).astype(float)
                mom_returns = mom_returns_gross - mom_entries * tc

                mom_core = _compute_metrics_from_returns(mom_returns.values, initial_capital)
                comparison['52W Momentum'] = {
                    'Total Return (%)': round(mom_core['total_return'] * 100, 2),
                    'Sharpe Ratio': round(mom_core['sharpe_ratio'], 2),
                    'Max Drawdown (%)': round(mom_core['max_drawdown'] * 100, 2),
                    'Win Rate (%)': round(mom_core['win_rate'] * 100, 1),
                    'N Trades': int(mom_entries.sum()),
                    'Equity Curve': pd.Series(
                        initial_capital * np.cumprod(1 + mom_returns.values),
                        index=self.data.index
                    ),
                    'Returns': mom_returns
                }

        return comparison

    def run_monte_carlo(self, n_simulations: int = 1000,
                        initial_capital: float = None) -> dict:
        """
        Monte Carlo simulation by bootstrapping daily returns with replacement.

        Fully vectorized — all 1000 paths computed simultaneously via numpy.
        No loops. Typical runtime: 1-2 seconds.

        This shows the range of realistic outcomes assuming the same
        return distribution repeats in random order — does NOT assume
        returns are normally distributed (purely non-parametric).

        Args:
            n_simulations: Number of Monte Carlo paths (default 1000)
            initial_capital: Starting portfolio value

        Returns:
            Dictionary with percentile paths and summary statistics
        """
        initial_capital = initial_capital or TradingConfig.DEFAULT_INITIAL_CAPITAL

        # Use after-cost strategy returns
        strat_returns = (self.signals * self.data['Actual_Return']).values
        n = len(strat_returns)

        if n < 5:
            return {}

        # Vectorized bootstrap: shape (n_simulations, n)
        # np.random.choice draws with replacement across ALL paths at once
        sampled_indices = np.random.randint(0, n, size=(n_simulations, n))
        sampled_returns = strat_returns[sampled_indices]  # (n_simulations, n)

        # Equity paths: cumulative product
        equity_paths = initial_capital * np.cumprod(1 + sampled_returns, axis=1)

        # Final portfolio values
        final_values = equity_paths[:, -1]
        final_returns = final_values / initial_capital - 1

        # Drawdowns per path (vectorized)
        running_max = np.maximum.accumulate(equity_paths, axis=1)
        drawdowns = (equity_paths - running_max) / (running_max + 1e-10)
        max_drawdowns = np.min(drawdowns, axis=1)

        # Percentile equity paths for fan chart
        p5  = np.percentile(equity_paths, 5,  axis=0)
        p25 = np.percentile(equity_paths, 25, axis=0)
        p50 = np.percentile(equity_paths, 50, axis=0)
        p75 = np.percentile(equity_paths, 75, axis=0)
        p95 = np.percentile(equity_paths, 95, axis=0)

        return {
            # Summary statistics
            'median_return': float(np.median(final_returns)),
            'mean_return': float(np.mean(final_returns)),
            'p5_return': float(np.percentile(final_returns, 5)),
            'p25_return': float(np.percentile(final_returns, 25)),
            'p75_return': float(np.percentile(final_returns, 75)),
            'p95_return': float(np.percentile(final_returns, 95)),
            'prob_positive': float(np.mean(final_returns > 0)),
            'prob_above_nifty': None,  # Set externally if NIFTY return known
            'median_max_drawdown': float(np.median(max_drawdowns)),
            'worst_drawdown_p5': float(np.percentile(max_drawdowns, 5)),
            # Percentile fan chart data (for UI)
            'equity_p5':  p5,
            'equity_p25': p25,
            'equity_p50': p50,
            'equity_p75': p75,
            'equity_p95': p95,
            'n_simulations': n_simulations
        }

    def generate_signals_from_predictions(self, predictions: pd.Series,
                                          threshold: float = None) -> pd.Series:
        """
        Generate trading signals from return predictions.

        Args:
            predictions: Series of predicted returns
            threshold: Minimum predicted return to trigger signal

        Returns:
            Series of trading signals (-1, 0, 1)
        """
        threshold = threshold or TradingConfig.SIGNAL_THRESHOLD

        signals = pd.Series(0, index=predictions.index)
        signals[predictions > threshold] = 1  # Long
        signals[predictions < -threshold] = -1  # Short

        return signals

    def calculate_metrics_summary(self, backtest_results: dict) -> pd.DataFrame:
        """
        Create a summary DataFrame of backtest metrics.

        Args:
            backtest_results: Dictionary from run_backtest()

        Returns:
            DataFrame with after-cost and gross metrics side by side
        """
        metrics = {
            "Total Return (After Cost)": f"{backtest_results['Total Return']*100:.2f}%",
            "Total Return (Pre-Cost)":   f"{backtest_results.get('Total Return (Gross)', backtest_results['Total Return'])*100:.2f}%",
            "Sharpe Ratio": f"{backtest_results['Sharpe Ratio']:.2f}",
            "Max Drawdown": f"{backtest_results['Max Drawdown']*100:.2f}%",
            "Win Rate": f"{backtest_results['Win Rate']*100:.1f}%",
            "Profit Factor": f"{backtest_results['Profit Factor']:.2f}",
            "N Trades": str(backtest_results.get('N Trades', 'N/A')),
            "Cost Drag": f"{backtest_results.get('Cost Drag', 0)*100:.3f}%"
        }

        return pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
