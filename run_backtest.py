import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

from services.backtester import WalkForwardBacktester

print("=== Starting Walk-Forward Backtest ===")
print("Universe: ~55 Nifty tickers, 2016-2026")
print("Design: 3y train / 6m test / 6m step, monthly rebalancing")
print()

bt = WalkForwardBacktester(start_year=2016, end_year=2026)
report = bt.run()

print()
print("=== BACKTEST RESULTS ===")
print()
print("Summary Statistics:")
for k, v in report.summary_stats.items():
    if isinstance(v, float):
        print(f"  {k:30s}: {v:.4f}")
    else:
        print(f"  {k:30s}: {v}")

print()
print("Walk-Forward Slices (per window):")
for i, sl in enumerate(report.walk_forward_slices):
    period   = f"{sl.test_start.date()} to {sl.test_end.date()}"
    sharpe   = sl.sharpe
    cagr     = sl.cagr * 100
    max_dd   = sl.max_drawdown * 100
    n_trades = sl.n_rebalances
    excess   = sl.excess_return * 100
    print(f"  Window {i+1:2d}: {period}  Sharpe={sharpe:.3f}  CAGR={cagr:.1f}%  MaxDD={max_dd:.1f}%  Excess={excess:+.1f}%  rebalances={n_trades}")

print()
if report.bootstrap_ci:
    print("Bootstrap 95% CI on Sharpe:")
    ci = report.bootstrap_ci
    lo = ci.get("lower", ci.get("lo", 0))
    hi = ci.get("upper", ci.get("hi", 0))
    med = ci.get("median", ci.get("med", 0))
    print(f"  lower:  {lo:.3f}")
    print(f"  upper:  {hi:.3f}")
    print(f"  median: {med:.3f}")

print()
print("Signal Attribution (avg IC per signal):")
for k, v in report.signal_attribution.items():
    print(f"  {k}: {v:.4f}")

print()
print("Equity curve (first and last 5 values):")
ec = report.equity_curve
if ec is not None and not ec.empty:
    print(ec.head(5).to_string())
    print("  ...")
    print(ec.tail(5).to_string())
