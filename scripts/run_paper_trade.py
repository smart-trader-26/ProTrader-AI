"""
Daily paper-trade runner (A9.3).

Runs the model's prediction pipeline for each ticker in a universe, feeds
the signals into :class:`PaperTradeService`, and prints a book summary.

Usage:
    # Manual run (e.g. at market close ~15:45 IST)
    python -m scripts.run_paper_trade

    # Override universe + params
    python -m scripts.run_paper_trade --tickers RELIANCE.NS,TCS.NS,INFY.NS

    # Dry-run: predict + log signals, but don't actually open/close positions
    python -m scripts.run_paper_trade --dry-run

After 30 trading days the accuracy tab + book_state() provide the reality
check described in A9.3 — compare paper P&L vs. backtest numbers.

This script can also be registered as a Celery beat task (runs at 15:45 IST)
via workers/beat_schedule.py.
"""

from __future__ import annotations

# ── Suppress noisy third-party warnings before any imports touch them ──
import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")          # TF C++ info/warn
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")         # oneDNN notice
warnings.filterwarnings("ignore", category=FutureWarning)    # numpy/pandas/sklearn
warnings.filterwarnings("ignore", category=UserWarning,
                        module=r"sklearn")                   # version/feature-name
warnings.filterwarnings("ignore", category=UserWarning,
                        module=r"pickle")                    # xgboost pickle
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import logging
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)


def _default_tickers() -> list[str]:
    """
    Resolve the default ticker universe.

    Priority:
      1. PROTRADER_PAPER_TICKERS env var (comma-separated)
      2. DataConfig.DEFAULT_STOCKS converted to .NS suffix
    """
    env = os.environ.get("PROTRADER_PAPER_TICKERS", "")
    if env:
        return [t.strip().upper() for t in env.split(",") if t.strip()]
    from config.settings import DataConfig

    stocks = list(DataConfig.DEFAULT_STOCKS)
    return [s if s.endswith(".NS") else f"{s}.NS" for s in stocks]


def run_cycle(
    tickers: list[str],
    db_path: Path | None = None,
    stop_pct: float = 0.02,
    target_pct: float = 0.04,
    qty: int = 1,
    dry_run: bool = False,
) -> dict:
    """
    One paper-trade cycle: predict → signal → mark-to-market.

    Returns a summary dict for logging / Celery result storage.
    """
    from services.paper_trade_service import PaperTradeService

    svc = PaperTradeService(db_path=db_path)

    signals: list[dict] = []
    errors: list[str] = []

    for ticker in tickers:
        t0 = time.monotonic()
        try:
            prob_up, threshold = _predict_signal(ticker)
            elapsed = time.monotonic() - t0
            signal = {
                "ticker": ticker,
                "prob_up": round(prob_up, 4),
                "threshold": round(threshold, 4),
                "action": "BUY" if prob_up > threshold else "HOLD/SELL",
                "elapsed_s": round(elapsed, 1),
            }
            signals.append(signal)
            log.info(
                "  %s  prob_up=%.3f  τ=%.3f  → %s  (%.1fs)",
                ticker, prob_up, threshold, signal["action"], elapsed,
            )

            if not dry_run:
                svc.on_signal(
                    ticker,
                    prob_up=prob_up,
                    threshold=threshold,
                    qty=qty,
                    stop_pct=stop_pct,
                    target_pct=target_pct,
                    reason=f"paper_trade_cycle",
                )
        except Exception as e:  # noqa: BLE001
            log.warning("  %s  FAILED: %s", ticker, e)
            errors.append(f"{ticker}: {e}")

    # Mark-to-market: close positions whose stop/target was hit
    closed = []
    if not dry_run:
        closed = svc.mark_to_market()
        for fill in closed:
            log.info(
                "  MTM closed %s: entry=%.2f exit=%.2f net_pnl=%.2f (%s)",
                fill.ticker, fill.entry_price, fill.exit_price,
                fill.net_pnl, fill.reason_exit,
            )

    # Book summary
    state = svc.book_state()
    summary = {
        "tickers_processed": len(signals),
        "errors": len(errors),
        "positions_closed_mtm": len(closed),
        "open_positions": state.n_open,
        "total_fills": state.n_fills,
        "realised_pnl": round(state.realised_pnl, 2),
        "unrealised_pnl": round(state.unrealised_pnl, 2),
        "equity": round(state.equity, 2),
        "dry_run": dry_run,
    }
    log.info("Book state: %s", summary)
    return summary


def _predict_signal(ticker: str) -> tuple[float, float]:
    """
    Run the prediction pipeline and return (prob_up, threshold).

    ``PredictionBundle`` stores probabilities as 0-100 percentages in
    ``last_directional_prob`` and per-point ``prob_up`` (0-1). The
    threshold lives in ``threshold_tuning.tau_star`` (Youden-J optimal,
    0-1).  Falls back to 0.55 if the model didn't compute one.
    """
    from services.prediction_service import predict

    bundle = predict(ticker)

    # Resolve prob_up (0..1 scale)
    if bundle.points:
        prob_up = bundle.points[-1].prob_up      # already 0..1
    elif bundle.last_directional_prob is not None:
        prob_up = bundle.last_directional_prob / 100.0
    elif bundle.avg_directional_prob is not None:
        prob_up = bundle.avg_directional_prob / 100.0
    else:
        prob_up = 0.5

    # Resolve threshold (τ*) — per-ticker Youden-J optimal
    threshold = 0.55  # safe default
    if bundle.threshold_tuning is not None and bundle.threshold_tuning.tau_star:
        threshold = bundle.threshold_tuning.tau_star

    return prob_up, threshold


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ProTrader paper-trade runner (A9.3)")
    parser.add_argument("--tickers", help="Comma-separated tickers (overrides env/defaults)")
    parser.add_argument("--db", type=Path, help="SQLite DB path (default: shared ledger)")
    parser.add_argument("--stop-pct", type=float, default=0.02, help="Stop-loss %% (default 2%%)")
    parser.add_argument("--target-pct", type=float, default=0.04, help="Target %% (default 4%%)")
    parser.add_argument("--qty", type=int, default=1, help="Shares per signal (default 1)")
    parser.add_argument("--dry-run", action="store_true", help="Predict only, no trades")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else _default_tickers()
    )

    log.info("paper-trade cycle: %d tickers, dry_run=%s", len(tickers), args.dry_run)
    summary = run_cycle(
        tickers=tickers,
        db_path=args.db,
        stop_pct=args.stop_pct,
        target_pct=args.target_pct,
        qty=args.qty,
        dry_run=args.dry_run,
    )

    print(f"\n{'='*50}")
    print("Paper-Trade Cycle Summary")
    print(f"{'='*50}")
    for k, v in summary.items():
        print(f"  {k:.<30} {v}")
    print(f"{'='*50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
