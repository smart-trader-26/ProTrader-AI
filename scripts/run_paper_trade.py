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
    stop_pct: float | None = None,
    target_pct: float | None = None,
    qty: int = 1,
    dry_run: bool = False,
    engine: str = "auto",
    max_hold_days: int | None = None,
) -> dict:
    """
    One paper-trade cycle: predict → signal → mark-to-market.

    `engine` picks which model output is traded:
      • "swing"   — the 30-day directional signal (the project's only
                    OOS-validated edge). Bracketless entry, disaster stop,
                    time exit at the signal horizon.
      • "nextday" — legacy next-day hybrid prob vs Youden-J τ (documented
                    as no-edge; kept for comparison runs).
      • "auto"    — swing when the swing model is trained, else nextday.

    Returns a summary dict for logging / Celery result storage.
    """
    from services.paper_trade_service import PaperTradeService

    svc = PaperTradeService(db_path=db_path)

    signals: list[dict] = []
    errors: list[str] = []
    engines_used: set[str] = set()
    swing_horizon_days: int | None = None

    for ticker in tickers:
        t0 = time.monotonic()
        try:
            sig = _predict_signal(ticker, engine=engine)
            elapsed = time.monotonic() - t0
            prob_up, threshold = sig["prob_up"], sig["threshold"]
            engines_used.add(sig["engine"])
            signal = {
                "ticker": ticker,
                "engine": sig["engine"],
                "prob_up": round(prob_up, 4),
                "threshold": round(threshold, 4),
                "action": "BUY" if prob_up > threshold else "HOLD/SELL",
                "elapsed_s": round(elapsed, 1),
            }
            signals.append(signal)
            log.info(
                "  %s  [%s]  prob_up=%.3f  τ=%.3f  → %s  (%.1fs)",
                ticker, sig["engine"], prob_up, threshold, signal["action"], elapsed,
            )

            if sig["engine"] == "swing30d":
                # The swing edge was measured as "buy, hold ~30 trading days,
                # count if higher" — no profit target (a target would cap the
                # very wins the stat is built on), only a disaster stop, and a
                # time exit at the horizon (handled in mark_to_market below).
                tk_stop = stop_pct if stop_pct is not None else 0.10
                tk_target = target_pct  # default None — hold to horizon
                if swing_horizon_days is None:
                    # trading days → calendar days (~7/5)
                    swing_horizon_days = int(sig["horizon_days"] * 7 / 5)
            else:
                tk_stop = stop_pct if stop_pct is not None else 0.02
                tk_target = target_pct if target_pct is not None else 0.04

            if not dry_run:
                svc.on_signal(
                    ticker,
                    prob_up=prob_up,
                    threshold=threshold,
                    qty=qty,
                    stop_pct=tk_stop,
                    target_pct=tk_target,
                    reason=f"paper_trade_cycle[{sig['engine']}]",
                )
        except Exception as e:  # noqa: BLE001
            log.warning("  %s  FAILED: %s", ticker, e)
            errors.append(f"{ticker}: {e}")

    # Mark-to-market: close positions whose stop/target/holding-period was hit
    closed = []
    if not dry_run:
        hold_limit = max_hold_days if max_hold_days is not None else swing_horizon_days
        closed = svc.mark_to_market(max_holding_days=hold_limit)
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
        "engines": ",".join(sorted(engines_used)) or "none",
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


def _predict_signal(ticker: str, engine: str = "auto") -> dict:
    """
    Run the prediction pipeline and return the tradeable signal:
    {prob_up, threshold, engine, horizon_days}.

    Engine resolution:
      • "swing" / "auto" with a trained swing model → the bundle's 30-day
        swing signal: calibrated prob vs its walk-forward-tuned τ*. This is
        the only signal in the project with a measured OOS edge.
      • otherwise → next-day path: day-0 calibrated blended prob (the actual
        call the ledger resolves — NOT the horizon-end MC fraction) vs the
        per-ticker Youden-J τ from ``threshold_tuning.tau_star``.
    """
    from services.prediction_service import predict

    bundle = predict(ticker)

    sw = bundle.swing_signal
    if engine in ("auto", "swing") and sw is not None and sw.trained:
        return {
            "prob_up": float(sw.prob_up),
            "threshold": float(sw.tau_star or 0.63),
            "engine": "swing30d",
            "horizon_days": int(sw.horizon_days),
        }
    if engine == "swing":
        raise RuntimeError(
            f"{ticker}: swing engine requested but the swing model is not trained"
        )

    # Resolve prob_up (0..1 scale) — day-0 is the real calibrated call.
    if bundle.points:
        prob_up = bundle.points[0].prob_up
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

    return {
        "prob_up": float(prob_up if prob_up is not None else 0.5),
        "threshold": float(threshold),
        "engine": "nextday",
        "horizon_days": 1,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ProTrader paper-trade runner (A9.3)")
    parser.add_argument("--tickers", help="Comma-separated tickers (overrides env/defaults)")
    parser.add_argument("--db", type=Path, help="SQLite DB path (default: shared ledger)")
    parser.add_argument(
        "--engine", choices=("auto", "swing", "nextday"), default="auto",
        help="Signal engine: swing = 30-day directional signal (the validated edge), "
             "nextday = legacy next-day prob, auto = swing when trained (default)",
    )
    parser.add_argument("--stop-pct", type=float, default=None,
                        help="Stop-loss %% (default: 10%% swing / 2%% nextday)")
    parser.add_argument("--target-pct", type=float, default=None,
                        help="Target %% (default: none for swing / 4%% nextday)")
    parser.add_argument("--max-hold-days", type=int, default=None,
                        help="Calendar-day time exit (default: swing horizon ~42d)")
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

    log.info("paper-trade cycle: %d tickers, engine=%s, dry_run=%s",
             len(tickers), args.engine, args.dry_run)
    summary = run_cycle(
        tickers=tickers,
        db_path=args.db,
        stop_pct=args.stop_pct,
        target_pct=args.target_pct,
        qty=args.qty,
        dry_run=args.dry_run,
        engine=args.engine,
        max_hold_days=args.max_hold_days,
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
