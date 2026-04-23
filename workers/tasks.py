"""
Task registry (B2.2) — paired sync + Celery implementations.

Each `kind` in `TASK_REGISTRY` carries:
  • `sync_fn`     : a plain Python callable for the in-process JobStore
  • `celery_task` : the registered Celery task for the Redis-backed JobStore

Both halves do the same work but the Celery side returns JSON-friendly
dicts (so the result lands in Redis cleanly) while the sync side returns
the live Pydantic object (and the API serializes on the way out).

Adding a new task = define `_xxx_sync`, register a `@app.task`, append a
`TaskPair` to the registry. The router just calls
`store.enqueue("xxx", **kwargs)` — it never names the function.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from workers.celery_app import app

log = logging.getLogger(__name__)


@dataclass
class TaskPair:
    sync_fn: Callable[..., Any]
    celery_task: Any  # celery.app.task.Task

    def to_jsonable(self, result: Any) -> Any:
        """Pydantic → JSON dict; passthrough otherwise."""
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        return result


# ───────────────────────── prediction ────────────────────────────────

def _predict_sync(**kwargs):
    from services import prediction_service

    return prediction_service.predict(**kwargs)


@app.task(name="protrader.predict")
def predict_task(**kwargs):
    return _to_json(_predict_sync(**kwargs))


# ───────────────────────── backtest ──────────────────────────────────

def _backtest_sync(**kwargs):
    from services import backtest_service

    return backtest_service.run_backtest(**kwargs)


@app.task(name="protrader.backtest")
def backtest_task(**kwargs):
    return _to_json(_backtest_sync(**kwargs))


# ───────────────────────── ledger backfill ───────────────────────────

def _ledger_backfill_sync(**kwargs):
    from services import ledger_service

    return {"resolved": ledger_service.backfill_actuals(**kwargs)}


@app.task(name="protrader.ledger_backfill")
def ledger_backfill_task(**kwargs):
    return _ledger_backfill_sync(**kwargs)


# ───────────────────────── news refresh (placeholder) ────────────────

def _news_refresh_sync(**kwargs):
    """
    Pre-warm the FinBERT cache for the most-watched tickers so the next
    user request sees the headlines + sentiment already in cache.
    Implementation light for now — real top-watchlist query lands with B3.3.
    """
    from data.news_sentiment import get_news

    tickers = kwargs.get("tickers") or ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    refreshed = 0
    for t in tickers:
        try:
            news = get_news(t) or []
            refreshed += len(news)
        except Exception as e:  # noqa: BLE001
            log.warning("news refresh failed for %s: %s", t, e)
    return {"refreshed_articles": refreshed, "tickers": tickers}


@app.task(name="protrader.news_refresh")
def news_refresh_task(**kwargs):
    return _news_refresh_sync(**kwargs)


# ───────────────────────── alert evaluation ──────────────────────────

def _alert_eval_sync(**kwargs):
    """
    Evaluate every active alert against current price. Trips alerts whose
    threshold was crossed since the last run.

    Defers to db.alerts_service when Supabase is configured; no-op otherwise.
    """
    from db import supabase_client as sb

    if not sb.has_service_role():
        return {"checked": 0, "triggered": 0, "reason": "no SUPABASE_SERVICE_ROLE_KEY"}

    from db.alerts_service import evaluate_active_alerts

    return evaluate_active_alerts()


@app.task(name="protrader.alert_eval")
def alert_eval_task(**kwargs):
    return _alert_eval_sync(**kwargs)


# ───────────────────────── paper trading ───────────────────────────

def _paper_trade_sync(**kwargs):
    """
    Run the daily paper-trade loop across the universe.
    """
    from scripts.run_paper_trade import run_cycle, _default_tickers
    
    tickers = kwargs.get("tickers") or _default_tickers()
    dry_run = kwargs.get("dry_run", False)
    
    return run_cycle(tickers, dry_run=dry_run)


@app.task(name="protrader.paper_trade")
def paper_trade_task(**kwargs):
    return _to_json(_paper_trade_sync(**kwargs))


# ───────────────────────── registry ──────────────────────────────────

TASK_REGISTRY: dict[str, TaskPair] = {
    "predict":         TaskPair(sync_fn=_predict_sync,         celery_task=predict_task),
    "backtest":        TaskPair(sync_fn=_backtest_sync,        celery_task=backtest_task),
    "ledger_backfill": TaskPair(sync_fn=_ledger_backfill_sync, celery_task=ledger_backfill_task),
    "news_refresh":    TaskPair(sync_fn=_news_refresh_sync,    celery_task=news_refresh_task),
    "alert_eval":      TaskPair(sync_fn=_alert_eval_sync,      celery_task=alert_eval_task),
    "paper_trade":     TaskPair(sync_fn=_paper_trade_sync,     celery_task=paper_trade_task),
}

def _to_json(result):
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return result
