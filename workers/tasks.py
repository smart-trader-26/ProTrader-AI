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
        """Pydantic model → JSON dict; passthrough for plain types."""
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        return result



def _predict_sync(**kwargs):
    from services import prediction_service
    import pandas as pd

    # Convert fii_dii_payload (list of dicts) back to a DataFrame.
    fii_dii_payload = kwargs.pop("fii_dii_payload", None)
    fii_dii_df: pd.DataFrame | None = None
    if fii_dii_payload:
        try:
            fii_dii_df = pd.DataFrame(fii_dii_payload)
            fii_dii_df["date"] = pd.to_datetime(fii_dii_df["date"])
            fii_dii_df = fii_dii_df.set_index("date").sort_index()

            col_map = {
                "fii_buy":  "FII_Buy_Value",
                "fii_sell": "FII_Sell_Value",
                "fii_net":  "FII_Net",
                "dii_buy":  "DII_Buy_Value",
                "dii_sell": "DII_Sell_Value",
                "dii_net":  "DII_Net",
            }
            fii_dii_df.rename(columns=col_map, inplace=True)

            if "FII_Net" in fii_dii_df.columns and "FII_Cumulative" not in fii_dii_df.columns:
                fii_dii_df["FII_Cumulative"] = fii_dii_df["FII_Net"].cumsum()
            if "DII_Net" in fii_dii_df.columns and "DII_Cumulative" not in fii_dii_df.columns:
                fii_dii_df["DII_Cumulative"] = fii_dii_df["DII_Net"].cumsum()

            if "FII_Net" not in fii_dii_df.columns or "DII_Net" not in fii_dii_df.columns:
                fii_dii_df = None
        except Exception:
            fii_dii_df = None

    # Extract LLM sentiment scores collected by the user BEFORE prediction ran.
    # Map to the sentiment_features keys that hybrid_model.py recognises.
    llm_sentiment_signal = kwargs.pop("llm_sentiment_signal", None)
    llm_mean_sentiment   = kwargs.pop("llm_mean_sentiment",   None)
    llm_mean_materiality = kwargs.pop("llm_mean_materiality", None)
    kwargs.pop("llm_top_headline", None)  # informational only

    sentiment_features: dict | None = None
    if llm_mean_sentiment is not None:
        sentiment_features = {
            "Sentiment":            float(llm_mean_sentiment),     # Feature 19 — primary sentiment
            "Sentiment_Confidence": float(llm_mean_materiality or 0.5),  # Feature 21 — materiality as confidence proxy
        }
        if llm_sentiment_signal is not None:
            sentiment_features["Multi_Sentiment"] = float(llm_sentiment_signal)  # Feature 20

    return prediction_service.predict(
        fii_dii_data=fii_dii_df,
        sentiment_features=sentiment_features,
        **kwargs,
    )



@app.task(name="protrader.predict")
def predict_task(**kwargs):
    return _to_json(_predict_sync(**kwargs))



def _backtest_sync(**kwargs):
    from services import backtest_service

    return backtest_service.run_backtest(**kwargs)


@app.task(name="protrader.backtest")
def backtest_task(**kwargs):
    return _to_json(_backtest_sync(**kwargs))



def _ledger_backfill_sync(**kwargs):
    from services import ledger_service

    return {"resolved": ledger_service.backfill_actuals(**kwargs)}


@app.task(name="protrader.ledger_backfill")
def ledger_backfill_task(**kwargs):
    return _ledger_backfill_sync(**kwargs)



def _news_refresh_sync(**kwargs):
    """Pre-warm the FinBERT cache for the most-watched tickers."""
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



def _alert_eval_sync(**kwargs):
    """Evaluate active alerts against current price; no-op when Supabase is unconfigured."""
    from db import supabase_client as sb

    if not sb.has_service_role():
        return {"checked": 0, "triggered": 0, "reason": "no SUPABASE_SERVICE_ROLE_KEY"}

    from db.alerts_service import evaluate_active_alerts

    return evaluate_active_alerts()


@app.task(name="protrader.alert_eval")
def alert_eval_task(**kwargs):
    return _alert_eval_sync(**kwargs)



def _paper_trade_sync(**kwargs):
    from scripts.run_paper_trade import run_cycle, _default_tickers
    
    tickers = kwargs.get("tickers") or _default_tickers()
    dry_run = kwargs.get("dry_run", False)
    
    return run_cycle(tickers, dry_run=dry_run)


@app.task(name="protrader.paper_trade")
def paper_trade_task(**kwargs):
    return _to_json(_paper_trade_sync(**kwargs))


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
