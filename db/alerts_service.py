"""
Alerts evaluation (B2.5, framework mode) — invoked by the `alert_eval`
Celery beat task.

For each active alert, fetch the latest price (or `prob_up` from the most
recent prediction) and compare against the threshold. If crossed, mark
`triggered_at = NOW()` and flip `active = false` so it doesn't re-fire on
the next tick.

Notification (email / push) is NOT in scope here — that lands in B7. The
`triggered_at` timestamp is the signal the FE polls.

Returns a small summary dict so the Celery result is human-readable in
Flower / logs:

    {"checked": 12, "triggered": 2, "tickers_seen": 7}
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from db.supabase_client import get_admin_client

log = logging.getLogger(__name__)


def evaluate_active_alerts(price_fetcher=None) -> dict:
    fetcher = price_fetcher or _yfinance_last_price
    client = get_admin_client()

    active = (
        client.table("alerts")
        .select("id,user_id,ticker,kind,threshold")
        .eq("active", True)
        .execute()
        .data
        or []
    )

    triggered = 0
    checked = 0
    tickers_seen: set[str] = set()
    price_cache: dict[str, float | None] = {}
    prob_cache: dict[str, float | None] = {}

    for alert in active:
        checked += 1
        tickers_seen.add(alert["ticker"])

        tripped = False
        kind = alert["kind"]
        threshold = alert["threshold"]
        ticker = alert["ticker"]

        if kind in ("price_above", "price_below"):
            if ticker not in price_cache:
                price_cache[ticker] = fetcher(ticker)
            price = price_cache[ticker]
            if price is None:
                continue
            if kind == "price_above" and price >= threshold:
                tripped = True
            elif kind == "price_below" and price <= threshold:
                tripped = True

        elif kind in ("prob_up_above", "prob_up_below"):
            if ticker not in prob_cache:
                prob_cache[ticker] = _latest_prob_up(client, ticker, alert.get("user_id"))
            prob = prob_cache[ticker]
            if prob is None:
                continue
            if kind == "prob_up_above" and prob >= threshold:
                tripped = True
            elif kind == "prob_up_below" and prob <= threshold:
                tripped = True

        else:
            log.warning("alerts: unknown kind %r on alert id=%s", kind, alert["id"])
            continue

        if tripped:
            client.table("alerts").update(
                {"triggered_at": datetime.now(UTC).isoformat(), "active": False}
            ).eq("id", alert["id"]).execute()
            triggered += 1

    return {
        "checked":      checked,
        "triggered":    triggered,
        "tickers_seen": len(tickers_seen),
    }


def _latest_prob_up(client, ticker: str, user_id: str | None) -> float | None:
    q = (
        client.table("predictions")
        .select("prob_up")
        .eq("ticker", ticker)
        .not_.is_("prob_up", "null")
    )
    if user_id is not None:
        q = q.eq("user_id", user_id)
    rows = q.order("made_at", desc=True).limit(1).execute().data or []
    if not rows:
        return None
    val = rows[0].get("prob_up")
    return float(val) if val is not None else None


def _yfinance_last_price(ticker: str) -> float | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        hist = yf.Ticker(ticker).history(period="1d", interval="1m")
    except Exception:
        return None
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    last = hist["Close"].dropna()
    if last.empty:
        return None
    return float(last.iloc[-1])
