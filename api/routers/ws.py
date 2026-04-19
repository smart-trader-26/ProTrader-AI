"""
WebSocket prices (B5.3).

  WS /api/v1/ws/prices?tickers=RELIANCE.NS,TCS.NS

Redis-backed when `REDIS_URL` is set: subscribes to `ticks:{SYMBOL}` and
relays frames as the tick publisher emits them. That lets multiple API
replicas share one upstream feed and decouples polling cadence from the
HTTP layer. See [workers/tick_broker.py](../../workers/tick_broker.py) +
[workers/tick_publisher.py](../../workers/tick_publisher.py).

Falls back to in-process yfinance polling when Redis is not configured —
handy for local dev without running the full worker stack.

Wire format (one frame per tick, JSON):
  {"ticker": "RELIANCE.NS", "ts": "2026-04-19T09:42:00Z", "price": 2954.6, "source": "yfinance"}
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from workers import tick_broker

log = logging.getLogger(__name__)
router = APIRouter(tags=["websocket"])

POLL_INTERVAL_SECONDS = 15.0
MAX_TICKERS = 25


@router.websocket("/ws/prices")
async def ws_prices(
    websocket: WebSocket,
    tickers: str = Query(default="", description="Comma-separated tickers"),
) -> None:
    await websocket.accept()
    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()][:MAX_TICKERS]
    if not symbols:
        await websocket.send_json({"error": "no tickers — add ?tickers=RELIANCE.NS,TCS.NS"})
        await websocket.close(code=1003)
        return

    source = "redis" if tick_broker.is_available() else "poll"
    await websocket.send_json({"event": "subscribed", "tickers": symbols, "source": source})

    try:
        if source == "redis":
            await _relay_from_broker(websocket, symbols)
        else:
            await _relay_from_poll(websocket, symbols)
    except WebSocketDisconnect:
        return
    except Exception as e:  # noqa: BLE001
        log.warning("ws_prices aborted: %s", e)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


async def _relay_from_broker(websocket: WebSocket, symbols: list[str]) -> None:
    """Stream ticks from Redis Pub/Sub to the client until disconnect."""
    async for tick in tick_broker.subscribe(symbols):
        await websocket.send_json(
            {"ticker": tick.ticker, "ts": tick.ts, "price": tick.price, "source": tick.source}
        )


async def _relay_from_poll(websocket: WebSocket, symbols: list[str]) -> None:
    """In-process fallback: yfinance poll loop, same wire format."""
    while True:
        for sym in symbols:
            price = _last_close(sym)
            if price is None:
                continue
            await websocket.send_json(
                {
                    "ticker": sym,
                    "ts": datetime.now(UTC).isoformat(),
                    "price": price,
                    "source": "yfinance",
                }
            )
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


def _last_close(ticker: str) -> float | None:
    """Best-effort: 1-min bar from yfinance, fall back to daily close."""
    try:
        import yfinance as yf

        intraday = yf.Ticker(ticker).history(period="1d", interval="1m")
        if intraday is not None and not intraday.empty:
            return float(intraday["Close"].iloc[-1])
        daily = yf.Ticker(ticker).history(period="5d")
        if daily is not None and not daily.empty:
            return float(daily["Close"].iloc[-1])
    except Exception:
        return None
    return None
