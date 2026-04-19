"""
WebSocket prices stub (B5.3 stub — full pipeline lands with B5.1 / B5.2).

  WS /api/v1/ws/prices?tickers=RELIANCE.NS,TCS.NS

Today: polls yfinance every 15s and pushes a snapshot per ticker. Latency
is bad (~30s end-to-end with the polling delay); good enough to validate
the wire format end-to-end before swapping in the Redis Pub/Sub bridge in
B5.3 proper.

Wire format (one frame per ticker, JSON):
  { "ticker": "RELIANCE.NS", "ts": "2026-04-18T09:42:00Z", "price": 2954.6 }
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["websocket"])

POLL_INTERVAL_SECONDS = 15.0
MAX_TICKERS = 25


@router.websocket("/ws/prices")
async def ws_prices(
    websocket: WebSocket,
    tickers: str = Query(default="", description="Comma-separated tickers"),
) -> None:
    await websocket.accept()
    symbols = [t.strip() for t in tickers.split(",") if t.strip()][:MAX_TICKERS]
    if not symbols:
        await websocket.send_json({"error": "no tickers — add ?tickers=RELIANCE.NS,TCS.NS"})
        await websocket.close(code=1003)
        return

    await websocket.send_json({"event": "subscribed", "tickers": symbols})

    try:
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
                    }
                )
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
    except WebSocketDisconnect:
        return


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
