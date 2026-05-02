"""
Redis Pub/Sub tick broker (B5.1 / B5.2).

One upstream quote feed (yfinance today, Upstox WS once KYC clears) writes
ticks to Redis channels `ticks:{SYMBOL}`; any number of WebSocket handlers
subscribe and fan them out to connected browsers. Decouples the polling
cadence from the HTTP layer, so:

  • multiple FastAPI replicas share one upstream connection
  • a slow consumer can never back-pressure the publisher
  • swapping yfinance for Upstox is a one-file edit (see tick_publisher.py)

Channel format: `ticks:RELIANCE.NS`. Payload is JSON:
  {"ticker": "RELIANCE.NS", "ts": "2026-04-19T09:42:00Z", "price": 2954.6, "source": "yfinance"}

Gracefully degrades: if `REDIS_URL` is unset, `is_available()` → False and
the ws router falls back to direct yfinance polling.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from config.settings import REDIS_URL

log = logging.getLogger(__name__)

CHANNEL_PREFIX = "ticks:"


def is_available() -> bool:
    return bool(REDIS_URL)


def channel_for(symbol: str) -> str:
    return f"{CHANNEL_PREFIX}{symbol.upper()}"


@dataclass
class Tick:
    ticker: str
    ts: str  # ISO-8601, UTC
    price: float
    source: str = "yfinance"

    def to_json(self) -> str:
        return json.dumps(self.__dict__, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "Tick":
        data: dict[str, Any] = json.loads(raw)
        return cls(
            ticker=data["ticker"],
            ts=data["ts"],
            price=float(data["price"]),
            source=data.get("source", "unknown"),
        )


def publish(tick: Tick) -> int:
    """Publish synchronously. Returns the subscriber count Redis reports. Reuses a process-global connection."""
    r = _sync_redis()
    return int(r.publish(channel_for(tick.ticker), tick.to_json()))


_SYNC_R: Any | None = None


def _sync_redis():
    global _SYNC_R
    if _SYNC_R is None:
        import redis  # late import so the module stays importable without redis installed

        if not REDIS_URL:
            raise RuntimeError("REDIS_URL not set — tick broker unavailable")
        _SYNC_R = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _SYNC_R


async def subscribe(symbols: list[str]) -> AsyncIterator[Tick]:
    """Yield ticks for the given symbols. Caller wraps in try/finally to unsubscribe on disconnect."""
    if not REDIS_URL:
        raise RuntimeError("REDIS_URL not set — tick broker unavailable")

    from redis import asyncio as aioredis  # late import

    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    pubsub = r.pubsub()
    channels = [channel_for(s) for s in symbols if s]
    if not channels:
        return
    try:
        await pubsub.subscribe(*channels)
        # get_message with a short timeout keeps us cancellation-responsive;
        # listen() blocks and doesn't play well with asyncio cancellation.
        while True:
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if msg is None:
                await asyncio.sleep(0)  # yield control
                continue
            data = msg.get("data")
            if not data:
                continue
            try:
                yield Tick.from_json(data)
            except Exception as e:  # noqa: BLE001
                log.warning("bad tick payload on %s: %s", msg.get("channel"), e)
    finally:
        try:
            await pubsub.unsubscribe(*channels)
            await pubsub.close()
            await r.aclose()
        except Exception:  # noqa: BLE001
            pass
