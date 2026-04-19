"""
Tick publisher process (B5.2).

Long-running producer that polls a quote source and publishes to Redis
Pub/Sub channels `ticks:{SYMBOL}`. Meant to be run as a separate process
alongside the FastAPI server + Celery worker:

    python -m workers.tick_publisher --tickers RELIANCE.NS,TCS.NS --interval 15

Symbol list sources (in order of precedence):
  1. `--tickers` CLI flag (comma-separated)
  2. `PROTRADER_TICK_SYMBOLS` env var
  3. `watchlist_tickers` table (if Supabase service role key is present)
  4. `config.settings.DataConfig.DEFAULT_STOCKS`

Source backend is currently yfinance (`_last_close`). When Upstox KYC
clears, swap `_poll_once` for an Upstox WS listener — the publish side
stays identical (publish a :class:`Tick` per symbol).
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import UTC, datetime

from workers.tick_broker import Tick, is_available, publish

log = logging.getLogger(__name__)


def _resolve_symbols(cli: str | None) -> list[str]:
    if cli:
        return [s.strip().upper() for s in cli.split(",") if s.strip()]
    env = os.environ.get("PROTRADER_TICK_SYMBOLS")
    if env:
        return [s.strip().upper() for s in env.split(",") if s.strip()]
    try:
        from db import supabase_client as sb

        if sb.has_service_role():
            client = sb.get_admin_client()
            rows = client.table("watchlist_tickers").select("ticker").execute()
            symbols = sorted({(r.get("ticker") or "").upper() for r in (rows.data or []) if r.get("ticker")})
            if symbols:
                return symbols
    except Exception as e:  # noqa: BLE001
        log.warning("watchlist fetch failed, falling back to defaults: %s", e)
    from config.settings import DataConfig

    return list(DataConfig.DEFAULT_STOCKS)


def _last_close(ticker: str) -> float | None:
    """yfinance best-effort snapshot. Minutely bar → daily close fallback."""
    try:
        import yfinance as yf

        intraday = yf.Ticker(ticker).history(period="1d", interval="1m")
        if intraday is not None and not intraday.empty:
            return float(intraday["Close"].iloc[-1])
        daily = yf.Ticker(ticker).history(period="5d")
        if daily is not None and not daily.empty:
            return float(daily["Close"].iloc[-1])
    except Exception as e:  # noqa: BLE001
        log.debug("yfinance fetch failed for %s: %s", ticker, e)
    return None


def _upstox_quote(ticker: str) -> float | None:
    """Upstox REST LTP when configured; `None` lets callers fall through."""
    try:
        from data.upstox_client import get_ltp

        return get_ltp(ticker)
    except Exception as e:  # noqa: BLE001
        log.debug("upstox LTP fetch failed for %s: %s", ticker, e)
        return None


def _fetch_quote(ticker: str) -> tuple[float | None, str]:
    """Try Upstox first (if configured) then yfinance. Returns (price, source)."""
    price = _upstox_quote(ticker)
    if price is not None and price > 0:
        return price, "upstox"
    return _last_close(ticker), "yfinance"


def _poll_once(symbols: list[str]) -> int:
    published = 0
    now = datetime.now(UTC).isoformat()
    for sym in symbols:
        price, source = _fetch_quote(sym)
        if price is None:
            continue
        publish(Tick(ticker=sym, ts=now, price=price, source=source))
        published += 1
    return published


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ProTrader tick publisher")
    parser.add_argument("--tickers", help="Comma-separated symbols (overrides env/db).")
    parser.add_argument("--interval", type=float, default=15.0, help="Poll interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Poll exactly once then exit.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not is_available():
        log.error("REDIS_URL not configured — publisher would fan out to /dev/null.")
        return 2

    symbols = _resolve_symbols(args.tickers)
    log.info("publisher starting: %d symbols, interval=%.1fs", len(symbols), args.interval)

    stop = {"flag": False}

    def _sigterm(*_: object) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sigterm)
    signal.signal(signal.SIGTERM, _sigterm)

    while not stop["flag"]:
        t0 = time.monotonic()
        try:
            n = _poll_once(symbols)
            log.info("tick round: published=%d/%d", n, len(symbols))
        except Exception as e:  # noqa: BLE001
            log.exception("poll round failed: %s", e)
        if args.once:
            break
        elapsed = time.monotonic() - t0
        sleep_for = max(0.0, args.interval - elapsed)
        # Break the sleep into 1 s increments so ctrl-C is responsive.
        slept = 0.0
        while slept < sleep_for and not stop["flag"]:
            time.sleep(min(1.0, sleep_for - slept))
            slept += 1.0

    log.info("publisher stopped cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
