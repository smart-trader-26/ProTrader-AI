"""
Yahoo Finance unofficial WebSocket streamer (A3.3).

Endpoint:
    wss://streamer.finance.yahoo.com/?version=2

This is an **undocumented** Yahoo feed. No auth, no paid tier — you send
a JSON subscribe message with ticker symbols and Yahoo pushes protobuf-
encoded price updates back. Latency is ~30 s (better than yfinance's ~15
min delay but worse than a paid broker WS).

⚠️ Reliability caveats:
    - Yahoo occasionally rotates the protocol — if decode fails, the
      callback simply gets an empty dict. Do not rely on this for live
      trading.
    - Some ISPs / regions block the WS (notably Jio in India has been
      flaky). Test from your target network before wiring into the UI.
    - Yahoo can soft-ban an IP if you reconnect too aggressively. Default
      backoff is exponential starting at 2 s.

For production-grade ticks use A3.4 (Dhan / Upstox WebSocket) — this
module is a free stopgap only.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

_WS_URL = "wss://streamer.finance.yahoo.com/?version=2"
_RECONNECT_BACKOFF_MIN = 2.0
_RECONNECT_BACKOFF_MAX = 60.0


class YahooWSClient:
    """
    Background WebSocket client. Subscribe once, receive tick dicts via
    callback. Thread-safe start/stop.

    Usage:
        client = YahooWSClient(["RELIANCE.NS", "TCS.NS"], on_tick=print)
        client.start()
        ...
        client.stop()
    """

    def __init__(
        self,
        tickers: Iterable[str],
        on_tick: Callable[[dict[str, Any]], None],
        on_error: Callable[[str], None] | None = None,
    ):
        self.tickers = list(tickers)
        self.on_tick = on_tick
        self.on_error = on_error or (lambda _msg: None)
        self._ws = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=5)

    # ────────────────── internals ──────────────────

    def _run(self) -> None:
        try:
            import websocket  # websocket-client
        except ImportError:
            self.on_error("websocket-client not installed; pip install websocket-client")
            return

        backoff = _RECONNECT_BACKOFF_MIN
        while not self._stop.is_set():
            try:
                self._ws = websocket.WebSocketApp(
                    _WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=lambda _w, e: self.on_error(f"ws_error: {e}"),
                    on_close=lambda _w, _c, _m: None,
                )
                # run_forever blocks until the socket dies
                self._ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:  # noqa: BLE001
                self.on_error(f"connect: {e}")

            if self._stop.is_set():
                break
            # Backoff + jitter before reconnect
            time.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_BACKOFF_MAX)

    def _on_open(self, ws) -> None:
        ws.send(json.dumps({"subscribe": self.tickers}))

    def _on_message(self, _ws, message: str) -> None:
        tick = _decode_pricing_message(message)
        if tick:
            self.on_tick(tick)


def _decode_pricing_message(message: str) -> dict[str, Any]:
    """
    Yahoo sends base64-encoded protobuf frames. Without the .proto file we
    can only extract a few well-known fields (symbol, price, time). This
    returns a partial dict; callers should treat any missing field as None.

    Fallback: if the message isn't valid b64 or protobuf, return {}.
    """
    try:
        raw = base64.b64decode(message)
    except Exception:
        return {}

    # Very lightweight varint-based extraction — avoids a protobuf dep.
    # Yahoo's pricing frame: tag 1 (string) = ticker, tag 2 (float) = price,
    # tag 7 (sfixed64) = unix-ms. Everything else is optional / ignored.
    try:
        idx = 0
        out: dict[str, Any] = {}
        while idx < len(raw):
            tag, field_type, idx = _read_varint_tag(raw, idx)
            if field_type == 2:  # length-delimited
                length, idx = _read_varint(raw, idx)
                chunk = raw[idx : idx + length]
                idx += length
                if tag == 1:
                    out["symbol"] = chunk.decode("utf-8", errors="replace")
            elif field_type == 5:  # 32-bit (float)
                val = int.from_bytes(raw[idx : idx + 4], "little")
                idx += 4
                if tag == 2:
                    out["price"] = _uint32_to_float(val)
            elif field_type == 1:  # 64-bit
                val = int.from_bytes(raw[idx : idx + 8], "little")
                idx += 8
                if tag == 7:
                    out["ts_ms"] = val
            elif field_type == 0:  # varint
                _, idx = _read_varint(raw, idx)
            else:
                # Unknown wire type — give up so we don't misalign the stream.
                break
        return out
    except Exception:
        return {}


def _read_varint(buf: bytes, idx: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while idx < len(buf):
        b = buf[idx]
        idx += 1
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return result, idx
        shift += 7
    raise ValueError("truncated varint")


def _read_varint_tag(buf: bytes, idx: int) -> tuple[int, int, int]:
    raw, idx = _read_varint(buf, idx)
    return raw >> 3, raw & 0x7, idx


def _uint32_to_float(n: int) -> float:
    """IEEE 754 single-precision reinterpret."""
    import struct

    return struct.unpack("<f", n.to_bytes(4, "little"))[0]
