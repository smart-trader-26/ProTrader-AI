"""
Upstox market-data client — A3.4 / A9.1 scaffolding.

Keyless until Upstox KYC + OAuth completes. After KYC:
  1. Register an app at https://account.upstox.com/developer/apps — you get
     an API key + secret.
  2. Run the OAuth flow once (user consent → access token valid for the
     current trading day, refresh daily).
  3. Set UPSTOX_ACCESS_TOKEN in .env / st.secrets. Optionally set
     UPSTOX_API_KEY if you automate the refresh step.

Until a token is present every public function here short-circuits to
`None` / `False`, and callers fall back to the yfinance path. That means
this file is importable and safely callable on a zero-key install.

Ticker → instrument_key resolution:
  Upstox uses `NSE_EQ|INE...` (ISIN-based) rather than `RELIANCE.NS`. The
  map source chain is:
    1. UPSTOX_INSTRUMENTS_JSON env → absolute path to a JSON `{ticker: key}`
    2. <repo>/config/upstox_instruments.json (ships empty; user populates
       after KYC)

Only REST LTP is implemented here because both the paper-trade engine and
the tick publisher poll at 15-second-ish cadence — WebSocket streaming is
a follow-up swap that only touches `workers/tick_publisher.py`.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path

import requests

from config.settings import _get_secret

log = logging.getLogger(__name__)

_LTP_URL = "https://api.upstox.com/v2/market-quote/ltp"
_REQUEST_TIMEOUT = 4.0  # seconds — keep the polling loop responsive

_DEFAULT_INSTRUMENT_FILE = (
    Path(__file__).resolve().parent.parent / "config" / "upstox_instruments.json"
)

_load_lock = threading.Lock()
_instrument_map: dict[str, str] | None = None


def is_configured() -> bool:
    """True iff an Upstox access token is present in settings."""
    return bool(_get_secret("UPSTOX_ACCESS_TOKEN", ""))


def _instruments() -> dict[str, str]:
    """Lazy-load the ticker→instrument_key map. Env path wins over default."""
    global _instrument_map
    if _instrument_map is not None:
        return _instrument_map
    with _load_lock:
        if _instrument_map is not None:
            return _instrument_map
        override = _get_secret("UPSTOX_INSTRUMENTS_JSON", "") or os.environ.get(
            "UPSTOX_INSTRUMENTS_JSON", ""
        )
        path = Path(override) if override else _DEFAULT_INSTRUMENT_FILE
        data: dict[str, str] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    data = {str(k).upper(): str(v) for k, v in raw.items() if v}
            except Exception as e:  # noqa: BLE001
                log.warning("failed to parse %s: %s", path, e)
        _instrument_map = data
        return _instrument_map


def resolve_instrument_key(ticker: str) -> str | None:
    """Map a yfinance-style ticker (RELIANCE.NS) to an Upstox instrument key."""
    if not ticker:
        return None
    key = _instruments().get(ticker.upper())
    return key or None


@lru_cache(maxsize=1)
def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"Accept": "application/json"})
    return s


def get_ltp(ticker: str) -> float | None:
    """
    Return the last traded price for `ticker`, or None when:
      • no token configured (pre-KYC)
      • no instrument key mapped for this ticker
      • any network / parse failure

    Never raises — callers can wrap this in a hot polling loop.
    """
    token = _get_secret("UPSTOX_ACCESS_TOKEN", "")
    if not token:
        return None
    instrument_key = resolve_instrument_key(ticker)
    if not instrument_key:
        return None
    try:
        resp = _session().get(
            _LTP_URL,
            params={"instrument_key": instrument_key},
            headers={"Authorization": f"Bearer {token}"},
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json() or {}
        data = payload.get("data") or {}
        # Upstox returns the instrument key with ":" separator in the response.
        candidate = data.get(instrument_key) or data.get(
            instrument_key.replace("|", ":")
        )
        if not candidate and data:
            candidate = next(iter(data.values()))
        if not candidate:
            return None
        price = candidate.get("last_price")
        if price is None:
            return None
        return float(price)
    except Exception as e:  # noqa: BLE001
        log.debug("upstox LTP fetch failed for %s: %s", ticker, e)
        return None


def upstox_fill_source(ticker: str) -> float | None:
    """Matches the `PaperTradeService(fill_source=...)` signature."""
    return get_ltp(ticker)
