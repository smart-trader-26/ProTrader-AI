"""
Swing screener (P4-2) — run the persisted 30-day directional signal across the
whole universe in one batch.

Why: the signal abstains ~94% of the time by design, so the per-ticker view
almost always shows NEUTRAL. The actionable product question is "which names
are firing today?" — that's what `scan()` answers.

Cost profile: one batched yfinance download (~2y closes for the universe) +
logistic-regression inference per name. No hybrid model, no GRU — a full scan
is seconds, not minutes. Results are cached in-process for `_CACHE_TTL` so the
dashboard can poll freely; pass `refresh=True` to force a re-scan.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime, timedelta

import pandas as pd

from schemas.swing import SwingScanResult, SwingScanRow

logger = logging.getLogger(__name__)

# The signal is daily — a fresh scan more than every 30 min buys nothing.
_CACHE_TTL = timedelta(minutes=30)
_MIN_BARS = 260  # signal features need ~252 bars (12-1 momentum)

_lock = threading.Lock()
_cache: SwingScanResult | None = None
_cache_at: datetime | None = None


def _default_universe() -> list[str]:
    from models.directional_signal import _DEFAULT_UNIVERSE

    return list(_DEFAULT_UNIVERSE)


def _download_closes(universe: list[str]) -> pd.DataFrame:
    """One batched download of ~2y of auto-adjusted closes (date × ticker)."""
    import yfinance as yf

    raw = yf.download(universe, period="2y", auto_adjust=True,
                      progress=False, timeout=60)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=universe[0])
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    return close.dropna(how="all")


def scan(universe: list[str] | None = None, refresh: bool = False) -> SwingScanResult:
    """
    Run the swing signal over `universe` (default: the training universe).

    Returns a `SwingScanResult` with every scanned name, fired UP rows first
    (sorted by calibrated probability). Serves a cached result when one is
    fresher than `_CACHE_TTL`, unless `refresh=True`.
    """
    global _cache, _cache_at

    use_cache = universe is None and not refresh
    if use_cache and _cache is not None and _cache_at is not None:
        if datetime.now(UTC) - _cache_at < _CACHE_TTL:
            return _cache.model_copy(update={"cached": True})

    with _lock:
        # Double-check after acquiring the lock — another thread may have
        # finished the same scan while we waited.
        if use_cache and _cache is not None and _cache_at is not None:
            if datetime.now(UTC) - _cache_at < _CACHE_TTL:
                return _cache.model_copy(update={"cached": True})

        result = _scan_uncached(universe or _default_universe())
        if universe is None:
            _cache, _cache_at = result, datetime.now(UTC)
        return result


def _scan_uncached(universe: list[str]) -> SwingScanResult:
    from models.directional_signal import get_signal

    sig = get_signal()
    result = SwingScanResult(
        generated_at=datetime.now(UTC),
        horizon_days=sig.horizon,
        trained=sig.is_trained(),
        tau_star=sig.stats.tau_star or None,
        expected_hit_rate=sig.stats.oos_precision or None,
        base_rate=sig.stats.oos_base_rate or None,
        coverage=sig.stats.oos_coverage or None,
    )
    if not sig.is_trained():
        return result

    try:
        closes = _download_closes(universe)
    except Exception as exc:
        logger.warning("swing scan: universe download failed: %s", exc)
        result.n_errors = len(universe)
        return result

    rows: list[SwingScanRow] = []
    n_errors = 0
    for ticker in universe:
        if ticker not in closes.columns:
            n_errors += 1
            continue
        s = closes[ticker].dropna()
        if len(s) < _MIN_BARS:
            n_errors += 1
            continue
        out = sig.predict(s, ticker=ticker)
        rows.append(SwingScanRow(
            ticker=ticker,
            signal=out["signal"],
            prob_up=float(out["prob_up"]),
            conviction=float(out.get("conviction", 0.0)),
            last_close=float(s.iloc[-1]),
            asof=str(s.index[-1].date()) if hasattr(s.index[-1], "date") else None,
        ))

    # Fired UP names first, then near-misses — both by probability descending.
    rows.sort(key=lambda r: (r.signal != "UP", -r.prob_up))
    result.rows = rows
    result.n_scanned = len(rows)
    result.n_fired = sum(1 for r in rows if r.signal == "UP")
    result.n_errors = n_errors
    return result
