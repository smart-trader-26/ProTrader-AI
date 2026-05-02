"""
Postgres ledger adapter (B3 framework mode) — drop-in for `services.ledger_service`.

Talks to Supabase over HTTPS via supabase-py / PostgREST. No direct Postgres
connection, no psycopg, no IPv6 requirement — the backend can live anywhere
HTTPS reaches.

Public surface (unchanged from the earlier SQLAlchemy version so callers
don't need to care which backend is live):

    log_prediction(bundle, anchor_price=..., user_id=...)  -> int
    log_from_future_df(ticker, future_df, anchor_price, prob_up, model_version,
                       user_id=...)                         -> int
    backfill_actuals(up_to=..., price_fetcher=...)          -> int
    accuracy_window(ticker, days=30, now=...)               -> AccuracyWindow
    recent_rows(ticker=None, limit=200, user_id=None)       -> list[LedgerRow]

All writes use the service-role client (RLS bypassed) — the ledger is a
backend-owned table. The row's `user_id` is preserved for per-user rollups
in `accuracy_window` / `recent_rows`.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta

import numpy as np

from db.supabase_client import get_admin_client
from schemas.ledger import AccuracyWindow, LedgerRow
from schemas.prediction import PredictionBundle, PredictionPoint

log = logging.getLogger(__name__)

_TABLE = "predictions"


# ───────────────────────── write path ────────────────────────────────────


def log_prediction(
    bundle: PredictionBundle,
    anchor_price: float | None = None,
    user_id: str | None = None,
) -> int:
    if not bundle.points:
        return 0

    # Drop forecast points whose target_date is already in the past relative
    # to made_at. yfinance returns the same close for the anchor day and any
    # earlier target, so logging stale points produces actual_price ==
    # anchor_price → meaningless directional hit (always 0 for non-flat).
    made_date = bundle.made_at.astimezone(UTC).date()
    points_to_log = [p for p in bundle.points if p.target_date >= made_date]
    if not points_to_log:
        return 0

    # Only the earliest forward target is a genuine directional call; later
    # points are MC scenario paths. Restrict prob_up to day-1 so calibration
    # metrics (ECE, Brier) aren't inflated by repeated identical probabilities.
    earliest_target = min(p.target_date for p in points_to_log)

    rows = []
    for p in points_to_log:
        is_day1 = (p.target_date == earliest_target)
        rows.append(
            {
                "ticker":           bundle.ticker,
                "made_at":          bundle.made_at.astimezone(UTC).isoformat(),
                "target_date":      p.target_date.isoformat(),
                "pred_dir":         p.direction,
                "pred_price":       float(p.pred_price),
                "anchor_price":     float(anchor_price) if anchor_price is not None else None,
                "ci_low":           float(p.ci_low) if p.ci_low is not None else None,
                "ci_high":          float(p.ci_high) if p.ci_high is not None else None,
                "confidence_level": float(p.confidence_level),
                "prob_up":          float(p.prob_up) if (p.prob_up is not None and is_day1) else None,
                "horizon_days":     int(bundle.horizon_days),
                "model_version":    bundle.model_version,
                "user_id":          user_id,
            }
        )

    # ignore_duplicates maps to Prefer: resolution=ignore-duplicates — same as ON CONFLICT DO NOTHING.
    resp = (
        get_admin_client()
        .table(_TABLE)
        .upsert(rows, on_conflict="ticker,made_at,target_date", ignore_duplicates=True)
        .execute()
    )
    return len(resp.data or [])


def log_from_future_df(
    ticker: str,
    future_df,
    anchor_price: float,
    prob_up: float | None,
    model_version: str,
    user_id: str | None = None,
) -> int:
    if future_df is None or len(future_df) == 0:
        return 0

    prices = (
        future_df["Predicted Price"].tolist()
        if "Predicted Price" in future_df.columns
        else future_df.iloc[:, 0].tolist()
    )
    dates = [(d.date() if hasattr(d, "date") else d) for d in future_df.index]
    lo = future_df["P5"].tolist() if "P5" in future_df.columns else [None] * len(prices)
    hi = future_df["P95"].tolist() if "P95" in future_df.columns else [None] * len(prices)

    points: list[PredictionPoint] = []
    for i, (d, p) in enumerate(zip(dates, prices)):
        # Compare every day's predicted price to the ORIGINAL anchor, not to the
        # previous predicted price. The ledger resolves `hit` using
        # `actual_price vs anchor_price`, so direction must use the same baseline.
        # The old rolling `prev = p` caused day-2+ directions to be computed
        # relative to a drifting predicted price, systematically mis-marking hits.
        if p > anchor_price * 1.001:
            direction = "up"
        elif p < anchor_price * 0.999:
            direction = "down"
        else:
            direction = "flat"

        # Set prob_up uniformly on every point. log_prediction restricts it to
        # the earliest forward-dated row (the genuine day-1 call) — pinning it
        # to i=0 here would lose the prob entirely when log_prediction filters
        # out today's already-past close (i=0) and leaves i=1 as day-1.
        day_prob = max(0.0, min(1.0, float(prob_up) if prob_up is not None else 0.5))

        points.append(
            PredictionPoint(
                target_date=d,
                pred_price=float(p),
                ci_low=float(lo[i]) if lo[i] is not None else None,
                ci_high=float(hi[i]) if hi[i] is not None else None,
                direction=direction,  # type: ignore[arg-type]
                prob_up=day_prob,
            )
        )

    bundle = PredictionBundle(
        ticker=ticker,
        made_at=datetime.now(UTC),
        model_version=model_version,
        horizon_days=len(points),
        points=points,
    )
    return log_prediction(bundle, anchor_price=anchor_price, user_id=user_id)


# ───────────────────────── resolve path ──────────────────────────────────


def unresolved_tickers(up_to: date | None = None) -> list[tuple[str, date]]:
    up_to = up_to or date.today()
    resp = (
        get_admin_client()
        .table(_TABLE)
        .select("ticker,target_date")
        .is_("actual_price", "null")
        .lte("target_date", up_to.isoformat())
        .execute()
    )
    seen: set[tuple[str, date]] = set()
    for r in resp.data or []:
        td = r["target_date"]
        if isinstance(td, str):
            td = date.fromisoformat(td[:10])
        seen.add((r["ticker"], td))
    return sorted(seen)


def backfill_actuals(
    up_to: date | None = None,
    price_fetcher=None,
) -> int:
    """Fill `actual_price` + `hit` for every row whose target_date has passed.

    `price_fetcher(ticker, lo, hi) -> dict[date, float]` is injectable so
    tests skip yfinance.
    """
    up_to = up_to or date.today()
    fetcher = price_fetcher or _yfinance_close

    pairs = unresolved_tickers(up_to)
    if not pairs:
        return 0

    by_ticker: dict[str, list[date]] = {}
    for ticker, d in pairs:
        by_ticker.setdefault(ticker, []).append(d)

    resolved = 0
    client = get_admin_client()

    for ticker, dates in by_ticker.items():
        lo, hi = min(dates), max(dates)
        price_by_date: dict[date, float] = fetcher(ticker, lo, hi)
        if not price_by_date:
            continue

        anchor_cache: dict[date, float] = {}

        for d in dates:
            actual = price_by_date.get(d)
            if actual is None:
                continue

            pending = (
                client.table(_TABLE)
                .select("id,pred_dir,made_at,anchor_price")
                .eq("ticker", ticker)
                .eq("target_date", d.isoformat())
                .is_("actual_price", "null")
                .execute()
            )

            for row in pending.data or []:
                anchor = row.get("anchor_price")
                if anchor is None:
                    made_at_raw = row["made_at"]
                    # Convert UTC timestamp → IST date to get the correct NSE
                    # trading day the prediction was originally made on.
                    made_date = _to_ist_date(made_at_raw)
                    anchor = anchor_cache.get(made_date)
                    if anchor is None:
                        anchor = _anchor_close(ticker, made_date, fetcher)
                        anchor_cache[made_date] = anchor

                hit: int | None
                if anchor is None or (isinstance(anchor, float) and np.isnan(anchor)):
                    hit = None
                elif row["pred_dir"] == "up":
                    hit = 1 if actual > anchor else 0
                elif row["pred_dir"] == "down":
                    hit = 1 if actual < anchor else 0
                else:
                    hit = None

                client.table(_TABLE).update(
                    {"actual_price": float(actual), "hit": hit}
                ).eq("id", row["id"]).execute()
                resolved += 1

    return resolved


def _anchor_close(ticker: str, made_date: date, fetcher) -> float:
    lookup = fetcher(ticker, made_date - timedelta(days=7), made_date)
    if not lookup:
        return float("nan")
    candidates = sorted(d for d in lookup if d <= made_date)
    if not candidates:
        return float("nan")
    return float(lookup[candidates[-1]])


_IST = None


def _to_ist_date(made_at_raw) -> date:
    """Convert a made_at value (UTC string or datetime) to an IST calendar date.

    NSE closes at 15:30 IST. Predictions logged after midnight UTC (= 05:30 IST)
    must be anchored against the same IST trading day, not the UTC date which may
    be one day ahead. Using UTC dates directly caused anchors to slip by one day
    for predictions logged in the evening IST.
    """
    global _IST
    if _IST is None:
        try:
            import pytz
            _IST = pytz.timezone("Asia/Kolkata")
        except ImportError:
            _IST = UTC  # graceful fallback — pytz not installed

    if isinstance(made_at_raw, str):
        dt = datetime.fromisoformat(made_at_raw.replace("Z", "+00:00"))
    else:
        dt = made_at_raw
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(_IST).date()


# ───────────────────────── read path ─────────────────────────────────────


def accuracy_window(
    ticker: str | None,
    days: int = 30,
    now: date | None = None,
    user_id: str | None = None,
) -> AccuracyWindow:
    now = now or date.today()
    cutoff = now - timedelta(days=days)

    client = get_admin_client()

    # Bug 4 fix: dedupe by (ticker, target_date) and keep the FIRST prediction
    # (MIN made_at) per pair. PostgREST doesn't expose GROUP BY, so we pull the
    # candidate rows ordered by made_at ASC and drop later duplicates client-side.
    # The earlier code counted every prediction for the same target as an
    # independent sample — they aren't, they share an outcome — and `recent_rows`
    # used to keep MAX(made_at) which is the easier (closest-to-target) call.
    #
    # Day-1 restriction: pre-filter the rows to `prob_up IS NOT NULL` before
    # deduping. Without that, the first-seen row for any target_date is
    # systematically the earliest *bundle* covering it — which means that
    # target appears as day-N (prob_up=NULL). The downstream `prob_up IS NOT
    # NULL` hit filter then rejects every selected row and directional
    # accuracy collapses to None.
    fetch_q = (
        client.table(_TABLE)
        .select(
            "ticker,target_date,made_at,pred_dir,pred_price,anchor_price,"
            "actual_price,prob_up,hit"
        )
        .gte("target_date", cutoff.isoformat())
        .not_.is_("prob_up", "null")
        .order("made_at", desc=False)
    )
    if ticker:
        fetch_q = fetch_q.eq("ticker", ticker)
    if user_id:
        fetch_q = fetch_q.eq("user_id", user_id)
    raw = fetch_q.execute().data or []

    seen: set[tuple[str, str]] = set()
    deduped = []
    for r in raw:
        key = (r.get("ticker"), str(r.get("target_date")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    total = len(deduped)
    rows = [r for r in deduped if r.get("actual_price") is not None]

    if not rows:
        return AccuracyWindow(
            ticker=ticker,
            window_days=days,
            n_predictions=int(total),
            n_resolved=0,
        )

    hits: list[int] = []
    price_err: list[float] = []
    probs: list[float] = []
    truths: list[int] = []
    for r in rows:
        actual = r.get("actual_price")
        pred = r.get("pred_price")
        anchor = r.get("anchor_price")
        if actual is None or pred is None:
            continue
        price_err.append(abs(actual - pred))
        # Only count hits for day-1 predictions (prob_up IS NOT NULL).
        # Days 2-10 are Monte Carlo scenario paths — their pred_dir is an
        # indication of trajectory direction vs anchor, not an independent
        # directional forecast. Including them in the hit rate would inflate the
        # denominator with rows that were never meant to be directional calls.
        if r.get("hit") is not None and r.get("prob_up") is not None:
            hits.append(int(r["hit"]))
        if r.get("prob_up") is not None and anchor is not None:
            probs.append(float(r["prob_up"]))
            truths.append(1 if actual > anchor else 0)

    acc = float(np.mean(hits)) if hits else None
    mae = float(np.mean(price_err)) if price_err else None

    brier = None
    ece = None
    if probs:
        p = np.array(probs)
        t = np.array(truths)
        brier = float(np.mean((p - t) ** 2))
        ece = _expected_calibration_error(p, t, n_bins=10)

    return AccuracyWindow(
        ticker=ticker,
        window_days=days,
        n_predictions=int(total),
        n_resolved=len(rows),
        directional_accuracy=acc,
        brier_score=brier,
        ece=ece,
        mae_price=mae,
    )


def recent_rows(
    ticker: str | None = None,
    limit: int = 200,
    user_id: str | None = None,
) -> list[LedgerRow]:
    """Return most recent ledger rows, deduped to one entry per
    (ticker, target_date) pair — keeping the FIRST prediction (MIN made_at).

    Bug 4 fix: matches the dedup policy in `accuracy_window`. The first
    prediction is the honest test (longest distance to target); keeping
    MAX(made_at) silently picked the easiest call.
    """
    # Pull ascending by made_at so the dedup keeps the earliest record per pair.
    q = get_admin_client().table(_TABLE).select("*")
    if ticker:
        q = q.eq("ticker", ticker)
    if user_id:
        q = q.eq("user_id", user_id)
    # Over-fetch so post-dedup we still hit `limit`. Hard cap at 5x to bound the trip.
    q = q.order("made_at", desc=False).order("id", desc=False).limit(limit * 5)
    raw = q.execute().data or []

    seen: set[tuple[str, str]] = set()
    deduped = []
    for r in raw:
        key = (r.get("ticker"), str(r.get("target_date")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    deduped.sort(key=lambda r: (r.get("made_at") or "", r.get("id") or 0), reverse=True)
    return [_row_to_dto(r) for r in deduped[:limit]]


# ───────────────────────── helpers ───────────────────────────────────────


def _expected_calibration_error(
    probs: np.ndarray, truths: np.ndarray, n_bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        mask = (probs >= edges[i]) & (
            probs < edges[i + 1] if i < n_bins - 1 else probs <= edges[i + 1]
        )
        if mask.sum() == 0:
            continue
        bin_prob = float(np.mean(probs[mask]))
        bin_truth = float(np.mean(truths[mask]))
        ece += (mask.sum() / n) * abs(bin_prob - bin_truth)
    return float(ece)


def _row_to_dto(r: dict) -> LedgerRow:
    made_at = r["made_at"]
    if isinstance(made_at, str):
        made_at = datetime.fromisoformat(made_at.replace("Z", "+00:00"))
    target_date = r["target_date"]
    if isinstance(target_date, str):
        target_date = date.fromisoformat(target_date[:10])
    return LedgerRow(
        id=r["id"],
        ticker=r["ticker"],
        made_at=made_at,
        target_date=target_date,
        pred_dir=r["pred_dir"],
        pred_price=r["pred_price"],
        anchor_price=r.get("anchor_price"),
        ci_low=r.get("ci_low"),
        ci_high=r.get("ci_high"),
        confidence_level=r.get("confidence_level") or 0.90,
        prob_up=r.get("prob_up"),
        horizon_days=r.get("horizon_days") or 1,
        model_version=r["model_version"],
        actual_price=r.get("actual_price"),
        hit=None if r.get("hit") is None else bool(r["hit"]),
    )


def _yfinance_close(ticker: str, lo: date, hi: date) -> dict[date, float]:
    try:
        import yfinance as yf
    except ImportError:
        return {}
    try:
        start = (lo - timedelta(days=3)).isoformat()
        end = (hi + timedelta(days=3)).isoformat()
        hist = yf.Ticker(ticker).history(start=start, end=end)
    except Exception:
        return {}
    if hist is None or hist.empty or "Close" not in hist.columns:
        return {}
    out: dict[date, float] = {}
    for ts, row in hist.iterrows():
        d = ts.date() if hasattr(ts, "date") else ts
        out[d] = float(row["Close"])
    return out
