"""
Accuracy ledger (A7) — SQLite-backed prediction log + backfill + rollups.

Write path:
    `log_prediction(bundle)` appends one row per forecast horizon step. UNIQUE
    (ticker, made_at, target_date) silently de-dupes if the same call is made
    twice in the same second (normal Streamlit rerun behaviour).

Resolve path:
    `backfill_actuals(up_to=today)` fetches close prices from yfinance for any
    row where `target_date <= up_to` and `actual_price IS NULL`, then sets
    `actual_price` and `hit` (pred_dir matches actual sign). Idempotent.

Read path:
    `accuracy_window(ticker, days)` returns an `AccuracyWindow` with hit rate +
    Brier + ECE over the resolved rows in the last N days. Used by:
      - `services.prediction_service.predict()` to attach `accuracy_30d`
      - the Streamlit "Accuracy" tab (A7.4)
      - the future FastAPI `/api/v1/accuracy` endpoint

No Streamlit imports — hard invariant CLAUDE.md §2. The DB path defaults to
`data/ledger/predictions.sqlite`; the daemon path uses the same file.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np

from schemas.ledger import AccuracyWindow, LedgerRow
from schemas.prediction import PredictionBundle, PredictionPoint

DEFAULT_DB_PATH = Path("data/ledger/predictions.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker           TEXT NOT NULL,
    made_at          TEXT NOT NULL,
    target_date      TEXT NOT NULL,
    pred_dir         TEXT NOT NULL,
    pred_price       REAL NOT NULL,
    anchor_price     REAL,
    ci_low           REAL,
    ci_high          REAL,
    confidence_level REAL DEFAULT 0.90,
    prob_up          REAL,
    horizon_days     INTEGER DEFAULT 1,
    model_version    TEXT NOT NULL,
    actual_price     REAL,
    hit              INTEGER,
    UNIQUE(ticker, made_at, target_date)
);

CREATE INDEX IF NOT EXISTS idx_pred_ticker      ON predictions(ticker);
CREATE INDEX IF NOT EXISTS idx_pred_target_date ON predictions(target_date);
"""

_LOCK = threading.Lock()


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def log_from_future_df(
    ticker: str,
    future_df,
    anchor_price: float,
    prob_up: float | None,
    model_version: str,
    db_path: Path | None = None,
) -> int:
    """
    Adapter for the Streamlit path: log the raw `hybrid_predict_prices`
    DataFrame without first constructing a `PredictionBundle`. Used by
    `app.py` so the Streamlit UI (which currently consumes raw metrics)
    keeps working while still feeding the ledger.
    """
    if future_df is None or len(future_df) == 0:
        return 0

    prices = future_df["Predicted Price"].tolist() if "Predicted Price" in future_df.columns else future_df.iloc[:, 0].tolist()
    dates = [(d.date() if hasattr(d, "date") else d) for d in future_df.index]
    lo = future_df["P5"].tolist() if "P5" in future_df.columns else [None] * len(prices)
    hi = future_df["P95"].tolist() if "P95" in future_df.columns else [None] * len(prices)

    points: list[PredictionPoint] = []
    for i, (d, p) in enumerate(zip(dates, prices)):
        # Compare every day vs the ORIGINAL anchor, not rolling prev.
        # Mirrors the fix applied to pg_ledger.log_from_future_df.
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
    return log_prediction(bundle, anchor_price=anchor_price, db_path=db_path)


def log_prediction(
    bundle: PredictionBundle,
    anchor_price: float | None = None,
    db_path: Path | None = None,
) -> int:
    """
    Append every horizon point to the ledger. Returns number of new rows.

    `anchor_price` is the close on `made_at`'s trading day (passed in from the
    caller — the prediction_service knows this). Stored so `hit` can be
    computed later without a second yfinance round-trip.
    """
    if not bundle.points:
        return 0

    # Drop any forecast points that fall before the run date.  The hybrid model
    # includes the last historical close (e.g. Apr 30) as "day-0" in future_df,
    # but that date is already in the past when the prediction runs on May 1.
    # If we log it, backfill_actuals resolves actual_price == anchor_price → hit=0
    # for every non-flat call, poisoning directional accuracy.
    made_date = bundle.made_at.astimezone(UTC).date()
    points_to_log = [p for p in bundle.points if p.target_date >= made_date]
    if not points_to_log:
        return 0

    # Only the earliest forward target is a genuine directional call; later
    # points are MC scenario paths.  Restrict prob_up to day-1 so
    # accuracy_window calibration (ECE, Brier) is not inflated.
    earliest_target = min(p.target_date for p in points_to_log)

    rows = []
    for p in points_to_log:
        is_day1 = (p.target_date == earliest_target)
        rows.append(
            (
                bundle.ticker,
                bundle.made_at.astimezone(UTC).isoformat(),
                p.target_date.isoformat(),
                p.direction,
                float(p.pred_price),
                float(anchor_price) if anchor_price is not None else None,
                float(p.ci_low) if p.ci_low is not None else None,
                float(p.ci_high) if p.ci_high is not None else None,
                float(p.confidence_level),
                float(p.prob_up) if (p.prob_up is not None and is_day1) else None,
                int(bundle.horizon_days),
                bundle.model_version,
            )
        )

    with _LOCK, _connect(db_path) as conn:
        cur = conn.executemany(
            """
            INSERT OR IGNORE INTO predictions
                (ticker, made_at, target_date, pred_dir, pred_price, anchor_price,
                 ci_low, ci_high, confidence_level, prob_up,
                 horizon_days, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return cur.rowcount or 0


def unresolved_tickers(
    up_to: date | None = None, db_path: Path | None = None
) -> list[tuple[str, date]]:
    """Return (ticker, target_date) pairs that need actuals filled in."""
    up_to = up_to or date.today()
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT ticker, target_date
            FROM predictions
            WHERE actual_price IS NULL AND target_date <= ?
            """,
            (up_to.isoformat(),),
        ).fetchall()
    return [(r["ticker"], _parse_date(r["target_date"])) for r in rows]


def backfill_actuals(
    up_to: date | None = None,
    db_path: Path | None = None,
    price_fetcher=None,
) -> int:
    """
    Fill `actual_price` + `hit` for every row whose target_date has passed.

    `price_fetcher(ticker, d) -> float | None` is injectable so tests skip
    yfinance. Default fetcher uses yfinance on a per-ticker batch (fast).
    Returns the number of rows resolved.
    """
    up_to = up_to or date.today()
    fetcher = price_fetcher or _yfinance_close

    pairs = unresolved_tickers(up_to, db_path)
    if not pairs:
        return 0

    # Group by ticker so a single yfinance call covers all outstanding dates.
    by_ticker: dict[str, list[date]] = {}
    for ticker, d in pairs:
        by_ticker.setdefault(ticker, []).append(d)

    resolved = 0
    with _LOCK, _connect(db_path) as conn:
        for ticker, dates in by_ticker.items():
            lo, hi = min(dates), max(dates)
            price_by_date: dict[date, float] = fetcher(ticker, lo, hi)
            if not price_by_date:
                continue

            # Cache anchor closes per made_at date so repeated lookups stay cheap.
            anchor_cache: dict[date, float] = {}

            for d in dates:
                actual = price_by_date.get(d)
                if actual is None:
                    continue

                pending = conn.execute(
                    "SELECT id, pred_dir, made_at, anchor_price FROM predictions "
                    "WHERE ticker = ? AND target_date = ? AND actual_price IS NULL",
                    (ticker, d.isoformat()),
                ).fetchall()

                for row in pending:
                    anchor = row["anchor_price"]
                    if anchor is None:
                        made_date = _parse_datetime(row["made_at"]).date()
                        anchor = anchor_cache.get(made_date)
                        if anchor is None:
                            anchor = _anchor_close(ticker, made_date, fetcher)
                            anchor_cache[made_date] = anchor

                    hit: int | None
                    if np.isnan(anchor):
                        hit = None
                    elif row["pred_dir"] == "up":
                        hit = 1 if actual > anchor else 0
                    elif row["pred_dir"] == "down":
                        hit = 1 if actual < anchor else 0
                    else:
                        hit = None

                    conn.execute(
                        "UPDATE predictions SET actual_price = ?, hit = ? WHERE id = ?",
                        (float(actual), hit, row["id"]),
                    )
                    resolved += 1

        conn.commit()
    return resolved


def _anchor_close(ticker: str, made_date: date, fetcher) -> float:
    """Close price on the trading day the forecast was made (for hit check)."""
    lookup = fetcher(ticker, made_date - timedelta(days=7), made_date)
    if not lookup:
        return float("nan")
    # Nearest prior trading day <= made_date
    candidates = sorted(d for d in lookup if d <= made_date)
    if not candidates:
        return float("nan")
    return float(lookup[candidates[-1]])


def accuracy_window(
    ticker: str | None,
    days: int = 30,
    db_path: Path | None = None,
    now: date | None = None,
) -> AccuracyWindow:
    """Rolling accuracy over the last `days` (by target_date). Resolved rows only.

    Bug 4 fix: dedupe by (ticker, target_date) and keep the FIRST prediction
    (MIN made_at). The schema previously let several predictions for the same
    target_date be counted as independent samples — they aren't, they share an
    outcome. Worse, the earlier `recent_rows` view kept MAX(made_at), which is
    the prediction made closest to target — systematically the easier call.
    Keeping MIN(made_at) here gives an honest accuracy of the model's first
    look at each target.

    Day-1 restriction: the dedup subquery filters to `prob_up IS NOT NULL` so
    the join always picks a genuine day-1 directional call. Without that
    filter, MIN(made_at) systematically picks the earliest *bundle* covering a
    target — typically a multi-day-ahead forecast where that target appears as
    day-N (prob_up=NULL). The downstream `prob_up IS NOT NULL` hit filter then
    rejects every selected row and `directional_accuracy` collapses to None.
    """
    now = now or date.today()
    cutoff = (now - timedelta(days=days)).isoformat()

    where = "target_date >= ? AND prob_up IS NOT NULL"
    params: list = [cutoff]
    if ticker:
        where += " AND ticker = ?"
        params.append(ticker)

    with _connect(db_path) as conn:
        # Total = unique (ticker, target_date) pairs that had a day-1 call.
        total = conn.execute(
            f"SELECT COUNT(*) AS n FROM ("
            f"  SELECT 1 FROM predictions WHERE {where} "
            f"  GROUP BY ticker, target_date"
            f")",
            params,
        ).fetchone()["n"]
        rows = conn.execute(
            f"""
            SELECT p.pred_dir, p.pred_price, p.anchor_price,
                   p.actual_price, p.prob_up, p.hit
              FROM predictions p
              INNER JOIN (
                  SELECT ticker, target_date, MIN(made_at) AS first_made_at
                    FROM predictions
                   WHERE {where}
                GROUP BY ticker, target_date
              ) first_call
                ON p.ticker      = first_call.ticker
               AND p.target_date = first_call.target_date
               AND p.made_at     = first_call.first_made_at
             WHERE p.actual_price IS NOT NULL
            """,
            params,
        ).fetchall()

    if not rows:
        return AccuracyWindow(
            ticker=ticker,
            window_days=days,
            n_predictions=int(total or 0),
            n_resolved=0,
        )

    hits: list[int] = []
    price_err: list[float] = []
    probs: list[float] = []
    truths: list[int] = []
    for r in rows:
        actual = r["actual_price"]
        pred = r["pred_price"]
        anchor = r["anchor_price"]
        if actual is None or pred is None:
            continue
        price_err.append(abs(actual - pred))
        # Only count hits for day-1 rows (prob_up IS NOT NULL).
        # Days 2-10 are scenario paths, not independent directional calls.
        if r["hit"] is not None and r["prob_up"] is not None:
            hits.append(int(r["hit"]))
        if r["prob_up"] is not None and anchor is not None:
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
        n_predictions=int(total or 0),
        n_resolved=len(rows),
        directional_accuracy=acc,
        brier_score=brier,
        ece=ece,
        mae_price=mae,
    )


def recent_rows(
    ticker: str | None = None,
    limit: int = 200,
    db_path: Path | None = None,
) -> list[LedgerRow]:
    """Return the most recent ledger rows, one per (ticker, target_date).

    Bug 4 fix: keep the FIRST prediction (MIN made_at) for each
    (ticker, target_date) pair, not the latest. Multiple runs on the same
    target inflate apparent accuracy when we cherry-pick the call made
    closest to target — the model has less time to drift, so MAX(made_at)
    systematically over-states the win rate. The first call is the honest
    test, so this view (and `accuracy_window`) both pin to MIN.
    """
    where = ""
    params: list = []
    if ticker:
        where = "WHERE ticker = ?"
        params.append(ticker)
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT p.*
            FROM predictions p
            INNER JOIN (
                SELECT ticker, target_date, MIN(made_at) AS first_made_at
                FROM predictions
                {where}
                GROUP BY ticker, target_date
            ) first_call
            ON p.ticker = first_call.ticker
            AND p.target_date = first_call.target_date
            AND p.made_at = first_call.first_made_at
            ORDER BY p.made_at DESC, p.id DESC
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()
    return [_row_to_dto(r) for r in rows]


# ───────────────────────── helpers ─────────────────────────────────

def _expected_calibration_error(probs: np.ndarray, truths: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        mask = (probs >= edges[i]) & (probs < edges[i + 1] if i < n_bins - 1 else probs <= edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_prob = float(np.mean(probs[mask]))
        bin_truth = float(np.mean(truths[mask]))
        ece += (mask.sum() / n) * abs(bin_prob - bin_truth)
    return float(ece)


def _row_to_dto(r: sqlite3.Row) -> LedgerRow:
    keys = r.keys()
    anchor = r["anchor_price"] if "anchor_price" in keys else None
    return LedgerRow(
        id=r["id"],
        ticker=r["ticker"],
        made_at=_parse_datetime(r["made_at"]),
        target_date=_parse_date(r["target_date"]),
        pred_dir=r["pred_dir"],
        pred_price=r["pred_price"],
        anchor_price=anchor,
        ci_low=r["ci_low"],
        ci_high=r["ci_high"],
        confidence_level=r["confidence_level"] or 0.90,
        prob_up=r["prob_up"],
        horizon_days=r["horizon_days"] or 1,
        model_version=r["model_version"],
        actual_price=r["actual_price"],
        hit=None if r["hit"] is None else bool(r["hit"]),
    )


def _parse_date(s: str) -> date:
    return datetime.fromisoformat(s).date() if "T" in s else date.fromisoformat(s)


def _parse_datetime(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def migrate_delete_stale_predictions(db_path: Path | None = None) -> int:
    """One-time cleanup: delete ledger rows whose target_date is strictly
    before the UTC date of made_at.

    These rows were logged before `log_prediction` filtered out past-dated
    forecast points. yfinance returns the same close for the prediction's
    anchor day and the (already past) target day, so `actual_price ==
    anchor_price` and the directional comparison is meaningless — every such
    row contributes a guaranteed-zero hit (and a bogus calibration sample) to
    the rolling accuracy metrics.

    Idempotent: rows that satisfy `target_date >= DATE(made_at)` are not
    touched. Returns the number of rows deleted.
    """
    with _connect(db_path) as conn:
        result = conn.execute(
            "DELETE FROM predictions WHERE target_date < DATE(made_at)"
        )
        deleted = result.rowcount
        conn.commit()
    return deleted


def migrate_nullify_non_day1_prob_up(db_path: Path | None = None) -> int:
    """One-time migration: set prob_up=NULL for every row that is NOT the
    earliest target_date for its (ticker, made_at) group.

    Before Phase-1C was deployed, all 10 horizon rows were stored with the
    same prob_up value (the day-1 directional probability). This inflated the
    denominator in accuracy_window's ECE / Brier computation and produced
    identical 7/30/90-day stats (all resolved rows fell within the 7-day
    window, and all had prob_up set, so nothing was filtered out).

    After this migration, only the genuine day-1 call has prob_up set;
    days 2-10 are NULL and are excluded from calibration scoring.

    Safe to run multiple times — rows already NULL are unaffected.
    Returns the number of rows updated.
    """
    with _connect(db_path) as conn:
        # SQLite supports correlated sub-queries: find the earliest target_date
        # for each (ticker, made_at) pair, then NULL-out every other row.
        result = conn.execute(
            """
            UPDATE predictions
            SET prob_up = NULL
            WHERE prob_up IS NOT NULL
              AND target_date != (
                  SELECT MIN(p2.target_date)
                  FROM predictions p2
                  WHERE p2.ticker    = predictions.ticker
                    AND p2.made_at   = predictions.made_at
              )
            """
        )
        updated = result.rowcount
        conn.commit()
    return updated


def _yfinance_close(ticker: str, lo: date, hi: date) -> dict[date, float]:
    """Default price fetcher — one yfinance history call per ticker span."""
    try:
        import yfinance as yf
    except ImportError:
        return {}

    try:
        # Pad one day on each side to survive weekends / holidays on the edges.
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
