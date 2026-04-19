"""
Unit tests for the A7 accuracy ledger.

No yfinance dependency — `price_fetcher` is injectable by design (CLAUDE.md
§2.4: "don't mock data sources". The fetcher itself is a seam, not a mock
of yfinance output). Every test uses a tmp_path SQLite file so runs are
isolated.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from schemas.prediction import PredictionBundle, PredictionPoint
from services import ledger_service


def _bundle(ticker: str, made_at: datetime, points: list[PredictionPoint]) -> PredictionBundle:
    return PredictionBundle(
        ticker=ticker,
        made_at=made_at,
        model_version="test-v1",
        horizon_days=len(points),
        points=points,
    )


def _point(target: date, pred_price: float, direction: str, prob_up: float = 0.6) -> PredictionPoint:
    return PredictionPoint(
        target_date=target,
        pred_price=pred_price,
        direction=direction,  # type: ignore[arg-type]
        prob_up=prob_up,
    )


def test_log_prediction_inserts_rows(tmp_path):
    db = tmp_path / "ledger.sqlite"
    made = datetime(2026, 4, 10, 9, 30, tzinfo=UTC)
    bundle = _bundle(
        "TCS.NS",
        made,
        [
            _point(date(2026, 4, 11), 4000.0, "up"),
            _point(date(2026, 4, 12), 4020.0, "up"),
        ],
    )

    n = ledger_service.log_prediction(bundle, anchor_price=3980.0, db_path=db)
    assert n == 2

    rows = ledger_service.recent_rows(db_path=db)
    assert len(rows) == 2
    assert {r.target_date for r in rows} == {date(2026, 4, 11), date(2026, 4, 12)}
    assert all(r.anchor_price == 3980.0 for r in rows)


def test_log_prediction_is_idempotent(tmp_path):
    """INSERT OR IGNORE on (ticker, made_at, target_date) — reruns must not dup."""
    db = tmp_path / "ledger.sqlite"
    made = datetime(2026, 4, 10, 9, 30, tzinfo=UTC)
    bundle = _bundle("TCS.NS", made, [_point(date(2026, 4, 11), 4000.0, "up")])

    ledger_service.log_prediction(bundle, anchor_price=3980.0, db_path=db)
    ledger_service.log_prediction(bundle, anchor_price=3980.0, db_path=db)

    rows = ledger_service.recent_rows(db_path=db)
    assert len(rows) == 1


def test_backfill_resolves_up_direction_correctly(tmp_path):
    db = tmp_path / "ledger.sqlite"
    made = datetime(2026, 4, 10, 9, 30, tzinfo=UTC)
    bundle = _bundle(
        "TCS.NS",
        made,
        [
            _point(date(2026, 4, 11), 4050.0, "up"),
            _point(date(2026, 4, 12), 4100.0, "up"),
        ],
    )
    ledger_service.log_prediction(bundle, anchor_price=4000.0, db_path=db)

    # Day 1 went up (4020 > 4000) → hit. Day 2 went down (3990 < 4000) → miss.
    fetched_prices = {
        date(2026, 4, 11): 4020.0,
        date(2026, 4, 12): 3990.0,
    }

    def fetcher(ticker, lo, hi):
        assert ticker == "TCS.NS"
        return fetched_prices

    resolved = ledger_service.backfill_actuals(
        up_to=date(2026, 4, 15), db_path=db, price_fetcher=fetcher
    )
    assert resolved == 2

    rows = sorted(ledger_service.recent_rows(db_path=db), key=lambda r: r.target_date)
    assert rows[0].hit is True
    assert rows[0].actual_price == 4020.0
    assert rows[1].hit is False
    assert rows[1].actual_price == 3990.0


def test_backfill_down_direction(tmp_path):
    db = tmp_path / "ledger.sqlite"
    made = datetime(2026, 4, 10, 9, 30, tzinfo=UTC)
    bundle = _bundle("INFY.NS", made, [_point(date(2026, 4, 11), 1450.0, "down")])
    ledger_service.log_prediction(bundle, anchor_price=1500.0, db_path=db)

    def fetcher(ticker, lo, hi):
        return {date(2026, 4, 11): 1480.0}  # actual < anchor → down was right

    ledger_service.backfill_actuals(
        up_to=date(2026, 4, 15), db_path=db, price_fetcher=fetcher
    )
    rows = ledger_service.recent_rows(db_path=db)
    assert rows[0].hit is True


def test_backfill_skips_unreached_target_dates(tmp_path):
    db = tmp_path / "ledger.sqlite"
    made = datetime(2026, 4, 10, 9, 30, tzinfo=UTC)
    bundle = _bundle(
        "TCS.NS",
        made,
        [
            _point(date(2026, 4, 11), 4050.0, "up"),
            _point(date(2026, 4, 20), 4200.0, "up"),  # future — shouldn't resolve
        ],
    )
    ledger_service.log_prediction(bundle, anchor_price=4000.0, db_path=db)

    def fetcher(ticker, lo, hi):
        return {date(2026, 4, 11): 4050.0}

    resolved = ledger_service.backfill_actuals(
        up_to=date(2026, 4, 15), db_path=db, price_fetcher=fetcher
    )
    assert resolved == 1

    rows = sorted(ledger_service.recent_rows(db_path=db), key=lambda r: r.target_date)
    assert rows[0].actual_price == 4050.0
    assert rows[1].actual_price is None


def test_accuracy_window_computes_hit_rate_and_brier(tmp_path):
    db = tmp_path / "ledger.sqlite"
    made = datetime(2026, 4, 10, 9, 30, tzinfo=UTC)

    # Three predictions, two hits, all with prob_up=0.7
    points = [
        _point(date(2026, 4, 11), 101.0, "up", prob_up=0.7),
        _point(date(2026, 4, 12), 102.0, "up", prob_up=0.7),
        _point(date(2026, 4, 13), 103.0, "up", prob_up=0.7),
    ]
    bundle = _bundle("ABC.NS", made, points)
    ledger_service.log_prediction(bundle, anchor_price=100.0, db_path=db)

    actuals = {
        date(2026, 4, 11): 101.0,  # up hit
        date(2026, 4, 12): 99.5,   # up miss
        date(2026, 4, 13): 102.0,  # up hit
    }

    def fetcher(ticker, lo, hi):
        return actuals

    ledger_service.backfill_actuals(
        up_to=date(2026, 4, 15), db_path=db, price_fetcher=fetcher
    )

    window = ledger_service.accuracy_window(
        "ABC.NS", days=30, db_path=db, now=date(2026, 4, 15)
    )
    assert window.n_resolved == 3
    assert window.directional_accuracy == pytest.approx(2 / 3, abs=1e-6)
    # Brier = mean((0.7 - truth)^2) with truths [1, 0, 1] = (0.09 + 0.49 + 0.09)/3
    assert window.brier_score == pytest.approx((0.09 + 0.49 + 0.09) / 3, abs=1e-6)
    assert window.mae_price is not None and window.mae_price > 0


def test_accuracy_window_empty_when_no_resolved(tmp_path):
    db = tmp_path / "ledger.sqlite"
    made = datetime(2026, 4, 10, 9, 30, tzinfo=UTC)
    bundle = _bundle("XYZ.NS", made, [_point(date(2026, 4, 20), 50.0, "up")])
    ledger_service.log_prediction(bundle, anchor_price=48.0, db_path=db)

    window = ledger_service.accuracy_window(
        "XYZ.NS", days=30, db_path=db, now=date(2026, 4, 12)
    )
    assert window.n_predictions == 1
    assert window.n_resolved == 0
    assert window.directional_accuracy is None


def test_log_from_future_df_adapter(tmp_path):
    import pandas as pd

    db = tmp_path / "ledger.sqlite"
    idx = pd.DatetimeIndex([datetime(2026, 4, 11), datetime(2026, 4, 12)])
    future_df = pd.DataFrame(
        {"Predicted Price": [101.0, 102.5], "P5": [99.0, 99.5], "P95": [103.0, 105.0]},
        index=idx,
    )
    n = ledger_service.log_from_future_df(
        "TCS.NS",
        future_df,
        anchor_price=100.0,
        prob_up=0.65,
        model_version="test-v1",
        db_path=db,
    )
    assert n == 2

    rows = sorted(ledger_service.recent_rows(db_path=db), key=lambda r: r.target_date)
    assert rows[0].ci_low == 99.0 and rows[0].ci_high == 103.0
    assert rows[0].prob_up == pytest.approx(0.65)
    assert rows[0].pred_dir == "up"


def test_backfill_uses_anchor_cache_when_row_missing_anchor(tmp_path):
    """
    Rows logged before the anchor_price column existed fall back to
    `_anchor_close` via the injected fetcher — cache prevents double calls.
    """
    import sqlite3

    db = tmp_path / "ledger.sqlite"
    # Log normally, then blank out anchor_price to simulate legacy rows.
    made = datetime(2026, 4, 10, 9, 30, tzinfo=UTC)
    bundle = _bundle("LEG.NS", made, [_point(date(2026, 4, 11), 55.0, "up")])
    ledger_service.log_prediction(bundle, anchor_price=50.0, db_path=db)

    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE predictions SET anchor_price = NULL")
        conn.commit()

    call_count = {"n": 0}

    def fetcher(ticker, lo, hi):
        call_count["n"] += 1
        # First call = target-date lookup, second = anchor lookup
        if date(2026, 4, 11) in _date_range(lo, hi):
            return {date(2026, 4, 11): 52.0}
        return {date(2026, 4, 10): 50.0}

    ledger_service.backfill_actuals(
        up_to=date(2026, 4, 15), db_path=db, price_fetcher=fetcher
    )
    rows = ledger_service.recent_rows(db_path=db)
    assert rows[0].actual_price == 52.0
    assert rows[0].hit is True  # 52 > 50 anchor
    assert call_count["n"] >= 2  # one for target, at least one for anchor


def _date_range(lo: date, hi: date) -> list[date]:
    out = []
    d = lo
    while d <= hi:
        out.append(d)
        d += timedelta(days=1)
    return out
