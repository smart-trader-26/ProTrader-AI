"""
Tests for the P5 trade-proposal queue.

No network, no broker: history + signal + broker are all stubbed. The same
tmp SQLite file backs both the proposals table and the accuracy ledger, so
the honesty-loop write is asserted directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import services.trade_proposal_service as tps
from services.broker_service import BrokerError, DryRunBroker


class FakeStats:
    tau_star = 0.63
    oos_precision = 0.606
    oos_base_rate = 0.58
    oos_coverage = 0.056


class FakeSignal:
    horizon = 30
    stats = FakeStats()

    def __init__(self, trained=True):
        self._trained = trained

    def is_trained(self):
        return self._trained

    def predict(self, close, ticker=""):
        last = float(close.iloc[-1])
        p = min(0.9, max(0.1, last / 200.0))
        return {"signal": "UP" if p >= 0.63 else "NEUTRAL", "prob_up": p}


def _series(level: float, bars: int = 300) -> pd.Series:
    idx = pd.bdate_range("2024-01-01", periods=bars)
    return pd.Series(np.full(bars, level), index=idx)


@pytest.fixture
def db(tmp_path):
    return tmp_path / "ledger.sqlite"


@pytest.fixture
def stubbed(monkeypatch):
    levels = {"AAA.NS": 150.0, "BBB.NS": 80.0, "CCC.NS": 90_000.0}
    monkeypatch.setattr(tps, "_get_signal", lambda: FakeSignal(trained=True))
    monkeypatch.setattr(tps, "_close_series", lambda t: _series(levels.get(t, 150.0)))
    return levels


def test_create_fires_sizes_and_skips(db, stubbed):
    res = tps.create_proposals(
        ["AAA.NS", "BBB.NS", "CCC.NS"], capital_per_slot=25_000, db_path=db
    )
    # AAA fires (p=0.75); BBB neutral (p=0.40); CCC fires but price > capital.
    assert [p.ticker for p in res.proposals] == ["AAA.NS"]
    p = res.proposals[0]
    assert p.status == "PROPOSED"
    assert p.qty == 166                      # floor(25000 / 150)
    assert p.entry_price == 150.0
    assert p.stop_price == pytest.approx(135.0)   # -10%
    assert p.target_price is None            # hold to horizon
    assert p.capital_required == pytest.approx(166 * 150.0)
    assert p.expected_hit_rate == pytest.approx(0.606)

    reasons = {s.ticker: s.reason for s in res.skipped}
    assert "below the conviction threshold" in reasons["BBB.NS"]
    assert "exceeds capital per position" in reasons["CCC.NS"]


def test_untrained_model_proposes_nothing(db, monkeypatch):
    monkeypatch.setattr(tps, "_get_signal", lambda: FakeSignal(trained=False))
    res = tps.create_proposals(["AAA.NS"], db_path=db)
    assert res.proposals == []
    assert res.skipped[0].reason == "swing model not trained"


def test_reanalysis_supersedes_open_proposal(db, stubbed):
    first = tps.create_proposals(["AAA.NS"], capital_per_slot=25_000, db_path=db)
    second = tps.create_proposals(["AAA.NS"], capital_per_slot=25_000, db_path=db)
    rows = tps.list_proposals(db_path=db)
    statuses = {r.id: r.status for r in rows}
    assert statuses[first.proposals[0].id] == "EXPIRED"
    assert statuses[second.proposals[0].id] == "PROPOSED"


def test_approve_places_dry_run_and_logs_to_ledger(db, stubbed, monkeypatch):
    monkeypatch.setattr(
        "services.broker_service.get_broker", lambda: (DryRunBroker(), False)
    )
    created = tps.create_proposals(["AAA.NS"], capital_per_slot=25_000, db_path=db)
    pid = created.proposals[0].id

    placed = tps.approve(pid, db_path=db)
    assert placed.status == "PLACED"
    assert placed.dry_run is True
    assert placed.broker == "dry_run"
    assert placed.broker_order_id.startswith("DRY-")

    # Honesty loop: the call landed in the SAME accuracy ledger.
    from services import ledger_service
    rows = ledger_service.recent_rows(ticker="AAA.NS", db_path=db)
    assert len(rows) == 1
    assert rows[0].model_version.startswith("swing30d-dry")
    assert rows[0].pred_dir == "up"
    assert rows[0].prob_up == pytest.approx(0.75)
    assert rows[0].target_date.isoformat() == placed.exit_by


def test_approve_marks_failed_on_broker_error(db, stubbed, monkeypatch):
    class BoomBroker:
        name = "kite"

        def place_buy_with_protection(self, **kw):
            raise BrokerError("token expired")

    monkeypatch.setattr(
        "services.broker_service.get_broker", lambda: (BoomBroker(), True)
    )
    created = tps.create_proposals(["AAA.NS"], capital_per_slot=25_000, db_path=db)
    pid = created.proposals[0].id

    with pytest.raises(BrokerError):
        tps.approve(pid, db_path=db)
    row = tps.list_proposals(db_path=db)[0]
    assert row.status == "FAILED"
    assert "token expired" in row.note


def test_reject_and_double_action_guard(db, stubbed):
    created = tps.create_proposals(["AAA.NS"], capital_per_slot=25_000, db_path=db)
    pid = created.proposals[0].id

    rejected = tps.reject(pid, db_path=db)
    assert rejected.status == "REJECTED"
    with pytest.raises(ValueError):
        tps.approve(pid, db_path=db)
    with pytest.raises(ValueError):
        tps.reject(pid, db_path=db)


def test_stale_proposals_expire(db, stubbed):
    created = tps.create_proposals(["AAA.NS"], capital_per_slot=25_000, db_path=db)
    pid = created.proposals[0].id

    # Backdate the row past the staleness window.
    with tps._conn(db) as conn:
        conn.execute(
            "UPDATE trade_proposals SET created_at = '2020-01-01T00:00:00+00:00' "
            "WHERE id = ?",
            (pid,),
        )
        conn.commit()

    rows = tps.list_proposals(db_path=db)  # triggers expire_stale
    assert rows[0].status == "EXPIRED"
    with pytest.raises(ValueError):
        tps.approve(pid, db_path=db)
