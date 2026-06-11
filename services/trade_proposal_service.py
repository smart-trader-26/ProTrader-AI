"""
Trade-proposal queue (P5) — analyze → propose → human approves → execute.

The contract, in one sentence: the model may *suggest* orders, but only an
explicit `approve()` call ever reaches a broker, and approving in dry-run
mode (the default) only simulates the fill.

Flow:
    create_proposals(tickers)   runs the 30-day swing signal on each ticker.
        • signal fires UP  → a sized proposal row (qty, entry, 10% disaster
          stop, exit-by date at the horizon) with status PROPOSED
        • anything else    → returned in `skipped` with the honest reason
    approve(id)                 places the order via the active broker
                                (services.broker_service) and — the honesty
                                loop — logs the call into the SAME accuracy
                                ledger the paper trades use, tagged
                                `swing30d-live[...]`, so live hits/misses are
                                resolved by the existing backfill and become
                                directly comparable to paper and backtest.
    reject(id)                  marks it declined; nothing happens.

Proposals expire after _STALE_DAYS — entry/stop prices are quotes from
proposal time and are not honoured once the market has moved on.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import threading
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from schemas.prediction import PredictionBundle, PredictionPoint
from schemas.trade import (
    ProposalCreateResponse,
    SkippedTicker,
    TradeProposal,
)
from services import ledger_service
from services.ledger_service import DEFAULT_DB_PATH, _connect

logger = logging.getLogger(__name__)

_DISASTER_STOP_PCT = 0.10     # matches the swing paper-trade engine
_STALE_DAYS = 3               # proposals older than this cannot be approved
_MIN_BARS = 260               # signal needs ~252 bars of history

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trade_proposals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    side            TEXT NOT NULL DEFAULT 'BUY',
    qty             INTEGER NOT NULL,
    entry_price     REAL NOT NULL,
    capital_required REAL NOT NULL,
    stop_price      REAL,
    target_price    REAL,
    exit_by         TEXT NOT NULL,
    engine          TEXT NOT NULL DEFAULT 'swing30d',
    prob_up         REAL NOT NULL,
    tau_star        REAL,
    expected_hit_rate REAL,
    status          TEXT NOT NULL DEFAULT 'PROPOSED',
    dry_run         INTEGER,
    broker          TEXT,
    broker_order_id TEXT,
    gtt_id          TEXT,
    note            TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON trade_proposals(status);
CREATE INDEX IF NOT EXISTS idx_proposals_ticker ON trade_proposals(ticker);
"""

_LOCK = threading.Lock()


def _conn(db_path: Path | None = None) -> sqlite3.Connection:
    conn = _connect(db_path or DEFAULT_DB_PATH)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


# ── analysis inputs (injectable for tests) ───────────────────────────────────

def _close_series(ticker: str):
    """~2y of closes for the swing signal. Returns a pd.Series or None."""
    from services.stock_service import get_history_df

    end = date.today()
    start = end - timedelta(days=820)
    df = get_history_df(ticker, start, end)
    if df is None or df.empty or "Close" not in df.columns:
        return None
    return df["Close"]


def _get_signal():
    from models.directional_signal import get_signal

    return get_signal()


# ── public API ────────────────────────────────────────────────────────────────

def create_proposals(
    tickers: list[str],
    capital_per_slot: float | None = None,
    db_path: Path | None = None,
) -> ProposalCreateResponse:
    """
    Run the swing signal on each ticker and stage sized proposals for the
    ones that fire. Existing un-actioned (PROPOSED) rows for a re-analyzed
    ticker are expired first — prices move, so a fresh quote wins.
    """
    if capital_per_slot is None:
        try:
            from config.settings import TRADE_CAPITAL_PER_SLOT
            capital_per_slot = float(TRADE_CAPITAL_PER_SLOT)
        except Exception:
            capital_per_slot = 25_000.0

    sig = _get_signal()
    proposals: list[TradeProposal] = []
    skipped: list[SkippedTicker] = []

    if not sig.is_trained():
        return ProposalCreateResponse(
            proposals=[],
            skipped=[SkippedTicker(ticker=t, reason="swing model not trained")
                     for t in tickers],
        )

    horizon = int(sig.horizon)
    exit_by = (date.today() + timedelta(days=int(horizon * 7 / 5))).isoformat()

    for raw in tickers:
        ticker = raw.strip().upper()
        if not ticker:
            continue
        try:
            close = _close_series(ticker)
            if close is None or len(close.dropna()) < _MIN_BARS:
                skipped.append(SkippedTicker(
                    ticker=ticker,
                    reason=f"insufficient history (need ~{_MIN_BARS} bars)",
                ))
                continue
            close = close.dropna()
            out = sig.predict(close, ticker=ticker)
            prob = float(out.get("prob_up", 0.5))
            if out.get("signal") != "UP":
                skipped.append(SkippedTicker(
                    ticker=ticker,
                    reason=(
                        f"NEUTRAL — P(up over {horizon}d) = {prob:.1%} is below "
                        f"the conviction threshold {sig.stats.tau_star:.0%}; "
                        "no measured edge, no trade"
                    ),
                    prob_up=prob,
                ))
                continue

            entry = float(close.iloc[-1])
            qty = int(math.floor(capital_per_slot / entry))
            if qty < 1:
                skipped.append(SkippedTicker(
                    ticker=ticker,
                    reason=(
                        f"price ₹{entry:,.2f} exceeds capital per position "
                        f"₹{capital_per_slot:,.0f} — raise the allocation"
                    ),
                    prob_up=prob,
                ))
                continue

            proposals.append(_insert_proposal(
                ticker=ticker,
                qty=qty,
                entry=entry,
                stop=round(entry * (1 - _DISASTER_STOP_PCT), 2),
                target=None,  # hold to horizon — how the edge was measured
                exit_by=exit_by,
                prob_up=prob,
                tau_star=float(sig.stats.tau_star),
                expected_hit_rate=float(sig.stats.oos_precision) or None,
                note=(
                    f"swing30d UP @ {prob:.1%} — historically right "
                    f"{sig.stats.oos_precision:.0%} of the time it fires"
                ),
                db_path=db_path,
            ))
        except Exception as exc:  # noqa: BLE001 — one bad ticker can't kill the batch
            logger.warning("create_proposals(%s) failed: %s", ticker, exc)
            skipped.append(SkippedTicker(
                ticker=ticker, reason=f"analysis error: {type(exc).__name__}"
            ))

    return ProposalCreateResponse(proposals=proposals, skipped=skipped)


def list_proposals(
    status: str | None = None,
    limit: int = 100,
    db_path: Path | None = None,
) -> list[TradeProposal]:
    expire_stale(db_path=db_path)
    where, params = ("WHERE status = ?", [status]) if status else ("", [])
    with _conn(db_path) as conn:
        rows = conn.execute(
            f"SELECT * FROM trade_proposals {where} "
            f"ORDER BY id DESC LIMIT ?",
            [*params, limit],
        ).fetchall()
    return [_row_to_dto(r) for r in rows]


def approve(proposal_id: int, db_path: Path | None = None) -> TradeProposal:
    """
    Execute an approved proposal via the active broker.

    Raises ValueError when the proposal is missing / not approvable, and
    BrokerError when the broker call fails (the row is marked FAILED first
    so the UI never shows a phantom pending order).
    """
    from services.broker_service import BrokerError, get_broker

    expire_stale(db_path=db_path)
    row = _get(proposal_id, db_path)
    if row is None:
        raise ValueError(f"proposal {proposal_id} not found")
    if row.status != "PROPOSED":
        raise ValueError(f"proposal {proposal_id} is {row.status}, not PROPOSED")

    broker, live = get_broker()
    try:
        res = broker.place_buy_with_protection(
            ticker=row.ticker,
            qty=row.qty,
            last_price=row.entry_price,
            stop_price=row.stop_price,
            target_price=row.target_price,
        )
    except BrokerError as exc:
        _update(proposal_id, db_path,
                status="FAILED", broker=broker.name,
                dry_run=int(not live), note=str(exc))
        raise

    _update(
        proposal_id, db_path,
        status="PLACED",
        broker=res.broker,
        broker_order_id=res.order_id,
        gtt_id=res.gtt_id,
        dry_run=int(res.dry_run),
        note=res.note,
    )

    # ── The honesty loop ─────────────────────────────────────────────────────
    # Log this live (or dry-run) call into the SAME accuracy ledger as the
    # paper trades. backfill_actuals resolves it into Hit/Miss at exit_by, so
    # "live vs paper vs backtest" reads off one table, filtered by
    # model_version. A failure here never un-places the order.
    try:
        updated = _get(proposal_id, db_path)
        if updated is not None:
            _log_to_ledger(updated, db_path=db_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ledger log failed for proposal %s: %s", proposal_id, exc)

    return _get(proposal_id, db_path)  # type: ignore[return-value]


def reject(proposal_id: int, db_path: Path | None = None) -> TradeProposal:
    row = _get(proposal_id, db_path)
    if row is None:
        raise ValueError(f"proposal {proposal_id} not found")
    if row.status != "PROPOSED":
        raise ValueError(f"proposal {proposal_id} is {row.status}, not PROPOSED")
    _update(proposal_id, db_path, status="REJECTED", note="declined by user")
    return _get(proposal_id, db_path)  # type: ignore[return-value]


def expire_stale(db_path: Path | None = None) -> int:
    """Expire PROPOSED rows older than _STALE_DAYS. Idempotent."""
    cutoff = (datetime.now(UTC) - timedelta(days=_STALE_DAYS)).isoformat()
    with _LOCK, _conn(db_path) as conn:
        cur = conn.execute(
            "UPDATE trade_proposals SET status = 'EXPIRED', "
            "note = 'expired — entry quote went stale' "
            "WHERE status = 'PROPOSED' AND created_at < ?",
            (cutoff,),
        )
        conn.commit()
        return cur.rowcount or 0


# ── internals ────────────────────────────────────────────────────────────────

def _insert_proposal(
    ticker: str, qty: int, entry: float, stop: float | None,
    target: float | None, exit_by: str, prob_up: float,
    tau_star: float | None, expected_hit_rate: float | None,
    note: str, db_path: Path | None,
) -> TradeProposal:
    now = datetime.now(UTC).isoformat()
    with _LOCK, _conn(db_path) as conn:
        # A fresh analysis supersedes any un-actioned proposal for the ticker.
        conn.execute(
            "UPDATE trade_proposals SET status = 'EXPIRED', "
            "note = 'superseded by a newer analysis' "
            "WHERE ticker = ? AND status = 'PROPOSED'",
            (ticker,),
        )
        cur = conn.execute(
            """
            INSERT INTO trade_proposals
                (created_at, ticker, side, qty, entry_price, capital_required,
                 stop_price, target_price, exit_by, engine, prob_up, tau_star,
                 expected_hit_rate, status, note)
            VALUES (?, ?, 'BUY', ?, ?, ?, ?, ?, ?, 'swing30d', ?, ?, ?, 'PROPOSED', ?)
            """,
            (now, ticker, qty, entry, round(qty * entry, 2), stop, target,
             exit_by, prob_up, tau_star, expected_hit_rate, note),
        )
        conn.commit()
        new_id = cur.lastrowid
    return _get(new_id, db_path)  # type: ignore[return-value]


def _log_to_ledger(p: TradeProposal, db_path: Path | None = None) -> None:
    mode = "live" if p.dry_run is False else "dry"
    bundle = PredictionBundle(
        ticker=p.ticker,
        made_at=datetime.now(UTC),
        model_version=f"swing30d-{mode}[{p.broker or 'dry_run'}]",
        horizon_days=max(1, int((date.fromisoformat(p.exit_by) - date.today()).days)),
        points=[
            PredictionPoint(
                target_date=date.fromisoformat(p.exit_by),
                pred_price=p.entry_price,   # direction is the call, not the level
                direction="up",
                prob_up=p.prob_up,
            )
        ],
    )
    ledger_service.log_prediction(bundle, anchor_price=p.entry_price, db_path=db_path)


def _get(proposal_id: int, db_path: Path | None = None) -> TradeProposal | None:
    with _conn(db_path) as conn:
        r = conn.execute(
            "SELECT * FROM trade_proposals WHERE id = ?", (proposal_id,)
        ).fetchone()
    return _row_to_dto(r) if r else None


def _update(proposal_id: int, db_path: Path | None, **fields) -> None:
    cols = ", ".join(f"{k} = ?" for k in fields)
    with _LOCK, _conn(db_path) as conn:
        conn.execute(
            f"UPDATE trade_proposals SET {cols} WHERE id = ?",
            (*fields.values(), proposal_id),
        )
        conn.commit()


def _row_to_dto(r: sqlite3.Row) -> TradeProposal:
    return TradeProposal(
        id=r["id"],
        created_at=datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")),
        ticker=r["ticker"],
        side=r["side"],
        qty=r["qty"],
        entry_price=r["entry_price"],
        capital_required=r["capital_required"],
        stop_price=r["stop_price"],
        target_price=r["target_price"],
        exit_by=r["exit_by"],
        engine=r["engine"],
        prob_up=r["prob_up"],
        tau_star=r["tau_star"],
        expected_hit_rate=r["expected_hit_rate"],
        status=r["status"],
        dry_run=None if r["dry_run"] is None else bool(r["dry_run"]),
        broker=r["broker"],
        broker_order_id=r["broker_order_id"],
        gtt_id=r["gtt_id"],
        note=r["note"] or "",
    )
