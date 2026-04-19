"""
Broker-agnostic paper-trading service (A9.2).

Design principle: the fill *source* is an interface. Today we quote fills
against live yfinance (15-min delayed but good enough for end-of-day
backtest-vs-live reconciliation). When Upstox KYC lands, swap `_yf_fill`
for `_upstox_sandbox_fill` — signature is identical.

The service reuses **A7 ledger + A8 cost model**:
    • Every paper fill appends a row to a separate `paper_fills` SQLite
      table in the same ledger DB.
    • P&L is computed with `models.nse_costs.round_trip_cost_fraction` so
      the live book's numbers match what the backtester reports.

This is NOT a broker. It never calls `place_order`. It just simulates
what would have happened if the model's signals had been traded, and
logs that to disk so you can compare with paper/real fills later.

Usage:
    svc = PaperTradeService()
    svc.on_signal("TCS.NS", prob_up=0.72, threshold=0.55,
                  stop_pct=0.02, target_pct=0.04)
    svc.mark_to_market()   # end-of-bar: check stops/targets vs live price
    state = svc.book_state()
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from schemas.paper_trade import PaperBookState, PaperFill, PaperPosition
from services.ledger_service import DEFAULT_DB_PATH, _connect

FillSource = Callable[[str], float | None]  # ticker -> latest price or None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_fills (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker         TEXT NOT NULL,
    opened_at      TEXT NOT NULL,
    closed_at      TEXT,
    side           TEXT NOT NULL,
    qty            INTEGER NOT NULL,
    entry_price    REAL NOT NULL,
    exit_price     REAL,
    gross_pnl      REAL DEFAULT 0,
    costs          REAL DEFAULT 0,
    net_pnl        REAL DEFAULT 0,
    reason_entry   TEXT,
    reason_exit    TEXT
);
CREATE INDEX IF NOT EXISTS idx_fills_ticker ON paper_fills(ticker);

CREATE TABLE IF NOT EXISTS paper_positions (
    ticker         TEXT PRIMARY KEY,
    side           TEXT NOT NULL,
    qty            INTEGER NOT NULL,
    entry_price    REAL NOT NULL,
    opened_at      TEXT NOT NULL,
    stop_price     REAL,
    target_price   REAL
);
"""

_LOCK = threading.Lock()


class PaperTradeService:
    """One book, many tickers. Thread-safe for concurrent signal writes."""

    def __init__(
        self,
        db_path: Path | None = None,
        starting_cash: float = 1_000_000.0,
        fill_source: FillSource | None = None,
    ):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.starting_cash = starting_cash
        self.fill_source = fill_source or default_fill_source
        self._ensure_schema()

    # ─────────────────── public API ───────────────────

    def on_signal(
        self,
        ticker: str,
        prob_up: float,
        threshold: float = 0.55,
        qty: int = 1,
        stop_pct: float = 0.02,
        target_pct: float = 0.04,
        reason: str = "",
    ) -> PaperPosition | None:
        """
        Consume a model signal. Opens a long if `prob_up > threshold` and
        the ticker is flat; flips to flat on the opposite signal. Returns
        the position after the action, or None if no action was taken.
        """
        price = self.fill_source(ticker)
        if price is None or price <= 0:
            return None

        pos = self._get_position(ticker)

        # Exit an existing long when prob_up drops below threshold
        if pos and pos.side == "long" and prob_up < threshold:
            self._close(pos, exit_price=price, reason_exit=reason or "signal_flip")
            return self._get_position(ticker)

        # Open a new long
        if (pos is None or pos.side == "flat") and prob_up > threshold:
            stop = price * (1 - stop_pct)
            target = price * (1 + target_pct)
            self._open(
                ticker=ticker,
                side="long",
                qty=qty,
                entry_price=price,
                stop_price=stop,
                target_price=target,
                reason_entry=reason or f"prob_up={prob_up:.2f}>{threshold:.2f}",
            )
            return self._get_position(ticker)

        return pos

    def mark_to_market(self) -> list[PaperFill]:
        """
        For each open position: fetch latest price, check stop/target hit,
        close if breached. Returns the list of newly-closed fills.
        """
        closed: list[PaperFill] = []
        for pos in self._open_positions():
            price = self.fill_source(pos.ticker)
            if price is None:
                continue
            if pos.side == "long":
                if pos.stop_price and price <= pos.stop_price:
                    closed.append(self._close(pos, price, "stop_hit"))
                elif pos.target_price and price >= pos.target_price:
                    closed.append(self._close(pos, price, "target_hit"))
        return closed

    def book_state(self) -> PaperBookState:
        with _connect(self.db_path) as conn:
            realised = conn.execute(
                "SELECT COALESCE(SUM(net_pnl), 0) AS s FROM paper_fills WHERE closed_at IS NOT NULL"
            ).fetchone()["s"] or 0.0
            n_fills = conn.execute(
                "SELECT COUNT(*) AS n FROM paper_fills WHERE closed_at IS NOT NULL"
            ).fetchone()["n"] or 0
            open_rows = conn.execute("SELECT * FROM paper_positions").fetchall()

        unrealised = 0.0
        for r in open_rows:
            price = self.fill_source(r["ticker"])
            if price is None:
                continue
            mult = 1 if r["side"] == "long" else -1
            unrealised += (price - r["entry_price"]) * r["qty"] * mult

        cash = self.starting_cash + realised
        return PaperBookState(
            cash=cash,
            realised_pnl=realised,
            unrealised_pnl=unrealised,
            n_fills=int(n_fills),
            n_open=len(open_rows),
            equity=cash + unrealised,
        )

    def recent_fills(self, limit: int = 50, ticker: str | None = None) -> list[PaperFill]:
        where, params = ("WHERE ticker = ?", [ticker]) if ticker else ("", [])
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT * FROM paper_fills {where} ORDER BY id DESC LIMIT ?",
                [*params, limit],
            ).fetchall()
        return [_row_to_fill(r) for r in rows]

    # ─────────────────── internals ───────────────────

    def _ensure_schema(self) -> None:
        with _connect(self.db_path) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _get_position(self, ticker: str) -> PaperPosition | None:
        with _connect(self.db_path) as conn:
            r = conn.execute(
                "SELECT * FROM paper_positions WHERE ticker = ?", (ticker,)
            ).fetchone()
        if not r:
            return None
        return _row_to_position(r)

    def _open_positions(self) -> list[PaperPosition]:
        with _connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM paper_positions").fetchall()
        return [_row_to_position(r) for r in rows]

    def _open(
        self,
        ticker: str,
        side: str,
        qty: int,
        entry_price: float,
        stop_price: float,
        target_price: float,
        reason_entry: str,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with _LOCK, _connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_positions
                    (ticker, side, qty, entry_price, opened_at, stop_price, target_price)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ticker, side, qty, entry_price, now, stop_price, target_price),
            )
            conn.execute(
                """
                INSERT INTO paper_fills
                    (ticker, opened_at, side, qty, entry_price, reason_entry)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ticker, now, side, qty, entry_price, reason_entry),
            )
            conn.commit()

    def _close(self, pos: PaperPosition, exit_price: float, reason_exit: str) -> PaperFill:
        from models.nse_costs import DELIVERY, round_trip_cost_fraction

        now = datetime.now(UTC).isoformat()
        notional = pos.qty * pos.entry_price
        mult = 1 if pos.side == "long" else -1
        gross = (exit_price - pos.entry_price) * pos.qty * mult
        costs = round_trip_cost_fraction(notional=notional, costs=DELIVERY) * notional
        net = gross - costs

        with _LOCK, _connect(self.db_path) as conn:
            # Close the *oldest* open fill for this ticker (there's at most one).
            conn.execute(
                """
                UPDATE paper_fills SET
                    closed_at  = ?,
                    exit_price = ?,
                    gross_pnl  = ?,
                    costs      = ?,
                    net_pnl    = ?,
                    reason_exit = ?
                WHERE id = (
                    SELECT id FROM paper_fills
                    WHERE ticker = ? AND closed_at IS NULL
                    ORDER BY id ASC LIMIT 1
                )
                """,
                (now, exit_price, gross, costs, net, reason_exit, pos.ticker),
            )
            conn.execute("DELETE FROM paper_positions WHERE ticker = ?", (pos.ticker,))
            conn.commit()

        return PaperFill(
            ticker=pos.ticker,
            opened_at=pos.opened_at or datetime.now(UTC),
            closed_at=datetime.now(UTC),
            side=pos.side,
            qty=pos.qty,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            gross_pnl=float(gross),
            costs=float(costs),
            net_pnl=float(net),
            reason_exit=reason_exit,
        )


# ─────────────────── fill sources ───────────────────

def _yf_fill(ticker: str) -> float | None:
    """Latest close from yfinance — 15-min delayed, keyless."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        hist = yf.Ticker(ticker).history(period="1d", interval="1m")
    except Exception:
        return None
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    return float(hist["Close"].iloc[-1])


def default_fill_source(ticker: str) -> float | None:
    """
    Auto-pick fill source at call time.

    Upstox → LTP when UPSTOX_ACCESS_TOKEN + a mapped instrument key exist;
    yfinance otherwise. Every miss falls through to the next source so a
    half-configured Upstox install still returns a price.
    """
    try:
        from data.upstox_client import get_ltp

        price = get_ltp(ticker)
        if price is not None and price > 0:
            return price
    except Exception:
        pass
    return _yf_fill(ticker)


# ─────────────────── row adapters ───────────────────

def _row_to_position(r: sqlite3.Row) -> PaperPosition:
    return PaperPosition(
        ticker=r["ticker"],
        side=r["side"],
        qty=r["qty"],
        entry_price=r["entry_price"],
        opened_at=_parse_dt(r["opened_at"]),
        stop_price=r["stop_price"],
        target_price=r["target_price"],
    )


def _row_to_fill(r: sqlite3.Row) -> PaperFill:
    return PaperFill(
        ticker=r["ticker"],
        opened_at=_parse_dt(r["opened_at"]),
        closed_at=_parse_dt(r["closed_at"]) if r["closed_at"] else None,
        side=r["side"],
        qty=r["qty"],
        entry_price=r["entry_price"],
        exit_price=r["exit_price"],
        gross_pnl=r["gross_pnl"] or 0.0,
        costs=r["costs"] or 0.0,
        net_pnl=r["net_pnl"] or 0.0,
        reason_exit=r["reason_exit"] or "",
    )


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# Re-export for convenience
__all__ = [
    "PaperTradeService",
    "PaperFill",
    "PaperPosition",
    "PaperBookState",
    "date",
    "timedelta",
]
