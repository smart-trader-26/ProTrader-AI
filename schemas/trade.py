"""
Trade Desk DTOs (P5) — broker-gated, confirmation-first order flow.

Lifecycle of a proposal:

    PROPOSED  → user clicked "Analyze": the swing signal fired and a sized
                order (qty, entry, stop, exit-by) is waiting for approval.
    PLACED    → user explicitly approved; the order went to the active broker
                (or the dry-run simulator when live trading is off).
    REJECTED  → user declined.
    FAILED    → approval was given but the broker call errored.
    EXPIRED   → proposal went stale (prices move; old entries are not honoured).

Nothing is ever sent to a broker without an explicit approve call, and the
approve path is the ONLY code that talks to the broker's order API.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

PROPOSAL_STATUSES = ("PROPOSED", "PLACED", "REJECTED", "FAILED", "EXPIRED")


class BrokerStatus(BaseModel):
    """Which execution backend an approval would hit right now."""

    broker: str = "dry_run"            # "kite" | "dry_run"
    live_trading: bool = False         # LIVE_TRADING flag AND broker configured
    configured: bool = False           # broker credentials present
    note: str = ""


class TradeProposal(BaseModel):
    id: int
    created_at: datetime
    ticker: str
    side: str = "BUY"                  # long/neutral system — always BUY
    qty: int
    entry_price: float                 # last close at proposal time (reference)
    capital_required: float
    stop_price: float | None = None    # disaster stop (10% below entry)
    target_price: float | None = None  # None = hold to horizon (how the edge was measured)
    exit_by: str                       # ISO date — time exit at the signal horizon
    engine: str = "swing30d"
    prob_up: float = Field(ge=0.0, le=1.0)
    tau_star: float | None = None
    expected_hit_rate: float | None = None
    status: str = "PROPOSED"
    dry_run: bool | None = None        # set at placement time
    broker: str | None = None
    broker_order_id: str | None = None
    gtt_id: str | None = None
    note: str = ""


class SkippedTicker(BaseModel):
    """Ticker analysed but NOT proposed — with the honest reason why."""

    ticker: str
    reason: str
    prob_up: float | None = None


class ProposalRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=25)
    capital_per_slot: float | None = Field(default=None, gt=0)


class ProposalCreateResponse(BaseModel):
    proposals: list[TradeProposal] = Field(default_factory=list)
    skipped: list[SkippedTicker] = Field(default_factory=list)
