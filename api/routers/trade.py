"""
Trade Desk router (P5) — confirmation-gated order flow.

  GET  /api/v1/trade/broker/status         → which backend approvals hit (dry-run/kite)
  POST /api/v1/trade/proposals             → analyze tickers, stage sized proposals
  GET  /api/v1/trade/proposals?status=     → list proposals (newest first)
  POST /api/v1/trade/proposals/{id}/approve → place the order (the ONLY execution path)
  POST /api/v1/trade/proposals/{id}/reject  → decline

Safety model: orders are simulated unless LIVE_TRADING=1 AND Kite credentials
are configured. Approve/reject are explicit per-order user actions — nothing
is ever auto-executed.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from schemas.trade import (
    BrokerStatus,
    ProposalCreateResponse,
    ProposalRequest,
    TradeProposal,
)
from services import trade_proposal_service as svc

router = APIRouter(prefix="/trade", tags=["trade"])


@router.get("/broker/status", response_model=BrokerStatus,
            summary="Active execution backend (dry-run vs live Kite)")
def broker_status() -> BrokerStatus:
    from services.broker_service import broker_status as _status

    return _status()


@router.post("/proposals", response_model=ProposalCreateResponse,
             summary="Analyze tickers with the swing signal and stage proposals")
def create_proposals(req: ProposalRequest) -> ProposalCreateResponse:
    return svc.create_proposals(req.tickers, capital_per_slot=req.capital_per_slot)


@router.get("/proposals", response_model=list[TradeProposal],
            summary="List trade proposals (newest first)")
def list_proposals(
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[TradeProposal]:
    return svc.list_proposals(status=status, limit=limit)


@router.post("/proposals/{proposal_id}/approve", response_model=TradeProposal,
             summary="Approve a proposal — places the order via the active broker")
def approve(proposal_id: int) -> TradeProposal:
    from services.broker_service import BrokerError

    try:
        return svc.approve(proposal_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except BrokerError as exc:
        raise HTTPException(status_code=502, detail=f"Broker error: {exc}") from exc


@router.post("/proposals/{proposal_id}/reject", response_model=TradeProposal,
             summary="Reject a proposal — nothing is placed")
def reject(proposal_id: int) -> TradeProposal:
    try:
        return svc.reject(proposal_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
