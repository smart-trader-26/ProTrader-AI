"""
Broker adapters (P5) — the ONLY module allowed to talk to a broker's order API.

Two backends behind one interface:

  • DryRunBroker  — always available. Returns simulated order ids. This is the
                    default; nothing leaves the machine.
  • KiteBroker    — Zerodha Kite Connect. Activated ONLY when all three of
                    KITE_API_KEY, KITE_ACCESS_TOKEN and LIVE_TRADING=1 are set
                    (see `get_broker`). Places a CNC market buy plus a
                    server-side GTT for the protective stop (OCO with target
                    when one is given) so exits survive without this process
                    running.

Free-tier-first invariant: with no keys configured everything still works in
dry-run mode, and the `kiteconnect` package is imported lazily so it is not a
hard dependency.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class BrokerError(RuntimeError):
    """Raised when an order/GTT placement fails at the broker."""


@dataclass
class OrderResult:
    broker: str
    order_id: str
    gtt_id: str | None = None
    dry_run: bool = True
    note: str = ""


class DryRunBroker:
    """Simulated execution — used whenever live trading is not fully enabled."""

    name = "dry_run"

    def is_configured(self) -> bool:
        return True

    def place_buy_with_protection(
        self,
        ticker: str,
        qty: int,
        last_price: float,
        stop_price: float | None,
        target_price: float | None,
    ) -> OrderResult:
        oid = f"DRY-{uuid.uuid4().hex[:10].upper()}"
        gtt = f"DRY-GTT-{uuid.uuid4().hex[:8].upper()}" if stop_price else None
        logger.info(
            "DRY-RUN order: BUY %d %s @~%.2f (stop=%s target=%s) -> %s",
            qty, ticker, last_price, stop_price, target_price, oid,
        )
        return OrderResult(
            broker=self.name, order_id=oid, gtt_id=gtt, dry_run=True,
            note="simulated — live trading is OFF",
        )


class KiteBroker:
    """Zerodha Kite Connect adapter. CNC (delivery) only, long only."""

    name = "kite"

    def __init__(self, api_key: str | None = None, access_token: str | None = None):
        if api_key is None or access_token is None:
            from config.settings import KITE_ACCESS_TOKEN, KITE_API_KEY
            api_key = api_key if api_key is not None else KITE_API_KEY
            access_token = access_token if access_token is not None else KITE_ACCESS_TOKEN
        self._api_key = api_key or ""
        self._access_token = access_token or ""
        self._kite = None

    def is_configured(self) -> bool:
        return bool(self._api_key and self._access_token)

    def _client(self):
        if self._kite is None:
            try:
                from kiteconnect import KiteConnect
            except ImportError as exc:  # pragma: no cover - env dependent
                raise BrokerError(
                    "kiteconnect not installed — pip install kiteconnect"
                ) from exc
            kite = KiteConnect(api_key=self._api_key)
            kite.set_access_token(self._access_token)
            self._kite = kite
        return self._kite

    @staticmethod
    def _split_symbol(ticker: str) -> tuple[str, str]:
        """'TCS.NS' → ('NSE', 'TCS'); '.BO' → BSE. Bare symbols default to NSE."""
        t = ticker.strip().upper()
        if t.endswith(".NS"):
            return "NSE", t[:-3]
        if t.endswith(".BO"):
            return "BSE", t[:-3]
        return "NSE", t

    def place_buy_with_protection(
        self,
        ticker: str,
        qty: int,
        last_price: float,
        stop_price: float | None,
        target_price: float | None,
    ) -> OrderResult:
        """
        Market CNC buy, then a GTT sell for protection:
          • stop only       → single-trigger GTT at the stop
          • stop + target   → two-leg OCO GTT (one cancels the other)

        GTT lives on Zerodha's servers (free, valid ~1 year), so the stop
        survives even when this app isn't running — the right shape for a
        position held ~30 trading days.
        """
        if not self.is_configured():
            raise BrokerError("Kite is not configured (KITE_API_KEY / KITE_ACCESS_TOKEN)")
        kite = self._client()
        exchange, symbol = self._split_symbol(ticker)

        try:
            order_id = kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=exchange,
                tradingsymbol=symbol,
                transaction_type=kite.TRANSACTION_TYPE_BUY,
                quantity=int(qty),
                product=kite.PRODUCT_CNC,
                order_type=kite.ORDER_TYPE_MARKET,
            )
        except Exception as exc:
            raise BrokerError(f"Kite order failed: {exc}") from exc

        gtt_id: str | None = None
        if stop_price:
            sell_leg = {
                "exchange": exchange,
                "tradingsymbol": symbol,
                "transaction_type": kite.TRANSACTION_TYPE_SELL,
                "quantity": int(qty),
                "product": kite.PRODUCT_CNC,
                "order_type": kite.ORDER_TYPE_LIMIT,
            }
            try:
                if target_price:
                    trigger = kite.place_gtt(
                        trigger_type=kite.GTT_TYPE_OCO,
                        tradingsymbol=symbol,
                        exchange=exchange,
                        trigger_values=[float(stop_price), float(target_price)],
                        last_price=float(last_price),
                        orders=[
                            {**sell_leg, "price": round(float(stop_price) * 0.995, 2)},
                            {**sell_leg, "price": round(float(target_price), 2)},
                        ],
                    )
                else:
                    trigger = kite.place_gtt(
                        trigger_type=kite.GTT_TYPE_SINGLE,
                        tradingsymbol=symbol,
                        exchange=exchange,
                        trigger_values=[float(stop_price)],
                        last_price=float(last_price),
                        orders=[{**sell_leg, "price": round(float(stop_price) * 0.995, 2)}],
                    )
                gtt_id = str(trigger.get("trigger_id", "")) or None
            except Exception as exc:
                # The buy went through but protection didn't — surface loudly;
                # the caller marks the proposal so the user can place the stop
                # manually rather than silently holding an unprotected position.
                logger.error("Kite GTT placement failed after buy %s: %s", order_id, exc)
                return OrderResult(
                    broker=self.name, order_id=str(order_id), gtt_id=None,
                    dry_run=False,
                    note=f"ORDER PLACED but GTT stop FAILED ({exc}) — set the stop manually in Kite",
                )

        return OrderResult(
            broker=self.name, order_id=str(order_id), gtt_id=gtt_id,
            dry_run=False, note="live order placed",
        )


def get_broker() -> tuple[DryRunBroker | KiteBroker, bool]:
    """
    Resolve the active execution backend.

    Live (KiteBroker) ONLY when LIVE_TRADING=1 AND Kite credentials exist;
    every other combination falls back to the dry-run simulator. Returns
    (broker, is_live).
    """
    try:
        from config.settings import LIVE_TRADING
    except Exception:
        LIVE_TRADING = False
    kite = KiteBroker()
    if LIVE_TRADING and kite.is_configured():
        return kite, True
    return DryRunBroker(), False


def broker_status() -> "BrokerStatus":
    from schemas.trade import BrokerStatus

    try:
        from config.settings import LIVE_TRADING
    except Exception:
        LIVE_TRADING = False
    kite = KiteBroker()
    broker, live = get_broker()
    if live:
        note = "LIVE — approved orders go to Zerodha Kite"
    elif kite.is_configured():
        note = "Kite keys present but LIVE_TRADING is off — orders are simulated"
    elif LIVE_TRADING:
        note = "LIVE_TRADING is on but Kite keys are missing — orders are simulated"
    else:
        note = "Dry-run mode — orders are simulated, nothing reaches a broker"
    return BrokerStatus(
        broker=broker.name,
        live_trading=live,
        configured=kite.is_configured(),
        note=note,
    )
