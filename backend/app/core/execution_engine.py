from __future__ import annotations

from typing import Dict, Optional
import logging

from app.core.broker_interface import BrokerAdapter, PaperBroker, Order
from app.core.risk_manager import RiskManager
from app.core.config import settings

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Prepare execution flow for Level 7 (broker integration later)."""

    def __init__(
        self,
        broker: Optional[BrokerAdapter] = None,
        capital: float = 100000000,
        risk_per_trade: float = 0.02
    ):
        self.broker = broker or PaperBroker(starting_cash=capital)
        self.risk_manager = RiskManager(capital=capital, risk_per_trade=risk_per_trade)

    def build_order(self, signal: Dict, current_price: Optional[float] = None) -> Optional[Order]:
        action = signal.get("action")
        if action not in ("buy", "sell"):
            return None

        entry = signal.get("entry_zone") or current_price
        stop_loss = signal.get("stop_loss_1")
        if entry is None:
            return None

        quantity = 0
        if stop_loss is not None:
            quantity = self.risk_manager.position_size(entry, stop_loss)

        return Order(
            symbol=signal.get("symbol", ""),
            side=action,
            quantity=quantity,
            order_type=signal.get("action_type", "market"),
            limit_price=entry if signal.get("action_type") == "limit" else None,
            stop_price=entry if signal.get("action_type") == "stop" else None,
            metadata={"risk_reward": signal.get("risk_reward")}
        )

    def execute_signal(self, signal: Dict, current_price: Optional[float] = None) -> Dict:
        order = self.build_order(signal, current_price=current_price)
        if order is None:
            return {"status": "skipped", "reason": "no_action"}

        if settings.BROKER_MODE == "disabled":
            logger.info("Broker mode disabled; execution skipped")
            return {"status": "skipped", "reason": "broker_disabled", "order": order.__dict__}

        return self.broker.place_order(order)
