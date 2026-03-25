from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class Order:
    symbol: str
    side: str  # buy/sell
    quantity: float
    order_type: str = "market"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "gtc"
    metadata: Dict = field(default_factory=dict)


class BrokerAdapter(ABC):
    """Abstract broker adapter. Concrete broker integrations plug in here."""

    @abstractmethod
    def place_order(self, order: Order) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def get_position(self, symbol: str) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def get_account(self) -> Dict:
        raise NotImplementedError


class PaperBroker(BrokerAdapter):
    """Minimal paper broker for dry-run executions."""

    def __init__(self, starting_cash: float = 100000000):
        self.cash = starting_cash
        self.positions: Dict[str, float] = {}
        self.orders: List[Order] = []

    def place_order(self, order: Order) -> Dict:
        self.orders.append(order)
        return {"status": "accepted", "order": order.__dict__}

    def get_position(self, symbol: str) -> Dict:
        return {"symbol": symbol, "quantity": self.positions.get(symbol, 0)}

    def get_account(self) -> Dict:
        return {"cash": self.cash, "positions": self.positions}
