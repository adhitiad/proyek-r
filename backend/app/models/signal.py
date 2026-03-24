from pydantic import BaseModel
from typing import Optional

class SignalResponse(BaseModel):
    symbol: str
    bias: str  # bullish/bearish/neutral
    action: str  # buy/sell/hold
    action_type: str  # limit/stop/market
    entry_zone: float
    stop_loss_1: Optional[float]
    stop_loss_2: Optional[float]
    take_profit_1: Optional[float]
    take_profit_2: Optional[float]
    risk_reward: str
    probability: int
    notes: str