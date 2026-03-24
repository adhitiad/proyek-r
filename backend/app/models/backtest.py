from pydantic import BaseModel
from typing import List, Optional
import datetime

class TradeRecord(BaseModel):
    symbol: str
    entry_date: datetime.datetime
    exit_date: datetime.datetime
    action: str
    entry_price: float
    exit_price: float
    shares: int
    profit: float
    profit_pct: float
    exit_reason: str
    action_type: str

class BacktestResult(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    cagr: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    expectancy: float
    trades: List[TradeRecord]

class PerformanceMetrics(BaseModel):
    total_return: str
    cagr: str
    sharpe_ratio: str
    max_drawdown: str
    avg_drawdown: str
    max_drawdown_duration: str
    avg_drawdown_duration: str

class TradingMetrics(BaseModel):
    num_trades: int
    win_rate: str
    loss_rate: str
    break_even_rate: str
    profit_factor: str
    avg_win: str
    avg_loss: str
    avg_win_percent: str
    avg_loss_percent: str
    win_loss_ratio: str
    expectancy: str
    max_consecutive_wins: int
    max_consecutive_losses: int
    recovery_factor: str

class BacktestMetricsResponse(BaseModel):
    symbol: str
    period: str
    performance_metrics: PerformanceMetrics
    trading_metrics: TradingMetrics