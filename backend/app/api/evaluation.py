from fastapi import APIRouter
from app.core.database import trades_collection
import pandas as pd
import numpy as np

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

@router.get("/metrics")
def get_metrics():
    trades = list(trades_collection.find({}, {"_id": 0, "profit": 1, "entry_date": 1, "exit_date": 1}))
    if not trades:
        return {"error": "No trades found"}
    df = pd.DataFrame(trades)
    
    win_rate = (df['profit'] > 0).mean()
    total_profit = df[df['profit'] > 0]['profit'].sum()
    total_loss = abs(df[df['profit'] < 0]['profit'].sum())
    profit_factor = (total_profit / total_loss) if total_loss != 0 else None
    
    # Hitung drawdown
    df['cumulative'] = df['profit'].cumsum()
    running_max = df['cumulative'].cummax()
    drawdown = running_max - df['cumulative']
    max_drawdown = drawdown.max()
    
    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "total_trades": len(df),
        "total_profit": df['profit'].sum()
    }
