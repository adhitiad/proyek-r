from fastapi import APIRouter, HTTPException, Depends
from app.core.backtester import Backtester
from app.core.screener import Screener
from app.core.database import db
from app.core.security import require_api_key
import asyncio
import datetime

router = APIRouter(prefix="/backtest", tags=["backtest"])

@router.post("/run", dependencies=[Depends(require_api_key)])
async def run_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: int = 100000000,
    risk_per_trade: float = 0.02,
    commission: float = 0.001,
    slippage: float = 0.0005,
):
    try:
        bt = Backtester(symbol, start_date, end_date, initial_capital, risk_per_trade, commission, slippage)
        result = await asyncio.to_thread(bt.run)
        # Simpan ke database
        result['timestamp'] = datetime.datetime.now()
        db.backtest_results.insert_one(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/screen", dependencies=[Depends(require_api_key)])
async def screen(
    symbols: list[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100000000,
    risk_per_trade: float = 0.02,
    commission: float = 0.001,
    slippage: float = 0.0005
):
    try:
        screener = Screener(symbols, start_date, end_date,
                            initial_capital=initial_capital,
                            risk_per_trade=risk_per_trade,
                            commission=commission,
                            slippage=slippage)
        results = await asyncio.to_thread(screener.run)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/metrics/{symbol}")
async def get_backtest_metrics(
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: int= 100000000
):
    """
    Mendapatkan metrik detail untuk suatu saham tanpa perlu menjalankan optimasi.
    """
    try:
        bt = Backtester(symbol, start_date, end_date, initial_capital)
        result = await asyncio.to_thread(bt.run)
        return {
            'symbol': symbol,
            'period': f"{start_date} to {end_date}",
            'performance_metrics': {
                'total_return': f"{result['total_return']:.2%}",
                'cagr': f"{result['cagr']:.2%}",
                'sharpe_ratio': f"{result['sharpe_ratio']:.2f}",
                'max_drawdown': f"{result['max_drawdown']:.2%}",
                'avg_drawdown': f"{result['avg_drawdown']:.2%}",
                'max_drawdown_duration': f"{result['max_drawdown_duration_days']} days",
                'avg_drawdown_duration': f"{result['avg_drawdown_duration_days']} days"
            },
            'trading_metrics': {
                'num_trades': result['num_trades'],
                'win_rate': f"{result['win_rate']:.2%}",
                'loss_rate': f"{result['loss_rate']:.2%}",
                'break_even_rate': f"{result['break_even_rate']:.2%}",
                'profit_factor': f"{result['profit_factor']:.2f}",
                'avg_win': f"Rp {result['avg_win']:,.0f}",
                'avg_loss': f"Rp {result['avg_loss']:,.0f}",
                'avg_win_percent': f"{result['avg_win_percent']:.2%}",
                'avg_loss_percent': f"{result['avg_loss_percent']:.2%}",
                'win_loss_ratio': f"{result['win_loss_ratio']:.2f}",
                'expectancy': f"Rp {result['expectancy']:,.0f}",
                'max_consecutive_wins': result['max_consecutive_wins'],
                'max_consecutive_losses': result['max_consecutive_losses'],
                'recovery_factor': f"{result['recovery_factor']:.2f}"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trades/{symbol}")
async def get_trades_history(
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: int = 100000000,
    limit: int = 100
):
    """
    Mendapatkan history trades untuk analisis lebih detail.
    """
    try:
        bt = Backtester(symbol, start_date, end_date, initial_capital)
        result = await asyncio.to_thread(bt.run)
        trades = result['trades'][-limit:]  # ambil terakhir limit
        return {
            'symbol': symbol,
            'total_trades': len(result['trades']),
            'trades': trades
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
