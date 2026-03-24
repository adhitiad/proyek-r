from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.optimizer import ParameterOptimizer
from app.core.database import db
import datetime
import uuid

router = APIRouter(prefix="/optimize", tags=["optimize"])

def run_optimization_task(opt_id, symbol, start_date, end_date, param_grid, metric, initial_capital, commission, slippage):
    optimizer = ParameterOptimizer(symbol, start_date, end_date, initial_capital, commission, slippage)
    result = optimizer.optimize(param_grid, metric)
    result['optimization_id'] = opt_id
    result['timestamp'] = datetime.datetime.now()
    db.optimization_results.insert_one(result)

@router.post("/start")
async def start_optimization(
    background_tasks: BackgroundTasks,
    symbol: str,
    start_date: str,
    end_date: str,
    param_grid: dict,
    metric: str = "sharpe_ratio",
    initial_capital: float = 100000000,
    commission: float = 0.001,
    slippage: float = 0.0005
):
    opt_id = str(uuid.uuid4())
    background_tasks.add_task(
        run_optimization_task,
        opt_id, symbol, start_date, end_date, param_grid, metric,
        initial_capital, commission, slippage
    )
    return {"optimization_id": opt_id, "status": "started"}

@router.get("/results/{opt_id}")
async def get_optimization_results(opt_id: str):
    result = db.optimization_results.find_one({"optimization_id": opt_id}, {"_id": 0})
    if not result:
        raise HTTPException(status_code=404, detail="Optimization not found")
    return result