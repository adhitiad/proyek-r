from fastapi import APIRouter, HTTPException, Depends, Body
from pydantic import BaseModel
from app.ml.trainer import ModelTrainer
from app.ml.automl import AutoMLSelector
from app.core.database import db
from app.core.security import require_api_key
from datetime import datetime, timedelta
import asyncio
from typing import Optional, Any

router = APIRouter(prefix="/model", tags=["model"])

class AutoMLRequest(BaseModel):
    symbols: Optional[list[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    target_days: Optional[int] = None
    trials: Optional[list[dict[str, Any]]] = None

@router.post("/train", dependencies=[Depends(require_api_key)])
async def train_model(
    symbols: list[str] = ["BBCA.JK", "BBRI.JK", "ASII.JK", "TLKM.JK"],
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    target_days: int = 5,
    epochs: int = 100
):
    try:
        trainer = ModelTrainer(symbols, start_date, end_date, target_days)
        metadata, accuracy = await asyncio.to_thread(trainer.train, epochs=epochs)
        return {"message": "Training completed", "accuracy": accuracy, "metadata": metadata}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/automl", dependencies=[Depends(require_api_key)])
async def run_automl(
    symbols: list[str] = ["BBCA.JK", "BBRI.JK", "ASII.JK", "TLKM.JK"],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    target_days: int = 5,
    payload: Optional[AutoMLRequest] = Body(default=None)
):
    try:
        req = payload or AutoMLRequest()
        final_symbols = req.symbols or symbols
        final_target_days = req.target_days or target_days
        end = datetime.fromisoformat(req.end_date) if req.end_date else (
            datetime.fromisoformat(end_date) if end_date else datetime.now()
        )
        start = datetime.fromisoformat(req.start_date) if req.start_date else (
            datetime.fromisoformat(start_date) if start_date else end - timedelta(days=365)
        )
        selector = AutoMLSelector(final_symbols, start.isoformat(), end.isoformat(), final_target_days)
        metadata, accuracy, trials = selector.run(trials=req.trials)
        return {
            "message": "AutoML completed",
            "accuracy": accuracy,
            "metadata": metadata,
            "trials": trials
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/automl/latest")
async def get_latest_automl():
    run = db.automl_runs.find_one(sort=[("timestamp", -1)])
    if not run:
        return {"latest": None}

    run["_id"] = str(run["_id"])
    if isinstance(run.get("timestamp"), datetime):
        run["timestamp"] = run["timestamp"].isoformat()
    return {"latest": run}

@router.get("/list")
async def list_models():
    models = list(db.model_metadata.find({}, {"_id": 0}).sort("timestamp", -1))
    return models

@router.post("/activate", dependencies=[Depends(require_api_key)])
async def activate_model(model_path: str):
    model = db.model_metadata.find_one({"model_path": model_path})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    db.config.update_one(
        {"key": "active_model"},
        {"$set": {"value": model_path, "updated_at": datetime.now()}},
        upsert=True
    )
    return {"message": f"Model {model_path} activated"}

@router.get("/active")
async def get_active_model():
    config = db.config.find_one({"key": "active_model"})
    if not config:
        return {"active_model": None}
    model_path = config["value"]
    model_meta = db.model_metadata.find_one({"model_path": model_path},
                                             {"_id": 0})
    return {"active_model": model_path, "metadata": model_meta}
