from fastapi import APIRouter, HTTPException, Depends
from app.ml.trainer import ModelTrainer
from app.core.database import db
from app.core.security import require_api_key
from datetime import datetime
import asyncio

router = APIRouter(prefix="/model", tags=["model"])

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
