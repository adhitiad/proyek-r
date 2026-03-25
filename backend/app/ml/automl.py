import logging
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

from app.core.database import db
from app.ml.trainer import ModelTrainer

logger = logging.getLogger(__name__)

DEFAULT_TRIALS: List[Dict[str, Any]] = [
    {"epochs": 60, "batch_size": 32, "learning_rate": 0.001},
    {"epochs": 100, "batch_size": 32, "learning_rate": 0.001},
    {"epochs": 120, "batch_size": 64, "learning_rate": 0.0005},
]


class AutoMLSelector:
    """Lightweight AutoML-style selector for model hyperparameters."""

    def __init__(self, symbols, start_date, end_date, target_days: int = 5):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.target_days = target_days

    def run(
        self,
        trials: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        trial_list = trials or DEFAULT_TRIALS
        results: List[Dict[str, Any]] = []

        for params in trial_list:
            try:
                trainer = ModelTrainer(
                    self.symbols,
                    self.start_date,
                    self.end_date,
                    self.target_days
                )
                metadata, accuracy = trainer.train(
                    epochs=params.get("epochs", 100),
                    batch_size=params.get("batch_size", 32),
                    learning_rate=params.get("learning_rate", 0.001)
                )
                results.append({
                    "accuracy": accuracy,
                    "metadata": metadata,
                    "params": params
                })
            except Exception as e:
                logger.error(f"AutoML trial failed {params}: {e}")
                results.append({
                    "accuracy": 0,
                    "metadata": None,
                    "params": params,
                    "error": str(e)
                })

        valid = [r for r in results if r.get("metadata")]
        if not valid:
            raise ValueError("AutoML failed: no successful trials")

        best = max(valid, key=lambda x: x["accuracy"])
        self._persist_run(best, results)
        self._activate_if_better(best)

        return best["metadata"], best["accuracy"], results

    def _persist_run(self, best: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
        try:
            db.automl_runs.insert_one({
                "timestamp": datetime.now(),
                "symbols": self.symbols,
                "start_date": str(self.start_date),
                "end_date": str(self.end_date),
                "target_days": self.target_days,
                "best": best,
                "results": results
            })
        except Exception as e:
            logger.warning(f"Failed to persist AutoML run: {e}")

    def _activate_if_better(self, best: Dict[str, Any]) -> None:
        try:
            active_config = db.config.find_one({"key": "active_model"})
            if active_config:
                active_meta = db.model_metadata.find_one({"model_path": active_config["value"]})
                if active_meta and best["accuracy"] <= active_meta.get("accuracy", 0):
                    return
            db.config.update_one(
                {"key": "active_model"},
                {"$set": {"value": best["metadata"]["model_path"], "updated_at": datetime.now()}},
                upsert=True
            )
        except Exception as e:
            logger.warning(f"AutoML activate failed: {e}")
