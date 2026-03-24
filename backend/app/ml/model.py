import torch
import joblib
import os
from datetime import datetime
from app.core.database import db
from app.ml.trainer import SignalModel

class ProbabilityModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.active_model_path = None
        self.last_load_time = None
        self.load_active_model()

    def _load_model(self, model_path):
        meta = db.model_metadata.find_one({"model_path": model_path})
        if not meta:
            return False
        scaler_path = meta["scaler_path"]
        if not os.path.exists(scaler_path) or not os.path.exists(model_path):
            return False
        self.scaler = joblib.load(scaler_path)
        self.feature_cols = meta["feature_cols"]
        input_dim = meta["input_dim"]
        self.model = SignalModel(input_dim)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        self.active_model_path = model_path
        return True

    def load_active_model(self, force=False):
        config = db.config.find_one({"key": "active_model"})
        if not config:
            return
        model_path = config["value"]
        if not force and self.active_model_path == model_path:
            return
        self._load_model(model_path)
        self.last_load_time = datetime.now()

    def predict(self, features):
        # Reload jika sudah lebih dari 60 detik (agar dinamis)
        if self.last_load_time and (datetime.now() - self.last_load_time).seconds > 60:
            self.load_active_model()
        if self.model is None or self.scaler is None:
            return 50
        if len(features) != len(self.feature_cols):
            raise ValueError(f"Expected {len(self.feature_cols)} features, got {len(features)}")
        scaled = self.scaler.transform([features])
        tensor = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            proba = self.model(tensor).item()
        return int(proba * 100)

    def get_active_model_info(self):
        return {
            "model_path": self.active_model_path,
            "feature_cols": self.feature_cols
        }