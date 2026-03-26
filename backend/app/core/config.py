import os
from pathlib import Path
from dotenv import load_dotenv

# Prefer repo root .env for local dev, fallback to backend/.env
BASE_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BASE_DIR.parent if BASE_DIR.name == "backend" else BASE_DIR

load_dotenv(REPO_ROOT / ".env", override=False)
load_dotenv(BASE_DIR / ".env", override=False)

class Settings:
    MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "forex_idx_signals")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
    API_KEY = os.getenv("API_KEY", "123456")
    DATA_PERIOD = os.getenv("DATA_PERIOD", "1y")
    DATA_INTERVAL = os.getenv("DATA_INTERVAL", "1h")
    TECH_TIMEFRAMES = [
        tf.strip()
        for tf in os.getenv("TECH_TIMEFRAMES", "1h,4h,1d").split(",")
        if tf.strip()
    ]
    ENABLE_AUTOML = os.getenv("ENABLE_AUTOML", "true").lower() in ("1", "true", "yes")
    BROKER_MODE = os.getenv("BROKER_MODE", "disabled")  # disabled|paper|live
    DISABLE_SCHEDULER = os.getenv("DISABLE_SCHEDULER", "false").lower() in ("1", "true", "yes")
    BANDAR_DISABLE_RTI = os.getenv("BANDAR_DISABLE_RTI", "true").lower() in ("1", "true", "yes")

settings = Settings()
