import os
from dotenv import load_dotenv

load_dotenv()

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

settings = Settings()
