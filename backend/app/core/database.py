from pymongo import MongoClient, errors
from app.core.config import settings

client = MongoClient(settings.MONGODB_URL, serverSelectionTimeoutMS=5000)
db = client[settings.DATABASE_NAME]

# Collections
signals_collection = db["signals"]
trades_collection = db["trades"]

def _init_indexes():
    try:
        db.screening_cache.create_index("expires_at", expireAfterSeconds=0)
        db.screening_cache.create_index("cache_key", unique=True)
    except errors.PyMongoError as exc:
        # Avoid crashing app startup when Mongo is temporarily unavailable
        print(f"[WARN] Mongo index init skipped: {exc}")

_init_indexes()
