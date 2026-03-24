from pymongo import MongoClient
from app.core.config import settings

client = MongoClient(settings.MONGODB_URL)
db = client[settings.DATABASE_NAME]

# Collections
signals_collection = db["signals"]
trades_collection = db["trades"]
db.screening_cache.create_index("expires_at", expireAfterSeconds=0)
db.screening_cache.create_index("cache_key", unique=True)