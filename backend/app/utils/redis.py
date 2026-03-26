"""Basic connection example.
"""

import redis

_redis_instance = None

def get_redis():
    """Get or create Redis connection instance"""
    global _redis_instance
    if _redis_instance is None:
        _redis_instance = redis.Redis(
            host='redis-15783.c334.asia-southeast2-1.gce.cloud.redislabs.com',
            port=15783,
            decode_responses=True,
            username="default",
            password="GlKSAasJujbD68J53qqE8Ds8DHyZtb6Y",
        )
    return _redis_instance

# Keep dbredis as alias for backward compatibility
def dbredis():
    return get_redis()


