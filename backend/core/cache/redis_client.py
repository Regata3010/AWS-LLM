# backend/core/cache/redis_client.py
"""
Redis client for BiasGuard caching and session management
"""

import redis
import json
import os
from typing import Optional, Any
from core.src.logger import logging

class RedisClient:
    """Simple Redis client with error handling"""
    
    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        
        try:
            self.client = redis.from_url(
                redis_url,
                decode_responses=True,  # Return strings instead of bytes
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.client.ping()
            logging.info(f" Redis connected: {redis_url}")
            self.enabled = True
        except Exception as e:
            logging.warning(f" Redis unavailable: {e}. Caching disabled.")
            self.client = None
            self.enabled = False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        if not self.enabled:
            return None
        
        try:
            value = self.client.get(key)
            if value:
                # Try to parse JSON
                try:
                    return json.loads(value)
                except:
                    return value
            return None
        except Exception as e:
            logging.error(f"Redis GET error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """
        Set value in Redis with TTL
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON-encoded if dict/list)
            ttl: Time to live in seconds (default 5 minutes)
        """
        if not self.enabled:
            return False
        
        try:
            # Serialize to JSON if dict/list
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            self.client.setex(key, ttl, value)
            return True
        except Exception as e:
            logging.error(f"Redis SET error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        if not self.enabled:
            return False
        
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logging.error(f"Redis DELETE error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        if not self.enabled:
            return False
        
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logging.error(f"Redis EXISTS error: {e}")
            return False
    
    def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter (useful for rate limiting)"""
        if not self.enabled:
            return None
        
        try:
            return self.client.incrby(key, amount)
        except Exception as e:
            logging.error(f"Redis INCR error: {e}")
            return None
    
    def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for key"""
        if not self.enabled:
            return None
        
        try:
            return self.client.ttl(key)
        except Exception as e:
            logging.error(f"Redis TTL error: {e}")
            return None
    
    def flush_all(self) -> bool:
        """Clear all cache (use with caution!)"""
        if not self.enabled:
            return False
        
        try:
            self.client.flushdb()
            logging.info("Redis cache cleared")
            return True
        except Exception as e:
            logging.error(f"Redis FLUSH error: {e}")
            return False


# Singleton instance
_redis_client: Optional[RedisClient] = None

def get_redis() -> RedisClient:
    """Get or create Redis client singleton"""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client