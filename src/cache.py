import redis
import hashlib
import numpy as np
import logging
from .config import Config
import time
logger = logging.getLogger(__name__)
class RedisCache:
    def __init__(self, config:Config):
        self.redis_client = redis.Redis(
            host = config.redis_host,
            port = config.redis_port,
            username = config.redis_username,
            password = config.redis_password
        )
        self.response_cache_ttl = config.response_cache_ttl

    def _embed_key(self, text: str) -> str:
        return f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
    
    def get_embedding(self, text:str)-> np.ndarray | None:
        raw = self.redis_client.get(self._embed_key(text))
        if raw:
            return np.frombuffer(raw, dtype=np.float32)
        return None
    
    def set_embedding(self, text: str, vector: np.ndarray) -> None:
        self.redis_client.set(
            self._embed_key(text),
            vector.astype(np.float32).tobytes()
        )
    def get_or_embed(self, text:str, encoder):
        start = time.time()
        cached = self.get_embedding(text)
        if cached is not None:
            return cached
        vector = encoder.encode(text)
        print("Embedding query in: ",time.time()-start)
        self.set_embedding(text, vector)
        return vector
    
    # ── Response cache ───────────────────────────────────────────

    def _response_key(self, query: str) -> str:
        return f"response:{hashlib.md5(query.encode()).hexdigest()}"

    def get_response(self, query: str) -> str | None:
        raw = self.redis_client.get(self._response_key(query))
        logger.info("getting response from the cache")
        return raw.decode() if raw else None

    def set_response(self, query: str, response: str) -> None:
        start = time.time()
        self.redis_client.setex(
            self._response_key(query),
            self.response_cache_ttl,
            response,
        )
        print("Setting response: ",time.time()-start)