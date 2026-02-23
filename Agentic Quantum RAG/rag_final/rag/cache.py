"""
rag/cache.py - Two-level semantic cache
L1: in-process LRU OrderedDict (fast, per-process)
L2: Redis HNSW vector index (persistent, cross-session)
Gracefully falls back to L1-only when Redis is unavailable.
"""
import hashlib, json, time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import numpy as np
from utils.config import cfg

REDIS_INDEX  = "rag_semantic_cache"
REDIS_PREFIX = "rag:cache:"

@dataclass
class CacheEntry:
    query: str; response: str; sources: List[str]
    timestamp: float = field(default_factory=time.time)

class RedisSemanticCache:
    def __init__(self):
        self._threshold = cfg.CACHE_SIMILARITY_THRESHOLD
        self._max_l1    = cfg.CACHE_MAX_L1
        self._ttl       = cfg.REDIS_CACHE_TTL
        self._l1: OrderedDict[str, Tuple[np.ndarray, CacheEntry]] = OrderedDict()
        self._redis = None; self._ok = False
        self._connect()
        if self._ok:
            self._ensure_index()

    def _connect(self):
        try:
            import redis
            r = redis.Redis(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT,
                password=cfg.REDIS_PASSWORD or None, db=cfg.REDIS_DB,
                decode_responses=False, socket_connect_timeout=2, socket_timeout=2)
            r.ping(); self._redis = r; self._ok = True
        except Exception:
            pass

    def _ensure_index(self):
        try:
            from redis.commands.search.field import VectorField, TextField, NumericField
            from redis.commands.search.indexDefinition import IndexDefinition, IndexType
            try:
                self._redis.ft(REDIS_INDEX).info()
            except Exception:
                self._redis.ft(REDIS_INDEX).create_index(
                    [TextField("query"), TextField("response"), TextField("sources"),
                     NumericField("timestamp"),
                     VectorField("embedding","HNSW",{"TYPE":"FLOAT32","DIM":cfg.EMBEDDING_DIM,
                         "DISTANCE_METRIC":"COSINE","M":16,"EF_CONSTRUCTION":200})],
                    definition=IndexDefinition(prefix=[REDIS_PREFIX], index_type=IndexType.HASH))
        except Exception:
            pass

    @staticmethod
    def _md5(t): return hashlib.md5(t.encode()).hexdigest()
    @staticmethod
    def _to_bytes(v): return v.astype(np.float32).tobytes()
    @staticmethod
    def _from_bytes(b): return np.frombuffer(b, dtype=np.float32)
    @staticmethod
    def _cosine(a, b):
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a,b)/d) if d else 0.0

    def _l1_evict(self):
        while len(self._l1) >= self._max_l1:
            self._l1.popitem(last=False)

    def get(self, query: str, emb: Optional[np.ndarray]=None) -> Optional[Tuple[str,List[str]]]:
        key = self._md5(query)
        if key in self._l1:
            self._l1.move_to_end(key)
            return self._l1[key][1].response, self._l1[key][1].sources
        if emb is not None:
            for k,(vec,entry) in self._l1.items():
                if self._cosine(emb,vec) >= self._threshold:
                    self._l1.move_to_end(k)
                    return entry.response, entry.sources
        if not self._ok: return None
        if emb is not None:
            try:
                from redis.commands.search.query import Query
                q = (Query("*=>[KNN 5 @embedding $vec AS score]").sort_by("score")
                     .return_fields("query","response","sources","timestamp","embedding","score").dialect(2))
                res = self._redis.ft(REDIS_INDEX).search(q, query_params={"vec": self._to_bytes(emb)})
                for doc in res.docs:
                    if 1.0 - float(getattr(doc,"score",2.0)) >= self._threshold:
                        response = doc.response; sources = json.loads(doc.sources)
                        self._l1_evict()
                        self._l1[key] = (self._from_bytes(doc.embedding),
                            CacheEntry(doc.query,response,sources,float(doc.timestamp)))
                        return response, sources
            except Exception: pass
        try:
            raw = self._redis.hgetall(REDIS_PREFIX+key)
            if raw:
                response = raw[b"response"].decode(); sources = json.loads(raw[b"sources"].decode())
                self._l1_evict()
                self._l1[key] = (self._from_bytes(raw[b"embedding"]),
                    CacheEntry(raw[b"query"].decode(),response,sources,float(raw[b"timestamp"])))
                return response, sources
        except Exception: pass
        return None

    def set(self, query: str, emb: np.ndarray, response: str, sources: List[str]):
        key = self._md5(query); entry = CacheEntry(query,response,sources)
        self._l1_evict(); self._l1[key] = (emb,entry)
        if not self._ok: return
        try:
            pipe = self._redis.pipeline(transaction=False)
            pipe.hset(REDIS_PREFIX+key, mapping={"query":query,"response":response,
                "sources":json.dumps(sources),"timestamp":str(entry.timestamp),
                "embedding":self._to_bytes(emb)})
            if self._ttl > 0: pipe.expire(REDIS_PREFIX+key, self._ttl)
            pipe.execute()
        except Exception: pass

    def stats(self) -> Dict:
        info = {"l1_entries": len(self._l1), "redis": self._ok}
        if self._ok:
            try:
                idx = self._redis.ft(REDIS_INDEX).info()
                info["l2_entries"] = idx.get("num_docs","?")
            except Exception: info["l2_entries"] = "?"
        return info

    def flush(self):
        self._l1.clear()
        if self._ok:
            try:
                keys = self._redis.keys(REDIS_PREFIX+"*")
                if keys: self._redis.delete(*keys)
            except Exception: pass

cache = RedisSemanticCache()
