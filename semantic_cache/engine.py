from typing import Dict, List, Optional
import redis
from pydantic import BaseModel
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer
from common.env import REDIS_URL, CACHE_NAME, CACHE_DISTANCE_THRESHOLD

class CacheResult(BaseModel):
    prompt: str
    response: str
    vector_distance: float
    cosine_similarity: float
    seed_id: Optional[int] = None

class CacheResults(BaseModel):
    query: str
    matches: List[CacheResult]

    def __repr__(self):
        return f"(Query: '{self.query}', Matches: {[m.prompt for m in self.matches]})"

def try_connect_to_redis(redis_url: str):
    try:
        r = redis.Redis.from_url(redis_url)
        r.ping()
        print("✅ Redis 正在运行且可访问!")
        return r
    except redis.ConnectionError:
        print("❌ 无法连接到 Redis。请确保 Redis 在运行")
        raise

class SemanticCacheWrapper:
    def __init__(self, embeddings_model: str = "BAAI/bge-large-zh-v1.5"):
        """
        初始化语义缓存引擎。
        使用单一配置源 common.env，移除多余初始化与 TTL 生命周期管理。
        """
        self.redis = try_connect_to_redis(REDIS_URL)
        
        # 移除 TTL，缓存数据基于物理清理而非时间过期
        self.embeddings_cache = EmbeddingsCache(redis_client=self.redis)
        self.langcache_embed = HFTextVectorizer(model=embeddings_model, cache=self.embeddings_cache)
        
        self.cache = SemanticCache(
            name=CACHE_NAME, 
            vectorizer=self.langcache_embed, 
            redis_client=self.redis, 
            distance_threshold=CACHE_DISTANCE_THRESHOLD
        )
        self._seed_id_by_question: Dict[str, int] = {}

    def clear(self):
        """物理清空整个向量索引和相关数据"""
        print("正在彻底清空旧语义缓存数据...")
        for key in self.redis.scan_iter(f"{self.cache.index.name}:*"):
            self.redis.delete(key)
            
        if hasattr(self, "embeddings_cache") and hasattr(self.embeddings_cache, "index"):
            for key in self.redis.scan_iter(f"{self.embeddings_cache.index.name}:*"):
                self.redis.delete(key)
        
        if self.cache.index.exists():
            self.cache.index.delete(drop=True)
        self.cache.index.create(overwrite=True, drop=False)
        
        self.cache.clear()
        self.embeddings_cache.clear()
        self._seed_id_by_question = {}

    def store_batch(self, qa_pairs: List[Dict], clear: bool = True):
        """
        批量存储预定义问答对 (解耦了 Pandas dataframe 的业务逻辑)。
        """
        if clear:
            self.clear()
            
        for item in qa_pairs:
            q, a = item["question"], item["answer"]
            seed_id = item.get("id")
            
            # 使用底层原生接口写入
            self.cache.store(prompt=q, response=a)
            self._seed_id_by_question[str(q)] = seed_id

    def check(self, query: str, distance_threshold: Optional[float] = None, num_results: int = 1) -> CacheResults:
        candidates = self.cache.check(query, distance_threshold=distance_threshold, num_results=num_results)
        
        if not candidates:
            return CacheResults(query=query, matches=[])
            
        results: List[CacheResult] = []
        for item in candidates[:num_results]:
            result = dict(item)
            result["vector_distance"] = float(result.get("vector_distance", 0.0))
            result["cosine_similarity"] = float((2 - result["vector_distance"]) / 2)
            result["query"] = query
            result["seed_id"] = self._seed_id_by_question.get(str(result.get("prompt", "")))
            results.append(CacheResult(**result))
            
        return CacheResults(query=query, matches=results)
