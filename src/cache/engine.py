from typing import Dict, List, Optional
import redis
import unicodedata
from pydantic import BaseModel
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer
from common.env import (
    REDIS_URL,
    CACHE_NAME,
    CACHE_DISTANCE_THRESHOLD,
    CACHE_L1_EXACT_ENABLED,
    CACHE_L1_EDIT_DISTANCE_ENABLED,
    CACHE_L1_EDIT_DISTANCE_MAX_DISTANCE,
)

class CacheResult(BaseModel):
    prompt: str
    response: str
    vector_distance: float
    cosine_similarity: float
    seed_id: Optional[int] = None
    match_type: str = "semantic"

class CacheResults(BaseModel):
    query: str
    matches: List[CacheResult]

    def __repr__(self):
        return f"(Query: '{self.query}', Matches: {[m.prompt for m in self.matches]})"

def try_connect_to_redis(redis_url: str):
    try:
        r = redis.Redis.from_url(redis_url)
        # --- 生产级安全对齐：容量与淘汰策略 ---
        # 1. 限制 Redis 最大内存为100MB，防止恶意攻击或大规模缓存导致服务器 OOM (Out Of Memory)
        r.config_set("maxmemory", "100mb")
        # 2. 设置淘汰策略：当容量达到上限时，淘汰全库最近最少使用的 Key (allkeys-lru)
        r.config_set("maxmemory-policy", "allkeys-lru")
        
        r.ping()
        print("✅ Redis 正在运行且可访问! (已配额: 100MB LRU 容量)")
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
        self._answer_by_question: Dict[str, str] = {}
        self._normalized_question_map: Dict[str, str] = {}
        self._near_exact_question_map: Dict[str, str] = {}

    @staticmethod
    def normalize_query(query: str) -> str:
        """归一化 query，用于 exact fast path。"""
        return " ".join(str(query).strip().lower().split())

    @staticmethod
    def normalize_surface_query(query: str) -> str:
        """更激进的表面归一化，仅用于 near_exact fast path。"""
        normalized = unicodedata.normalize("NFKC", str(query)).lower().strip()
        collapsed = "".join(normalized.split())
        allowed_chars = []
        for char in collapsed:
            if unicodedata.category(char).startswith("P"):
                continue
            allowed_chars.append(char)
        return "".join(allowed_chars)

    @staticmethod
    def split_query_segments(query: str) -> List[str]:
        """按常见分句符与连接词拆分复合问题，供子问题候选扫描使用。"""
        normalized = unicodedata.normalize("NFKC", str(query))
        for separator in ["？", "?", "！", "!", "。", "；", ";", "，", ",", "另外", "还有", "以及", "并且"]:
            normalized = normalized.replace(separator, "\n")
        segments = []
        for segment in normalized.splitlines():
            cleaned = segment.strip()
            if len(cleaned) >= 2:
                segments.append(cleaned)
        return segments

    def find_subquery_candidate(self, query: str) -> Optional[CacheResult]:
        """在复合问题中扫描已缓存子问题，命中后返回 rerank 候选。"""
        for segment in self.split_query_segments(query):
            normalized_segment = self.normalize_query(segment)
            exact_match = self._normalized_question_map.get(normalized_segment)
            if exact_match:
                print(f"⚡ [子问题命中] exact subquery hit: '{segment}' -> '{exact_match}'")
                return CacheResult(
                    prompt=exact_match,
                    response=self._answer_by_question[exact_match],
                    vector_distance=0.0,
                    cosine_similarity=1.0,
                    seed_id=self._seed_id_by_question.get(exact_match),
                    match_type="subquery_exact",
                )

            near_exact_segment = self.normalize_surface_query(segment)
            near_exact_match = self._near_exact_question_map.get(near_exact_segment)
            if near_exact_match:
                print(f"⚡ [子问题命中] near-exact subquery hit: '{segment}' -> '{near_exact_match}'")
                return CacheResult(
                    prompt=near_exact_match,
                    response=self._answer_by_question[near_exact_match],
                    vector_distance=0.0,
                    cosine_similarity=1.0,
                    seed_id=self._seed_id_by_question.get(near_exact_match),
                    match_type="subquery_near_exact",
                )

        return None

    @staticmethod
    def _levenshtein_distance_with_limit(source: str, target: str, max_distance: int) -> Optional[int]:
        if source == target:
            return 0
        if abs(len(source) - len(target)) > max_distance:
            return None
        if not source:
            return len(target) if len(target) <= max_distance else None
        if not target:
            return len(source) if len(source) <= max_distance else None

        previous_row = list(range(len(target) + 1))
        for row_index, source_char in enumerate(source, start=1):
            current_row = [row_index]
            row_min = row_index
            for col_index, target_char in enumerate(target, start=1):
                insertions = previous_row[col_index] + 1
                deletions = current_row[col_index - 1] + 1
                substitutions = previous_row[col_index - 1] + (source_char != target_char)
                current_value = min(insertions, deletions, substitutions)
                current_row.append(current_value)
                row_min = min(row_min, current_value)
            if row_min > max_distance:
                return None
            previous_row = current_row

        distance = previous_row[-1]
        return distance if distance <= max_distance else None

    def find_edit_distance_candidate(self, query: str) -> Optional[CacheResult]:
        """用小编辑距离识别错别字、同音字和 OCR 噪声。"""
        normalized_query = self.normalize_surface_query(query)
        best_match = None
        best_distance = None

        for normalized_candidate, original_question in self._near_exact_question_map.items():
            distance = self._levenshtein_distance_with_limit(
                normalized_query,
                normalized_candidate,
                CACHE_L1_EDIT_DISTANCE_MAX_DISTANCE,
            )
            if distance is None:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_match = original_question

        if best_match is None or best_distance is None:
            return None

        print(f"⚡ [编辑距离命中] edit-distance hit: '{query}' -> '{best_match}' (distance={best_distance})")
        return CacheResult(
            prompt=best_match,
            response=self._answer_by_question[best_match],
            vector_distance=0.0,
            cosine_similarity=max(0.0, 1.0 - best_distance / max(len(normalized_query), 1)),
            seed_id=self._seed_id_by_question.get(best_match),
            match_type="edit_distance",
        )

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
        self._answer_by_question = {}
        self._normalized_question_map = {}
        self._near_exact_question_map = {}

    def store_batch(self, qa_pairs: List[Dict], clear: bool = True):
        """
        批量存储预定义问答对 (解耦了 Pandas dataframe 的业务逻辑)。
        """
        if clear:
            self.clear()

        for item in qa_pairs:
            self.register_entry(item["question"], item["answer"], seed_id=item.get("id"))

    def register_entry(self, prompt: str, answer: str, seed_id: Optional[int] = None) -> None:
        """Write `(prompt, answer)` into the vector store and refresh the L1 lookup maps.

        Single source of truth for cache writes used by `store_batch` (FAQ seed) and by
        the workflow's runtime writeback path (`_store_cache_entry` in nodes.py).
        Keeping the four `_*_by_question` maps consistent here means call sites no longer
        need to know about cache internals.
        """
        if not prompt or not answer:
            return

        prompt_str = str(prompt)
        self.cache.store(prompt=prompt_str, response=answer)
        self._seed_id_by_question[prompt_str] = seed_id
        self._answer_by_question[prompt_str] = answer
        self._normalized_question_map[self.normalize_query(prompt_str)] = prompt_str
        self._near_exact_question_map[self.normalize_surface_query(prompt_str)] = prompt_str

    def contains_prompt_variant(self, prompt: str) -> bool:
        """True if a normalized form of `prompt` is already registered in L1 maps.

        Used by background subquery writeback to skip prompts that would only re-store an
        existing exact/near_exact entry.
        """
        if not prompt:
            return False
        if self.normalize_query(prompt) in self._normalized_question_map:
            return True
        if self.normalize_surface_query(prompt) in self._near_exact_question_map:
            return True
        return False

    def check(self, query: str, distance_threshold: Optional[float] = None, num_results: int = 1) -> CacheResults:
        # ===== L1 exact fast path：归一化后完全一致则直接命中 =====
        if CACHE_L1_EXACT_ENABLED:
            normalized_query = self.normalize_query(query)
            exact_match = self._normalized_question_map.get(normalized_query)
            if exact_match:
                print(f"⚡ [精确命中] normalized exact hit: '{query}' -> '{exact_match}'")
                return CacheResults(query=query, matches=[
                    CacheResult(
                        prompt=exact_match,
                        response=self._answer_by_question[exact_match],
                        vector_distance=0.0,
                        cosine_similarity=1.0,
                        seed_id=self._seed_id_by_question.get(exact_match),
                        match_type="exact",
                    )
                ])

            near_exact_query = self.normalize_surface_query(query)
            near_exact_match = self._near_exact_question_map.get(near_exact_query)
            if near_exact_match:
                print(f"⚡ [近精确命中] normalized surface hit: '{query}' -> '{near_exact_match}'")
                return CacheResults(query=query, matches=[
                    CacheResult(
                        prompt=near_exact_match,
                        response=self._answer_by_question[near_exact_match],
                        vector_distance=0.0,
                        cosine_similarity=1.0,
                        seed_id=self._seed_id_by_question.get(near_exact_match),
                        match_type="near_exact",
                    )
                ])

            if CACHE_L1_EDIT_DISTANCE_ENABLED:
                edit_distance_candidate = self.find_edit_distance_candidate(query)
                if edit_distance_candidate:
                    return CacheResults(query=query, matches=[edit_distance_candidate])

            subquery_candidate = self.find_subquery_candidate(query)
            if subquery_candidate:
                return CacheResults(query=query, matches=[subquery_candidate])
        
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
            result["match_type"] = "semantic"
            results.append(CacheResult(**result))
            
        return CacheResults(query=query, matches=results)
