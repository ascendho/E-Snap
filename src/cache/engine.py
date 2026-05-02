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
        # L1 fast path 的两个 map 都是“归一化后的 query -> 原问题文本”，
        # 而不是“归一化后的 query -> 答案”。
        # 这样做的目的，是把“用户当时真实问法”保留为中间层：
        # 先用归一化 key 找到原问题，再用原问题去 _answer_by_question 查答案。
        # 这能让缓存层同时兼顾：
        # 1) 快速查找；2) 保留原始 prompt；3) 让 seed_id / answer / prompt 三者保持同一主键。
        self._normalized_question_map: Dict[str, str] = {}
        self._near_exact_question_map: Dict[str, str] = {}

    @staticmethod
    def normalize_query(query: str) -> str:
        """归一化 query，用于 exact fast path。

        这里刻意只做“保守归一化”：
        - 小写化
        - 去首尾空白
        - 折叠连续空白

        它的目标不是“尽量把不同写法揉成同一个问题”，
        而是保证 exact 命中依然代表“几乎同一问法”。
        更激进的清洗逻辑留给 near_exact 层处理。
        """
        return " ".join(str(query).strip().lower().split())

    @staticmethod
    def normalize_surface_query(query: str) -> str:
        """更激进的表面归一化，仅用于 near_exact fast path。

        这一层主要解决“字面形式不同、语义其实没变”的情况，例如：
        - 全角 / 半角差异
        - 多余空格
        - 中英文标点差异

        这里仍然是 L1 fast path，而不是 semantic matching；
        它本质上处理的是“表面噪声”，不是“语义改写”。
        """
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
        """按常见分句符与连接词拆分复合问题，供子问题候选扫描使用。

        这里的目标不是做完整自然语言句法分析，而是做一个“足够便宜”的
        复合问题切分器，为 subquery fast path 提供候选段落。

        最小长度保留在 2 个字符，是为了兼容中文里很短但合法的子问题，
        同时避免切出大量无意义碎片。
        """
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
        """在复合问题中扫描已缓存子问题，命中后返回 rerank 候选。

        注意这里返回的是“候选”而不是直接整句命中：
        某个子问题问过，并不等于整个复合问题都已经被回答。
        后续是否能直接复用、是否只补缺口，交给 workflow 层再决定。
        """
        for segment in self.split_query_segments(query):
            normalized_segment = self.normalize_query(segment)
            # 这里的 exact_match 仍然是“原问题文本”，不是答案。
            # 先用归一化后的子问题命中 L1 map，再拿原问题文本去 _answer_by_question 查答案。
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
        """计算编辑距离，但只关心“是否仍在可接受上限内”。

        返回值不是 bool，而是：
        - int: 真实编辑距离，且该距离 <= max_distance
        - None: 明确超出上限，不值得继续算

        `max_distance` 既是匹配阈值，也是性能剪枝条件：
        一旦某一整行的最小值已经大于上限，后续就不可能再回到可接受范围，
        因此可以提前结束，避免在 L1 typo 检查里浪费 CPU。
        """
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
                # 提前终止：当前整行最小值都已经超出上限，
                # 说明不可能再得到“距离 <= max_distance”的结果。
                return None
            previous_row = current_row

        distance = previous_row[-1]
        return distance if distance <= max_distance else None

    def find_edit_distance_candidate(self, query: str) -> Optional[CacheResult]:
        """用小编辑距离识别错别字、同音字和 OCR 噪声。

        它解决的是“同一个问题打错了一两个字”的场景，而不是语义改写：
        - 例如：商品名错一个字、OCR 少一个标点、输入法误按
        - 不负责判断“两个不同说法是否语义相近”，那属于 semantic cache / reranker 的职责

        返回值只会是“最佳单候选”或 None，不会返回多个近似问题。
        这样可以让 L1 fast path 继续保持确定性和低成本。
        """
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

        这是整个缓存系统的统一写入口：
        - FAQ seed 预热时会走这里
        - workflow 运行时写回缓存时也会走这里
        - 后台子问题提取后的写回同样走这里

        为什么不能只调用 `self.cache.store()`？
        因为本项目不是只有 L2 semantic cache，还维护了多套 L1 fast path map。
        只有在这里同时刷新：
        - `_answer_by_question`
        - `_seed_id_by_question`
        - `_normalized_question_map`
        - `_near_exact_question_map`
        才能保证“写入一条问答后，所有查询路径看到的是同一份事实”。
        """
        if not prompt or not answer:
            return

        prompt_str = str(prompt)
        self.cache.store(prompt=prompt_str, response=answer)
        self._seed_id_by_question[prompt_str] = seed_id
        self._answer_by_question[prompt_str] = answer
        # 注意：两个 L1 map 存进去的 value 都是“原问题文本”，不是答案本身。
        # 这也是为什么 exact_match / near_exact_match 取出来后，还要再去 _answer_by_question 查答案。
        self._normalized_question_map[self.normalize_query(prompt_str)] = prompt_str
        self._near_exact_question_map[self.normalize_surface_query(prompt_str)] = prompt_str

    def contains_prompt_variant(self, prompt: str) -> bool:
        """True if a normalized form of `prompt` is already registered in L1 maps.

        这里检查的是“这个问法的归一化变体是否已经存在”，不是语义相似度判断。
        典型用途是写回前去重：
        - 如果只是空格、标点、全半角不同，就没必要再重复存一条
        - 如果两个问题只是语义相近但字面不同，这里不会把它们视作同一条

        因此它是 L1 写回的去重屏障，不是 semantic cache 的召回逻辑。
        """
        if not prompt:
            return False
        if self.normalize_query(prompt) in self._normalized_question_map:
            return True
        if self.normalize_surface_query(prompt) in self._near_exact_question_map:
            return True
        return False

    def check(self, query: str, distance_threshold: Optional[float] = None, num_results: int = 1) -> CacheResults:
        """统一缓存查询入口。

        真实顺序是：
        1. L1 exact
        2. L1 near_exact
        3. L1 edit_distance
        4. L1 subquery
        5. L2 semantic cache

        前四层都属于“便宜、确定性更高”的快速路径；
        只有它们都失败后，才值得支付向量检索成本。
        """
        # ===== L1 exact fast path：归一化后完全一致则直接命中 =====
        if CACHE_L1_EXACT_ENABLED:
            normalized_query = self.normalize_query(query)
            exact_match = self._normalized_question_map.get(normalized_query)
            # exact_match 取到的是“原问题文本”，不是答案。
            # 下一步再用这个原问题文本去 _answer_by_question 里拿答案，
            # 这样可以保持 prompt / answer / seed_id 三者都以同一个原问题为主键。
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

            # near_exact 仍然属于“同一问法的表面噪声”，所以继续留在 L1 fast path。
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
                # edit_distance 放在 near_exact 之后：
                # 只有前两层都失败时，才把错别字 / OCR 这类微小扰动当作候选。
                edit_distance_candidate = self.find_edit_distance_candidate(query)
                if edit_distance_candidate:
                    return CacheResults(query=query, matches=[edit_distance_candidate])

            # subquery 检查放在 semantic 前面，是为了优先利用“确定的局部复用信息”，
            # 避免复合问题一上来就掉入更昂贵、更不透明的向量召回。
            subquery_candidate = self.find_subquery_candidate(query)
            if subquery_candidate:
                return CacheResults(query=query, matches=[subquery_candidate])
        
        # 到达这里说明所有 L1 fast path 都没拦住，才进入 L2 semantic cache。
        candidates = self.cache.check(query, distance_threshold=distance_threshold, num_results=num_results)
        
        if not candidates:
            return CacheResults(query=query, matches=[])
            
        results: List[CacheResult] = []
        for item in candidates[:num_results]:
            result = dict(item)
            # RedisVL 返回的是 vector_distance；workflow 层更常用 cosine_similarity 来记录与展示。
            result["vector_distance"] = float(result.get("vector_distance", 0.0))
            result["cosine_similarity"] = float((2 - result["vector_distance"]) / 2)
            result["query"] = query
            result["seed_id"] = self._seed_id_by_question.get(str(result.get("prompt", "")))
            result["match_type"] = "semantic"
            results.append(CacheResult(**result))
            
        return CacheResults(query=query, matches=results)
