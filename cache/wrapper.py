from typing import Dict, List, Optional
import pandas as pd
import redis
from pydantic import BaseModel
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer

# ==========================================
# 数据模型定义 (基于 Pydantic)
# 用于规范化缓存命中后返回的数据结构
# ==========================================

class CacheResult(BaseModel):
    """单条缓存命中结果的数据模型"""
    prompt: str                   # 缓存库中原本存储的“问题” (例如预装的 FAQ 问题)
    response: str                 # 对应的“答案”
    vector_distance: float        # Redis 底层计算出的向量距离 (距离越小，语义越相似)
    cosine_similarity: float      # 转化为人类更易读的余弦相似度 (0.0~1.0，越接近 1 说明越相似)
    seed_id: Optional[int] = None # 这条记录在原始表格中的溯源 ID (如果是预热数据的话)

class CacheResults(BaseModel):
    """一次检索请求返回的整体结果对象"""
    query: str                  # 用户当前真正提问的句子
    matches: List[CacheResult]  # 匹配到的缓存条目列表 (如果没有命中，这里就是空列表)

    def __repr__(self):
        # 重写打印格式，方便在终端调试时直观查看命中了哪些问题
        return f"(Query: '{self.query}', Matches: {[m.prompt for m in self.matches]})"

# ==========================================
# 工具函数
# ==========================================

def try_connect_to_redis(redis_url: str):
    """
    Redis 连通性健康检查。
    在启动智能体前，确保向量数据库是存活的，避免运行时崩溃。
    """
    try:
        r = redis.Redis.from_url(redis_url)
        r.ping() # 发送 ping 信号测试连通性
        print("✅ Redis 正在运行且可访问!")
        return r
    except redis.ConnectionError:
        print("❌ 无法连接到 Redis。请确保 Redis 在运行")
        raise

# ==========================================
# 核心类：语义缓存包装器
# 封装了底层 redisvl 的复杂操作，对外提供极简接口
# ==========================================

class SemanticCacheWrapper:
    def __init__(self, name: str = "semantic-cache", distance_threshold: float = 0.3, ttl: int = 3600, redis_url: Optional[str] = None):
        """
        初始化语义缓存引擎。
        
        💡 架构亮点：这里实现了“双重缓存(Dual Cache)”
        1. EmbeddingsCache: 缓存纯粹的“文本到向量”的转换结果，防止大模型重复计算同一个词的向量。
        2. SemanticCache: 缓存真正的“问题-答案”对。
        """
        redis_conn_url = redis_url or "redis://localhost:6379"
        self.redis = try_connect_to_redis(redis_conn_url)
        
        # 初始化第 1 层缓存：向量特征缓存 (存活期故意设得比语义缓存长，这里设为 ttl 的 24 倍)
        self.embeddings_cache = EmbeddingsCache(redis_client=self.redis, ttl=ttl * 24)
        
        # 初始化“翻译官”模型，并挂载刚刚创建的向量特征缓存
        self.langcache_embed = HFTextVectorizer(model="BAAI/bge-large-zh-v1.5", cache=self.embeddings_cache)
        
        # 初始化第 2 层缓存：核心的智能体语义缓存
        self.cache = SemanticCache(
            name=name, 
            vectorizer=self.langcache_embed, 
            redis_client=self.redis, 
            distance_threshold=distance_threshold, # 决定缓存命中宽容度的核心参数
            ttl=ttl                                # LLM 生成的动态答案默认存活时间
        )
        
        # 内部字典：用于追踪内存中的问题与原始表格 ID 的映射关系
        self._seed_id_by_question: Dict[str, int] = {}

    @classmethod
    def from_config(cls, config) -> "SemanticCacheWrapper":
        """
        工厂方法：允许直接通过配置字典 (dict) 或环境变量来实例化缓存对象。
        """
        return cls(
            redis_url=config.get("redis_url", "redis://localhost:6379"),
            name=config.get("cache_name", "semantic-cache"),
            distance_threshold=float(config.get("distance_threshold", 0.3)),
            ttl=int(config.get("ttl_seconds", 3600)),
        )

    def hydrate_from_df(self, df: pd.DataFrame, q_col: str = "question", a_col: str = "answer", clear: bool = True, ttl_override: Optional[int] = None, return_id_map: bool = False) -> Optional[Dict[str, int]]:
        """
        语义缓存预热核心函数：从 DataFrame 批量加载基础 FAQ 数据。
        常用于系统冷启动时，将高频标准问题注入 Redis，实现第一道防线拦截。
        """
        # 1. 重置缓存池，防止旧版本脏数据污染
        if clear:
            # 物理清空整个向量索引和相关数据（只针对语义缓存，不影响知识库库）
            if self.cache.index.exists():
                self.cache.index.delete(drop=True)
            self.cache.index.create(overwrite=True, drop=False)
            self.embeddings_cache.clear()
            
        self._seed_id_by_question = {}
        question_to_id: Dict[str, int] = {}
        
        # 判断表格里有没有自带的 'id' 字段，没有就用循环的索引替代
        id_col = "id" if "id" in df.columns else None
        
        # 2. 遍历表格，逐条转为向量并存入 Redis
        for idx, row in enumerate(df.to_dict(orient="records")):
            q, a = row[q_col], row[a_col]
            
            # 核心写入操作。ttl_override 若为 None，通常意味着让基础 FAQ 永久有效
            self.cache.store(prompt=q, response=a, ttl=ttl_override)
            
            # 3. 建立并记录映射关系，便于后续数据分析和对账
            seed_id = int(row[id_col]) if id_col and row.get(id_col) is not None else idx
            self._seed_id_by_question[str(q)] = seed_id
            if return_id_map and q not in question_to_id:
                question_to_id[q] = seed_id
                
        return question_to_id if return_id_map else None

    def check(self, query: str, distance_threshold: Optional[float] = None, num_results: int = 1) -> CacheResults:
        """
        检查缓存：判断用户的当前提问，是否命中了我们之前存入的 FAQ 或大模型历史回答。
        
        Args:
            query (str): 用户实时的提问语句。
            distance_threshold (float, optional): 临时覆盖全局阈值。
            num_results (int): 希望返回几个最相近的结果，默认返回 1 个。
        """
        # 1. 呼叫底层的 redisvl 框架进行向量相似度检索
        candidates = self.cache.check(query, distance_threshold=distance_threshold, num_results=num_results)
        
        # 如果底层没查到任何符合条件的结果（Cache Miss）
        if not candidates:
            return CacheResults(query=query, matches=[])
            
        # 2. 数据清洗与组装 (Cache Hit)
        results: List[CacheResult] = []
        for item in candidates[:num_results]:
            result = dict(item)
            
            # 提取向量距离
            result["vector_distance"] = float(result.get("vector_distance", 0.0))
            
            # 将生涩的向量距离(Distance)通过数学公式转化为易懂的余弦相似度(Cosine Similarity)
            # 在 L2 归一化向量下，关系通常为: distance = 2 * (1 - cosine_sim) -> cosine_sim = (2 - distance) / 2
            result["cosine_similarity"] = float((2 - result["vector_distance"]) / 2)
            
            result["query"] = query
            # 尝试去内存字典中找回这个回答当年是由哪个 FAQ ID 生成的
            result["seed_id"] = self._seed_id_by_question.get(str(result.get("prompt", "")))
            
            # 将处理好的字典转化为结构化的 Pydantic 对象
            results.append(CacheResult(**result))
            
        # 返回包装好的整体结果
        return CacheResults(query=query, matches=results)