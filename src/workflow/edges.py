import logging  # 导入日志模块，用于追踪智能体的决策路径
from typing import Literal  # 用于类型提示，限制函数返回值的具体范围

# 获取名为 "agentic-workflow" 的日志记录器
logger = logging.getLogger("agentic-workflow")

def cache_router(state) -> Literal["rerank_cache", "research"]:
    """
    语义缓存路由决策器（前置层）。

    逻辑：
    1. 如果 check_cache 找到了候选（cache_hit=True），先把它送到 rerank_cache 做 LLM 复用裁判，不再直接放行。
    2. 没找到候选则进入 research（RAG）。
    """
    query = state["query"]
    cache_hit = state.get("cache_hit", False)

    if cache_hit:
        logger.info(f"👉 路由: 缓存有候选，进入 Reranker -> '{query[:20]}...'")
        return "rerank_cache"
    else:
        logger.info(f"👉 路由: 未命中缓存，开始研究 -> '{query[:20]}...'")
        return "research"

def cache_rerank_router(state) -> Literal["synthesize_response", "research"]:
    """
    缓存复用裁判后置路由。

    逻辑：
    1. Reranker 判定可复用 -> 直接合成回答。
    2. Reranker 判定不可复用或调用失败 -> 走 RAG 重新检索。
    """
    query = state["query"]
    passed = state.get("cache_rerank_passed", False)
    score = state.get("cache_rerank_score", 0.0)

    if passed:
        logger.info(f"👉 路由: Reranker 通过 ({score:.2f})，直接合成 -> '{query[:20]}...'")
        return "synthesize_response"
    else:
        logger.info(f"👉 路由: Reranker 拒绝 ({score:.2f})，进入研究 -> '{query[:20]}...'")
        return "research"