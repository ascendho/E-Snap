import logging  # 导入日志模块，用于追踪智能体的决策路径
from typing import Literal  # 用于类型提示，限制函数返回值的具体范围

# 获取名为 "agentic-workflow" 的日志记录器
logger = logging.getLogger("agentic-workflow")

def cache_router(state) -> Literal["synthesize_response", "rerank_cache", "research"]:
    """
    语义缓存路由决策器（前置层）。

    逻辑：
    1. 如果 check_cache 找到了 exact 或 near_exact 候选，直接进入 synthesize_response，跳过 Reranker。
    2. 如果找到了 semantic 或 subquery 候选，则进入 rerank_cache 做 LLM 复用裁判。
    3. 没找到候选则进入 research（RAG）。
    """
    query = state["query"]
    cache_hit = state.get("cache_hit", False)
    match_type = state.get("cache_match_type", "none")

    if cache_hit:
        if match_type in {"exact", "near_exact"}:
            logger.info(f"👉 路由: {match_type} 命中，跳过 Reranker 直接合成 -> '{query[:20]}...'")
            return "synthesize_response"
        logger.info(f"👉 路由: 缓存有候选[{match_type}]，进入 Reranker -> '{query[:20]}...'")
        return "rerank_cache"
    else:
        logger.info(f"👉 路由: 未命中缓存，开始研究 -> '{query[:20]}...'")
        return "research"

def cache_rerank_router(state) -> Literal["synthesize_response", "research_supplement", "research"]:
    """
    缓存复用裁判后置路由。

    逻辑：
    1. full_reuse -> 直接合成回答。
    2. partial_reuse -> 进入补充研究，只查缺失部分。
    3. reject 或调用失败 -> 走 RAG 重新检索。
    """
    query = state["query"]
    reuse_mode = state.get("cache_reuse_mode", "none")
    score = state.get("cache_rerank_score", 0.0)

    if reuse_mode == "full_reuse":
        logger.info(f"👉 路由: Reranker 通过 ({score:.2f})，直接合成 -> '{query[:20]}...'")
        return "synthesize_response"
    if reuse_mode == "partial_reuse":
        logger.info(f"👉 路由: Reranker 判定部分复用 ({score:.2f})，进入补充研究 -> '{query[:20]}...'")
        return "research_supplement"

    logger.info(f"👉 路由: Reranker 拒绝 ({score:.2f})，进入研究 -> '{query[:20]}...'")
    return "research"