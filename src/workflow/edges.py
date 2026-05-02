"""LangGraph 工作流路由层。

Phase 3 重构目标：把以前散落在 ``graph.py`` lambda、``cache_router``、
``cache_rerank_router`` 中的字符串路由目标集中成 ``RouteTarget`` 常量，
并把 ``pre_check`` 之后的“拦截/继续”决策抽出具名 ``pre_check_router``，
与其他 router 形态一致，避免分散决策点。

所有 router 仍然只读 state、不写 state，决策依据保留在节点内部。
"""

from __future__ import annotations

import logging
from typing import Final, Literal

logger = logging.getLogger("agentic-workflow")


class RouteTarget:
    """LangGraph 节点名常量，与 ``graph.py`` 中 ``add_node`` 注册的名字保持一致。"""

    PRE_CHECK: Final[str] = "pre_check"
    CHECK_CACHE: Final[str] = "check_cache"
    RERANK_CACHE: Final[str] = "rerank_cache"
    RESEARCH: Final[str] = "research"
    RESEARCH_SUPPLEMENT: Final[str] = "research_supplement"
    SYNTHESIZE_RESPONSE: Final[str] = "synthesize_response"


PreCheckTarget = Literal["check_cache", "synthesize_response"]
CacheRouterTarget = Literal["synthesize_response", "rerank_cache", "research_supplement", "research"]
RerankRouterTarget = Literal["synthesize_response", "research_supplement", "research"]


def pre_check_router(state) -> PreCheckTarget:
    """前置拦截后置路由：拦截则直接合成兜底回答，否则继续查缓存。"""
    # 这里故意只看 `intercepted` 一个布尔位，保证 pre_check 的职责非常单纯：
    # 节点负责“识别是否该拦截”，router 负责“决定拦截后走向哪里”。
    if state.get("intercepted", False):
        return RouteTarget.SYNTHESIZE_RESPONSE  # type: ignore[return-value]
    return RouteTarget.CHECK_CACHE  # type: ignore[return-value]


def cache_router(state) -> CacheRouterTarget:
    """语义缓存路由决策器（前置层）。

    逻辑：
    1. 如果 check_cache 找到了 exact、near_exact 或 edit_distance 候选，直接进入 synthesize_response，跳过 Reranker。
    2. 如果规则子问题命中已可确定缺口，则直接进入 research_supplement。
    3. 如果找到了 semantic 或其他 subquery 候选，则进入 rerank_cache 做 LLM 复用裁判。
    4. 没找到候选则进入 research（RAG）。
    """
    query = state["query"]
    cache_hit = state.get("cache_hit", False)
    match_type = state.get("cache_match_type", "none")
    reuse_mode = state.get("cache_reuse_mode", "none")

    # `check_cache_node()` 已经提前处理掉一种非常特殊的情况：
    # 如果 subquery 命中后能确定“只缺一个明确子问题”，它会直接把 state 标成
    # `partial_reuse`，此时没必要再让 reranker 看一次“子问题答案 vs 复合问题”。
    if reuse_mode == "partial_reuse":
        logger.info(f"👉 路由: 规则子问题命中，直接进入补充研究 -> '{query[:20]}...'")
        return RouteTarget.RESEARCH_SUPPLEMENT  # type: ignore[return-value]

    if cache_hit:
        # 这三类 L1 fast path 被视为“足够安全的直接命中”：
        # - exact: 归一化后完全相同
        # - near_exact: 只差格式/标点/全半角
        # - edit_distance: 很小的表面噪声（如 OCR / 错别字）
        # 因为它们本质上仍然是“同一个问题”，不值得再花一次分析模型调用。
        if match_type in {"exact", "near_exact", "edit_distance"}:
            logger.info(f"👉 路由: {match_type} 命中，跳过 Reranker 直接合成 -> '{query[:20]}...'")
            return RouteTarget.SYNTHESIZE_RESPONSE  # type: ignore[return-value]
        # 走到这里的候选通常是 semantic / subquery_*。
        # 它们只是“可能相关”，还没有强到可以直接复用答案，必须交给 reranker 再判断。
        logger.info(f"👉 路由: 缓存有候选[{match_type}]，进入 Reranker -> '{query[:20]}...'")
        return RouteTarget.RERANK_CACHE  # type: ignore[return-value]

    # 完全未命中时直接进入 research；router 不尝试做任何补救逻辑，
    # 补救策略应始终留在节点内部，避免路由层膨胀成第二套业务逻辑。
    logger.info(f"👉 路由: 未命中缓存，开始研究 -> '{query[:20]}...'")
    return RouteTarget.RESEARCH  # type: ignore[return-value]


def cache_rerank_router(state) -> RerankRouterTarget:
    """缓存复用裁判后置路由。

    逻辑：
    1. full_reuse -> 直接合成回答。
    2. partial_reuse -> 进入补充研究，只查缺失部分。
    3. reject 或调用失败 -> 走 RAG 重新检索。
    """
    query = state["query"]
    reuse_mode = state.get("cache_reuse_mode", "none")
    score = state.get("cache_rerank_score", 0.0)

    # 到达这里时，rerank_cache_node() 已经把复杂判断折叠成少数几种稳定状态。
    # router 不再关心模型返回细节，只消费最终的 `cache_reuse_mode`。
    if reuse_mode == "full_reuse":
        logger.info(f"👉 路由: Reranker 通过 ({score:.2f})，直接合成 -> '{query[:20]}...'")
        return RouteTarget.SYNTHESIZE_RESPONSE  # type: ignore[return-value]
    if reuse_mode == "partial_reuse":
        # partial_reuse 的核心思想是“缓存答案已经覆盖了一部分，
        # 剩下的问题可以缩成一个更小的 residual_query 去研究”。
        logger.info(f"👉 路由: Reranker 判定部分复用 ({score:.2f})，进入补充研究 -> '{query[:20]}...'")
        return RouteTarget.RESEARCH_SUPPLEMENT  # type: ignore[return-value]

    logger.info(f"👉 路由: Reranker 拒绝 ({score:.2f})，进入研究 -> '{query[:20]}...'")
    return RouteTarget.RESEARCH  # type: ignore[return-value]


__all__ = [
    "RouteTarget",
    "PreCheckTarget",
    "CacheRouterTarget",
    "RerankRouterTarget",
    "pre_check_router",
    "cache_router",
    "cache_rerank_router",
]
