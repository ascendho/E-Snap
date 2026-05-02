"""工作流状态边界与状态相关辅助函数。

第二阶段重构目标：把以前散落在 ``nodes.py`` 顶部的状态类型定义、初始化、
LLM 用量累计、后台任务等待这些“横切”逻辑集中到独立模块，让 ``nodes.py``
回归节点实现职责。

为了不破坏既有调用者（``graph.py`` / ``api/server.py`` / ``tests/runner.py``），
``nodes.py`` 仍会再导出本模块的全部公共名字。本次保留单一扁平的
``WorkflowState`` 不做一次性迁移，仅提供 ``CacheStateView`` /
``RoutingStateView`` 等 ``total=False`` 子视图，作为读侧文档与未来分组依据。
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, TypedDict

from common.env import (
    ANALYSIS_CACHED_INPUT_PRICE_RMB_PER_1K,
    ANALYSIS_INPUT_PRICE_RMB_PER_1K,
    ANALYSIS_OUTPUT_PRICE_RMB_PER_1K,
    RESEARCH_CACHED_INPUT_PRICE_RMB_PER_1K,
    RESEARCH_INPUT_PRICE_RMB_PER_1K,
    RESEARCH_OUTPUT_PRICE_RMB_PER_1K,
)

logger = logging.getLogger("agentic-workflow")


# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------

CacheMatchType = Literal[
    "exact",
    "near_exact",
    "edit_distance",
    "subquery_exact",
    "subquery_near_exact",
    "semantic",
    "none",
]
CacheReuseMode = Literal["full_reuse", "partial_reuse", "dual_subquery", "reject", "none"]
RerankAttempt = Literal["none", "skipped", "primary", "fallback", "failed"]
ModelFamily = Literal["analysis", "research"]


# ---------------------------------------------------------------------------
# 类型字典结构定义
# ---------------------------------------------------------------------------

class LLMUsage(TypedDict):
    analysis_calls: int
    analysis_input_tokens: int
    analysis_output_tokens: int
    analysis_cached_input_tokens: int
    research_calls: int
    research_input_tokens: int
    research_output_tokens: int
    research_cached_input_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    total_cached_input_tokens: int
    analysis_cost_rmb: float
    research_cost_rmb: float
    total_cost_rmb: float


class WorkflowMetrics(TypedDict):
    """工作流性能指标的字典结构。"""
    total_latency: float            # 总耗时
    precheck_latency: float         # 前置拦截检查耗时
    cache_latency: float            # 缓存检查耗时
    rerank_latency: float           # 缓存复用裁判耗时
    research_latency: float         # 知识检索耗时
    supplement_latency: float       # 补充研究耗时
    synthesis_latency: float        # 回答合成耗时
    total_research_iterations: int  # 总研究循环次数


class WorkflowState(TypedDict):
    """整个计算图共享的扁平状态对象。

    内部按职能分组：``cache_*`` 字段属于缓存子状态、``metrics`` 与
    ``llm_usage`` 是单独子结构、``execution_path`` / ``intercepted`` /
    ``research_iterations`` / ``current_research_strategy`` 描述路由进度。
    具体分组语义可参考 ``CacheStateView`` 与 ``RoutingStateView``。
    """
    # --- 输入 / 输出 ---
    # `query` 由 API / runner 在入口处写入。
    # `answer` 是“当前阶段产出的主答案草稿”，会在多个节点间被反复覆盖。
    # `final_response` 只在最终收口阶段稳定下来，供 API 层直接返回给用户。
    query: str
    answer: str
    final_response: Optional[str]

    # --- 缓存命中原始信息（主要由 check_cache_node 写入） ---
    # 这组字段描述“缓存层到底看到了什么”：是否命中、命中的问题是谁、相似度多少、命中类型是什么。
    cache_hit: bool
    cache_matched_question: Optional[str]
    cache_confidence: float
    cache_seed_id: Optional[int]
    cache_match_type: CacheMatchType
    cache_base_answer: str
    cache_enabled: bool

    # --- 路由 / 执行进度（由 pre_check / research / synthesize 等节点推进） ---
    intercepted: bool
    research_iterations: int

    # --- 缓存复用裁判结果（由 check_cache_node / rerank_cache_node 写入） ---
    # `cache_reuse_mode` 是后续 router 最核心的决策输入。
    # 相关 reason / residual 字段用于解释为什么复用、为什么拒绝、以及还缺什么问题。
    cache_rerank_passed: bool
    cache_reuse_mode: CacheReuseMode
    cache_rerank_attempt: RerankAttempt
    cache_rerank_score: float
    cache_rerank_reason: str
    cache_reranker_reason: str
    cache_validation_reason: str
    cache_reranker_residual_query: str
    cache_residual_query: str

    # --- 缓存写回计划（主要由 supplement / synthesize 阶段生产） ---
    # `cache_writeback_entries` 是待写回的问答对清单；
    # `cache_written_prompts` 是已经确认写回或计划写回的 prompt 列表：
    # 一方面会进入最终 state / API 响应供前端展示，另一方面会被 synthesize 阶段
    # 当作“本轮已经处理过哪些 prompt”的外部可观察结果。
    cache_writeback_entries: List[Dict[str, str]]
    cache_written_prompts: List[str]

    # --- 当前运行轨迹 ---
    # `execution_path` 是整个工作流的面包屑轨迹，测试与报表都会依赖它进行路径分类。
    current_research_strategy: str
    execution_path: List[str]

    # --- 可观测性与统计 ---
    # `metrics` 记录节点级耗时；`llm_calls` 记录按模型族划分的调用次数；
    # `llm_usage` 记录真实 token / 成本；`background_threads` 用于延迟的子任务写回。
    metrics: WorkflowMetrics
    timestamp: str
    llm_calls: Dict[str, int]
    llm_usage: LLMUsage
    llm_usage_lock: Any
    background_threads: List[Any]


# ---------------------------------------------------------------------------
# 子视图（仅用于文档说明，WorkflowState 仍保持扁平结构）
# ---------------------------------------------------------------------------

class CacheStateView(TypedDict, total=False):
    """缓存子状态的逻辑分组（仅用于读取/调试，不参与运行时构造）。"""
    cache_hit: bool
    cache_enabled: bool
    cache_match_type: CacheMatchType
    cache_reuse_mode: CacheReuseMode
    cache_matched_question: Optional[str]
    cache_confidence: float
    cache_seed_id: Optional[int]
    cache_base_answer: str
    cache_rerank_passed: bool
    cache_rerank_attempt: RerankAttempt
    cache_rerank_score: float
    cache_rerank_reason: str
    cache_reranker_reason: str
    cache_validation_reason: str
    cache_reranker_residual_query: str
    cache_residual_query: str
    cache_writeback_entries: List[Dict[str, str]]
    cache_written_prompts: List[str]


class RoutingStateView(TypedDict, total=False):
    """路由 / 执行进度子状态的逻辑分组。"""
    intercepted: bool
    research_iterations: int
    current_research_strategy: str
    execution_path: List[str]


# ---------------------------------------------------------------------------
# 初始化函数
# ---------------------------------------------------------------------------

def initialize_metrics() -> WorkflowMetrics:
    """初始化指标字典的默认值。"""
    return {
        "total_latency": 0.0,
        "precheck_latency": 0.0,
        "cache_latency": 0.0,
        "rerank_latency": 0.0,
        "research_latency": 0.0,
        "supplement_latency": 0.0,
        "synthesis_latency": 0.0,
        "total_research_iterations": 0,
    }


def initialize_llm_usage() -> LLMUsage:
    """初始化真实 token 与成本统计。"""
    return {
        "analysis_calls": 0,
        "analysis_input_tokens": 0,
        "analysis_output_tokens": 0,
        "analysis_cached_input_tokens": 0,
        "research_calls": 0,
        "research_input_tokens": 0,
        "research_output_tokens": 0,
        "research_cached_input_tokens": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cached_input_tokens": 0,
        "analysis_cost_rmb": 0.0,
        "research_cost_rmb": 0.0,
        "total_cost_rmb": 0.0,
    }


def build_initial_state(query: str) -> WorkflowState:
    """构建统一的工作流初始状态，避免 API 与测试入口漂移。"""
    return {
        "query": query,
        "answer": "",
        "final_response": "",
        "cache_hit": False,
        "cache_matched_question": None,
        "cache_confidence": 0.0,
        "cache_seed_id": None,
        "cache_match_type": "none",
        "cache_base_answer": "",
        "cache_enabled": True,
        "intercepted": False,
        "research_iterations": 0,
        "cache_rerank_passed": False,
        "cache_reuse_mode": "none",
        "cache_rerank_attempt": "none",
        "cache_rerank_score": 0.0,
        "cache_rerank_reason": "",
        "cache_reranker_reason": "",
        "cache_validation_reason": "",
        "cache_reranker_residual_query": "",
        "cache_residual_query": "",
        "cache_writeback_entries": [],
        "cache_written_prompts": [],
        "current_research_strategy": "",
        "execution_path": ["start"],
        "metrics": initialize_metrics(),
        "timestamp": datetime.now().isoformat(),
        "llm_calls": {},
        "llm_usage": initialize_llm_usage(),
        "llm_usage_lock": threading.Lock(),
        "background_threads": [],
    }


def update_metrics(metrics: WorkflowMetrics, **kwargs) -> WorkflowMetrics:
    """通用指标更新函数：数值字段累加、其他字段覆盖。"""
    new_metrics = metrics.copy()
    for key, value in kwargs.items():
        if key in new_metrics and isinstance(new_metrics[key], (int, float)):
            new_metrics[key] += value
        else:
            new_metrics[key] = value
    return new_metrics


# ---------------------------------------------------------------------------
# 大模型调用用量统计
# ---------------------------------------------------------------------------

def _extract_token_usage(response: Any) -> Dict[str, int]:
    """兼容 ``usage_metadata`` 与 ``response_metadata.token_usage`` 两种来源。"""
    usage_metadata = getattr(response, "usage_metadata", None) or {}
    response_metadata = getattr(response, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage", {}) if isinstance(response_metadata, dict) else {}

    input_tokens = int(usage_metadata.get("input_tokens") or token_usage.get("prompt_tokens") or 0)
    output_tokens = int(usage_metadata.get("output_tokens") or token_usage.get("completion_tokens") or 0)

    input_details = usage_metadata.get("input_token_details", {}) if isinstance(usage_metadata, dict) else {}
    prompt_details = token_usage.get("prompt_tokens_details", {}) if isinstance(token_usage, dict) else {}
    cached_input_tokens = int(input_details.get("cache_read") or prompt_details.get("cached_tokens") or 0)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
    }


def _calculate_llm_cost_rmb(
    model_family: ModelFamily,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int,
) -> float:
    billable_input_tokens = max(input_tokens - cached_input_tokens, 0)
    if model_family == "analysis":
        input_price = ANALYSIS_INPUT_PRICE_RMB_PER_1K
        output_price = ANALYSIS_OUTPUT_PRICE_RMB_PER_1K
        cached_input_price = ANALYSIS_CACHED_INPUT_PRICE_RMB_PER_1K
    else:
        input_price = RESEARCH_INPUT_PRICE_RMB_PER_1K
        output_price = RESEARCH_OUTPUT_PRICE_RMB_PER_1K
        cached_input_price = RESEARCH_CACHED_INPUT_PRICE_RMB_PER_1K

    return (
        billable_input_tokens / 1000.0 * input_price
        + output_tokens / 1000.0 * output_price
        + cached_input_tokens / 1000.0 * cached_input_price
    )


def _record_llm_usage(
    llm_usage: Optional[LLMUsage],
    model_family: ModelFamily,
    response: Any,
    llm_calls: Optional[Dict[str, int]] = None,
    usage_lock: Optional[Any] = None,
) -> None:
    """把单次 LLM 响应的 token / 成本累计写回 state 的 ``llm_usage`` / ``llm_calls``。"""
    if llm_usage is None or response is None:
        return

    usage = _extract_token_usage(response)
    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]
    cached_input_tokens = usage["cached_input_tokens"]
    cost_rmb = _calculate_llm_cost_rmb(
        model_family, input_tokens, output_tokens, cached_input_tokens
    )

    usage_call_key = f"{model_family}_calls"
    usage_input_key = f"{model_family}_input_tokens"
    usage_output_key = f"{model_family}_output_tokens"
    usage_cached_key = f"{model_family}_cached_input_tokens"
    usage_cost_key = f"{model_family}_cost_rmb"
    llm_call_key = f"{model_family}_llm"

    if usage_lock is not None:
        usage_lock.acquire()
    try:
        llm_usage[usage_call_key] += 1
        llm_usage[usage_input_key] += input_tokens
        llm_usage[usage_output_key] += output_tokens
        llm_usage[usage_cached_key] += cached_input_tokens
        llm_usage[usage_cost_key] += cost_rmb
        llm_usage["total_input_tokens"] += input_tokens
        llm_usage["total_output_tokens"] += output_tokens
        llm_usage["total_cached_input_tokens"] += cached_input_tokens
        llm_usage["total_cost_rmb"] += cost_rmb
        if llm_calls is not None:
            llm_calls[llm_call_key] = llm_calls.get(llm_call_key, 0) + 1
    finally:
        if usage_lock is not None:
            usage_lock.release()


# ---------------------------------------------------------------------------
# 后台任务同步
# ---------------------------------------------------------------------------

def wait_for_background_tasks(state: WorkflowState) -> WorkflowState:
    """等待 state 中登记的后台线程结束，确保成本/缓存写回统计完整。"""
    threads = list(state.get("background_threads", []) or [])
    for thread in threads:
        if thread and hasattr(thread, "join"):
            thread.join()
    state["background_threads"] = []
    return state


__all__ = [
    "CacheMatchType",
    "CacheReuseMode",
    "RerankAttempt",
    "ModelFamily",
    "LLMUsage",
    "WorkflowMetrics",
    "WorkflowState",
    "CacheStateView",
    "RoutingStateView",
    "initialize_metrics",
    "initialize_llm_usage",
    "build_initial_state",
    "update_metrics",
    "_extract_token_usage",
    "_calculate_llm_cost_rmb",
    "_record_llm_usage",
    "wait_for_background_tasks",
]
