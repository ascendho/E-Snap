import json
import logging  # 导入日志模块，记录程序运行状态
import os       # 导入操作系统接口，用于读取环境变量
import threading
import time     # 导入时间模块，用于性能耗时统计
import unicodedata
from typing import Any, Dict, List, Literal, Optional, Tuple  # 导入类型提示，增强代码可读性和健壮性

from langchain_core.messages import HumanMessage, SystemMessage  # 导入 LangChain 的消息对象
from langchain_openai import ChatOpenAI           # 导入 LangChain 的 OpenAI 兼容模型接口
from pydantic import BaseModel, Field
from workflow.tools import search_knowledge_base  # 导入自定义的知识库检索工具
from workflow.prompts import (
    SYSTEM_PROMPT,
    RESEARCH_PROMPT_INITIAL,
    RESEARCH_PROMPT_SUPPLEMENT,
    RERANK_SYSTEM_PROMPT,
    RERANK_PROMPT,
    PARTIAL_REUSE_MERGE_PROMPT,
)
from common.env import (
    ANALYSIS_MODEL_NAME,
    ARK_BASE_URL,
    RESEARCH_MODEL_NAME,
)
# 状态类型与状态相关 helper 集中在 workflow.state 中；本模块再导出以保持向后兼容。
from workflow.state import (
    CacheMatchType,
    CacheReuseMode,
    LLMUsage,
    ModelFamily,
    RerankAttempt,
    WorkflowMetrics,
    WorkflowState,
    _calculate_llm_cost_rmb,
    _extract_token_usage,
    _record_llm_usage,
    build_initial_state,
    initialize_llm_usage,
    initialize_metrics,
    update_metrics,
    wait_for_background_tasks,
)

# 获取名为 "agentic-workflow" 的日志记录器
logger = logging.getLogger("agentic-workflow")

# 全局变量占位符，用于实现 LLM 实例的单例模式（懒加载）
_analysis_llm = None
_research_llm = None


def get_analysis_llm():
    """获取用于缓存裁判等分析任务的 LLM 实例（低随机性，重逻辑）"""
    global _analysis_llm
    # 如果尚未初始化
    if _analysis_llm is None:  
        _analysis_llm = ChatOpenAI(
            model=ANALYSIS_MODEL_NAME,
            temperature=0.1,                                     # 设置低随机性，确保评估结果稳定
            max_tokens=1000,                                     # 为结构化输出与 fallback 留出余量
            api_key=os.getenv("ARK_API_KEY"),                    # 从环境变量获取 API 密钥
            base_url=ARK_BASE_URL
        )
    return _analysis_llm

def get_research_llm():
    """获取用于信息研究和文本生成的 LLM 实例（稍高随机性，重生成）"""
    global _research_llm
    # 如果尚未初始化
    if _research_llm is None:  
        _research_llm = ChatOpenAI(
            model=RESEARCH_MODEL_NAME,
            temperature=0.2,                                     # 设置适度的创造性
            stream_usage=True,                                   # 流式接口也返回 usage metadata，便于统一统计 token / 成本
            max_tokens=400,                                      # 限制输出长度
            api_key=os.getenv("ARK_API_KEY"),                    # 共享 API 密钥
            base_url=ARK_BASE_URL
        )
    return _research_llm

# 全局语义缓存实例占位符
_cache_instance = None
RERANK_QUERY_CHAR_LIMIT = 160
RERANK_QUESTION_CHAR_LIMIT = 160
RERANK_ANSWER_CHAR_LIMIT = 360
RERANK_FALLBACK_QUERY_CHAR_LIMIT = 96
RERANK_FALLBACK_QUESTION_CHAR_LIMIT = 96
RERANK_FALLBACK_ANSWER_CHAR_LIMIT = 180
PARTIAL_REUSE_MIN_SCORE = 0.97
PARTIAL_REUSE_MAX_RESIDUAL_CHARS = 18
PARTIAL_REUSE_MAX_RESIDUAL_RATIO = 0.28
PARTIAL_REUSE_SHORT_RESIDUAL_MIN_SCORE = 0.90
PARTIAL_REUSE_SHORT_RESIDUAL_MAX_CHARS = 5
PARTIAL_REUSE_SHORT_RESIDUAL_MAX_RATIO = 0.30
PARTIAL_REUSE_MAX_CACHED_ANSWER_CHARS = 320
RERANK_FALLBACK_PROMPT = """请仅输出一行 JSON，不要加解释。\n"
RERANK_FALLBACK_PROMPT += '{"reuse_mode":"full_reuse|partial_reuse|reject","score":0.0,"reason":"不超过20字","residual_query":"partial_reuse 时填写，否则留空"}\n'
RERANK_FALLBACK_PROMPT += "新问题：{query}\n旧问题：{cached_question}\n旧答案摘要：{cached_answer_excerpt}"""

def initialize_nodes(sys_cache):
    """初始化节点，注入语义缓存实例"""
    global _cache_instance
    _cache_instance = sys_cache

import re

def pre_check_node(state: WorkflowState) -> WorkflowState:
    """节点：前置拦截器（第零道防线）拦截时效性问题和特定商品型号查询等"""
    start_time = time.perf_counter()
    query = state["query"]
    logger.info(f"🛡️ 执行前置拦截检查: '{query}'")
    
    # 1. 检查时效性问题（基于 jieba 词性分析自动提取时间实体）
    import jieba.posseg as pseg
    is_time_sensitive = False
    time_entities = []
    
    for word, flag in pseg.cut(query):
        # flag 为 't' 或 'tg' 表示时间词
        if flag.startswith('t'):
            is_time_sensitive = True
            time_entities.append(word)
            
    if is_time_sensitive:
        logger.info(f"   ⏱️ 提取到时间实体: {time_entities}")
    
    # 2. 检查特定商品型号（使用正则匹配）
    product_model_pattern = re.compile(r'[a-zA-Z]+\d+(?:-\d+)+')
    mentions_product_model = bool(product_model_pattern.search(query))
    
    # 将库存作为业务关键词判断补齐（因为jieba时间判断不会涵盖它）
    mentions_inventory = "库存" in query

    precheck_time = (time.perf_counter() - start_time) * 1000
    metrics = update_metrics(state.get("metrics", initialize_metrics()), precheck_latency=precheck_time)
    
    if is_time_sensitive or mentions_product_model or mentions_inventory:
        logger.warning(f"   ⛔ 拦截触发: 时间实体={time_entities}, 特定商品={mentions_product_model}, 库存={mentions_inventory}")
        canned_response = "抱歉，我们这个助手无法获取具体的实时信息（如动态时间查询、实时库存或某些精确具体的商品型号信息）。如需进一步确认，请咨询人工客服。"
        return {
            **state,
            "answer": canned_response,
            "final_response": canned_response,
            "intercepted": True,
            "execution_path": state.get("execution_path", []) + ["pre_check_intercepted"],
            "metrics": metrics,
        }
    
    logger.info("   ✅ 通过前置检查，放行")
    return {
        **state,
        "intercepted": False,
        "execution_path": state.get("execution_path", []) + ["pre_check_passed"],
        "metrics": metrics,
    }

def check_cache_node(state: WorkflowState) -> WorkflowState:
    """节点：检查缓存并把结果翻译成“路由可消费”的状态。

    这个节点并不只是查 cache；它还负责把底层 cache engine 返回的结果
    转成 workflow 后续真正要消费的字段，例如：
    - `cache_hit`
    - `cache_match_type`
    - `cache_reuse_mode`
    - `cache_residual_query`

    尤其是 subquery 命中时，这里会提前决定它应该进入 partial_reuse fast path
    还是直接回退到 full RAG，而不是把这个判断拖到 router 再做。
    """
    start_time = time.perf_counter()  # 记录节点开始时间
    query = state["query"]            # 获取用户提问
    
    logger.info(f"🔍 开始检查语义缓存: '{query}'")
    
    # 检查缓存是否可用或已启用
    if not state.get("cache_enabled", True) or not _cache_instance:
        logger.info("   ⚠️ 缓存未启用或未初始化")
        cache_hit = False
        cache_matched_question = None
        cache_confidence = 0.0
        cache_seed_id = None
        cache_match_type = "none"
        cache_base_answer = ""
        answer = ""
        cache_reuse_mode = state.get("cache_reuse_mode", "none")
        cache_rerank_attempt = state.get("cache_rerank_attempt", "none")
        cache_rerank_score = state.get("cache_rerank_score", 0.0)
        cache_rerank_reason = state.get("cache_rerank_reason", "")
        cache_reranker_reason = state.get("cache_reranker_reason", "")
        cache_validation_reason = state.get("cache_validation_reason", "")
        cache_reranker_residual_query = state.get("cache_reranker_residual_query", "")
        cache_residual_query = state.get("cache_residual_query", "")
    else:
        # 在 Redis 中执行语义检索，设定阈值从 env 读取
        from common.env import CACHE_DISTANCE_THRESHOLD
        results = _cache_instance.check(query, distance_threshold=CACHE_DISTANCE_THRESHOLD)
        if results.matches:  # 如果找到了足够相似的历史记录
            best_match = results.matches[0]
            cache_hit = True
            cache_matched_question = best_match.prompt
            cache_confidence = best_match.cosine_similarity
            cache_seed_id = best_match.seed_id
            cache_match_type = best_match.match_type
            cache_base_answer = best_match.response
            answer = best_match.response
            cache_reuse_mode = "none"
            cache_rerank_attempt = "none"
            cache_rerank_score = 0.0
            cache_rerank_reason = ""
            cache_reranker_reason = ""
            cache_validation_reason = ""
            cache_reranker_residual_query = ""
            cache_residual_query = ""

            if str(cache_match_type).startswith("subquery_"):
                # subquery 命中只说明“复合问题中的某一段曾被问过”。
                # 真正关键的是：剩余未覆盖部分能否被稳定、唯一地提取出来。
                deterministic_residual = _derive_deterministic_subquery_residual(query, cache_matched_question)
                if deterministic_residual:
                    # 这里故意把 `cache_hit` 翻回 False：
                    # 后续路径不应被当作“整题缓存命中”，而应被当作
                    # “已有部分答案，可直接补缺口”的 partial_reuse 流程。
                    cache_hit = False
                    answer = ""
                    cache_reuse_mode = "partial_reuse"
                    cache_rerank_attempt = "skipped"
                    cache_rerank_reason = "deterministic_subquery_fastpath"
                    cache_reranker_reason = "deterministic_subquery_fastpath"
                    cache_reranker_residual_query = deterministic_residual
                    cache_residual_query = deterministic_residual
                    logger.info(
                        "   🧩 规则子问题命中，跳过 Reranker 直接补充研究: '%s' | 缺口: '%s'",
                        query,
                        deterministic_residual,
                    )
                else:
                    # 缺口不确定（0 个或 >1 个未覆盖段落）：跳过 Reranker，直接走全量 RAG。
                    # Reranker 无法正确处理「子问题答案 vs 复合问题」的局部匹配场景，
                    # 此时让它通过只会浪费一次 LLM 调用后依然被拒绝。
                    cache_hit = False
                    cache_matched_question = None
                    cache_confidence = 0.0
                    cache_seed_id = None
                    cache_match_type = "none"
                    cache_base_answer = ""
                    answer = ""
                    cache_reuse_mode = "none"
                    logger.info(
                        "   ⚠️ 子问题命中但缺口不确定，跳过 Reranker 直接走 RAG: '%s'",
                        query,
                    )
            else:
                # 对 exact / near_exact / edit_distance / semantic，这里只负责保留候选事实；
                # 真正是否直出、是否 rerank，交给 router 再判断。
                logger.info(f"   ✅ 缓存命中[{cache_match_type}] ({cache_confidence:.3f}): '{query}' -> 匹配到了 '{cache_matched_question}'")
        else:  # 未命中
            cache_hit = False
            cache_matched_question = None
            cache_confidence = 0.0
            cache_seed_id = None
            cache_match_type = "none"
            cache_base_answer = ""
            answer = ""
            cache_reuse_mode = "none"
            cache_rerank_attempt = "none"
            cache_rerank_score = 0.0
            cache_rerank_reason = ""
            cache_reranker_reason = ""
            cache_validation_reason = ""
            cache_reranker_residual_query = ""
            cache_residual_query = ""
            logger.info(f"   ❌ 缓存未命中: '{query}'")

    # 计算该节点耗时（毫秒）
    cache_time = (time.perf_counter() - start_time) * 1000
    
    # 更新性能指标
    metrics = state.get("metrics", initialize_metrics())
    metrics = update_metrics(
        metrics,
        cache_latency=cache_time,
    )
    
    # 返回更新后的状态
    return {
        **state,
        "answer": answer,
        "cache_hit": cache_hit,
        "cache_matched_question": cache_matched_question,
        "cache_confidence": cache_confidence,
        "cache_seed_id": cache_seed_id,
        "cache_match_type": cache_match_type,
        "cache_base_answer": cache_base_answer,
        "cache_reuse_mode": cache_reuse_mode,
        "cache_rerank_attempt": cache_rerank_attempt,
        "cache_rerank_score": cache_rerank_score,
        "cache_rerank_reason": cache_rerank_reason,
        "cache_reranker_reason": cache_reranker_reason,
        "cache_validation_reason": cache_validation_reason,
        "cache_reranker_residual_query": cache_reranker_residual_query,
        "cache_residual_query": cache_residual_query,
        "execution_path": state["execution_path"] + ["cache_checked"], # 更新执行路径轨迹
        "metrics": metrics,
    }

# ---------------------------------------------------------------------------
# 缓存语义复用裁判（LLM Reranker）
# ---------------------------------------------------------------------------
class RerankerEvaluation(BaseModel):
    """LLM Reranker 的结构化输出。"""
    reuse_mode: Literal["full_reuse", "partial_reuse", "reject"] = Field(description="缓存答案对新问题的复用模式")
    score: float = Field(description="对该判定的置信度，范围 0.0 到 1.0")
    reason: str = Field(description="一句极短中文理由，20字以内")
    residual_query: str = Field(description="若为 partial_reuse，用一句独立可检索的话描述未覆盖的部分；否则留空")

def _clip_rerank_answer(answer: str, max_chars: int = RERANK_ANSWER_CHAR_LIMIT) -> str:
    """压缩缓存答案，避免 Reranker 因上下文过长而耗尽输出预算。"""
    compact = " ".join(answer.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "..."

def _build_rerank_attempts(query: str, cached_question: str, cached_answer: str) -> List[Dict[str, str]]:
    return [
        {
            "label": "primary",
            "query": _clip_rerank_answer(query, max_chars=RERANK_QUERY_CHAR_LIMIT),
            "cached_question": _clip_rerank_answer(cached_question, max_chars=RERANK_QUESTION_CHAR_LIMIT),
            "cached_answer_excerpt": _clip_rerank_answer(cached_answer, max_chars=RERANK_ANSWER_CHAR_LIMIT),
        },
        {
            "label": "fallback",
            "query": _clip_rerank_answer(query, max_chars=RERANK_FALLBACK_QUERY_CHAR_LIMIT),
            "cached_question": _clip_rerank_answer(cached_question, max_chars=RERANK_FALLBACK_QUESTION_CHAR_LIMIT),
            "cached_answer_excerpt": _clip_rerank_answer(cached_answer, max_chars=RERANK_FALLBACK_ANSWER_CHAR_LIMIT),
        },
    ]

def _extract_json_object(raw_text: str) -> str:
    cleaned_text = str(raw_text or "").strip()
    if cleaned_text.startswith("```"):
        cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text, flags=re.I)
        cleaned_text = re.sub(r"\s*```$", "", cleaned_text)

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned_text):
        try:
            payload, _ = decoder.raw_decode(cleaned_text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return json.dumps(payload, ensure_ascii=False)

    raise ValueError("未找到可解析 JSON 对象")

def _build_safe_reranker_reject(reason: str) -> RerankerEvaluation:
    fallback_reason = (reason or "fallback_parse_error").strip()[:64]
    return RerankerEvaluation(
        reuse_mode="reject",
        score=0.0,
        reason=fallback_reason,
        residual_query="",
    )

def _parse_reranker_fallback_response(raw_text: str) -> RerankerEvaluation:
    try:
        payload = json.loads(_extract_json_object(raw_text))
    except Exception as exc:
        logger.warning(
            "   ⚠️ Reranker fallback 解析失败，安全降级为 reject | error=%s | raw=%s",
            exc,
            _clip_rerank_answer(str(raw_text), max_chars=160),
        )
        return _build_safe_reranker_reject("fallback_parse_error")

    if not isinstance(payload, dict):
        logger.warning("   ⚠️ Reranker fallback 返回了非对象 JSON，安全降级为 reject | payload=%s", payload)
        return _build_safe_reranker_reject("fallback_json_not_object")

    if "reuse_mode" not in payload:
        logger.warning("   ⚠️ Reranker fallback 缺少 reuse_mode，安全降级为 reject | payload=%s", payload)
        return _build_safe_reranker_reject("fallback_missing_reuse_mode")

    try:
        score = float(payload.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0

    return RerankerEvaluation(
        reuse_mode=_normalize_reuse_mode(str(payload.get("reuse_mode", "reject"))),
        score=score,
        reason=str(payload.get("reason", "") or ""),
        residual_query=str(payload.get("residual_query", "") or ""),
    )

def _invoke_reranker(
    query: str,
    cached_question: str,
    cached_answer: str,
    llm_usage: Optional[LLMUsage] = None,
    usage_lock: Optional[Any] = None,
) -> Tuple[RerankerEvaluation, int, str]:
    attempts = _build_rerank_attempts(query, cached_question, cached_answer)
    primary = attempts[0]
    logger.info(
        "   🧪 Reranker 输入长度[primary]: q=%s cq=%s ca=%s",
        len(primary["query"]),
        len(primary["cached_question"]),
        len(primary["cached_answer_excerpt"]),
    )

    try:
        structured_llm = get_analysis_llm().with_structured_output(RerankerEvaluation, include_raw=True)
        result_bundle = structured_llm.invoke([
            SystemMessage(content=RERANK_SYSTEM_PROMPT),
            HumanMessage(content=RERANK_PROMPT.format(
                query=primary["query"],
                cached_question=primary["cached_question"],
                cached_answer_excerpt=primary["cached_answer_excerpt"],
            )),
        ])
        _record_llm_usage(llm_usage, "analysis", result_bundle.get("raw"), usage_lock=usage_lock)
        if result_bundle.get("parsing_error"):
            raise result_bundle["parsing_error"]
        result = result_bundle.get("parsed")
        if result is None:
            raise ValueError("structured_rerank_missing_parsed_result")
        return result, 1, primary["label"]
    except Exception as primary_exc:
        fallback = attempts[1]
        logger.warning(
            "   ⚠️ Structured rerank 失败，尝试紧凑 fallback | q=%s cq=%s ca=%s | error=%s",
            len(fallback["query"]),
            len(fallback["cached_question"]),
            len(fallback["cached_answer_excerpt"]),
            primary_exc,
        )
        try:
            raw_result = get_analysis_llm().invoke([
                SystemMessage(content=RERANK_SYSTEM_PROMPT),
                HumanMessage(content=RERANK_FALLBACK_PROMPT.format(
                    query=fallback["query"],
                    cached_question=fallback["cached_question"],
                    cached_answer_excerpt=fallback["cached_answer_excerpt"],
                )),
            ])
            _record_llm_usage(llm_usage, "analysis", raw_result, usage_lock=usage_lock)
        except Exception as fallback_invoke_exc:
            logger.warning(
                "   ⚠️ Fallback rerank 调用失败，安全降级为 reject | error=%s",
                fallback_invoke_exc,
            )
            return _build_safe_reranker_reject("fallback_invoke_error"), 2, fallback["label"]

        fallback_result = _parse_reranker_fallback_response(
            raw_result.content if hasattr(raw_result, "content") else str(raw_result)
        )
        return fallback_result, 2, fallback["label"]

def _normalize_reuse_mode(raw_mode: str) -> CacheReuseMode:
    mapped_modes = {
        "full_reuse": "full_reuse",
        "full": "full_reuse",
        "partial_reuse": "partial_reuse",
        "partial": "partial_reuse",
        "reject": "reject",
        "no_reuse": "reject",
        "none": "reject",
    }
    return mapped_modes.get((raw_mode or "").strip().lower(), "reject")

def _normalize_surface_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).lower().strip()
    collapsed = "".join(normalized.split())
    allowed_chars = []
    for char in collapsed:
        if unicodedata.category(char).startswith("P"):
            continue
        allowed_chars.append(char)
    return "".join(allowed_chars)

def _split_query_segments(query: str) -> List[str]:
    normalized = unicodedata.normalize("NFKC", str(query))
    # 逗号常出现在单个子问题内部，不能一律视为复合问题分隔符。
    for separator in ["？", "?", "！", "!", "。", "；", ";", "另外", "还有", "以及", "并且"]:
        normalized = normalized.replace(separator, "\n")
    segments = []
    for segment in normalized.splitlines():
        cleaned = segment.strip()
        if len(cleaned) >= 2:
            segments.append(cleaned)
    return segments

def _derive_deterministic_subquery_residual(query: str, cached_question: str) -> str:
    normalized_cached_question = _normalize_surface_text(cached_question)
    residual_segments = [
        segment for segment in _split_query_segments(query)
        if _normalize_surface_text(segment) != normalized_cached_question
    ]
    if len(residual_segments) != 1:
        return ""
    return residual_segments[0].strip()

def _refine_residual_query(query: str, cached_question: str, residual_query: str) -> str:
    """优先从原始复合问题中提炼真实缺口，避免 Reranker 输出过长的解释式 residual。"""
    segments = _split_query_segments(query)
    if len(segments) <= 1:
        return residual_query.strip()

    normalized_cached_question = _normalize_surface_text(cached_question)
    normalized_residual_query = _normalize_surface_text(residual_query)
    remaining_segments = [
        segment for segment in segments
        if _normalize_surface_text(segment) != normalized_cached_question
    ]
    if not remaining_segments:
        return residual_query.strip()

    if normalized_residual_query:
        for segment in remaining_segments:
            normalized_segment = _normalize_surface_text(segment)
            if normalized_segment and (
                normalized_segment in normalized_residual_query
                or normalized_residual_query in normalized_segment
            ):
                return segment.strip()

    return min(remaining_segments, key=len).strip()

def _build_supplement_cache_writeback_entries(
    original_query: str,
    residual_query: str,
    reranker_residual_query: str,
    supplemental_answer: str,
) -> List[Dict[str, str]]:
    normalized_answer = supplemental_answer.strip().rstrip("。！？!?")
    if not normalized_answer or normalized_answer == "无需补充":
        return []

    entries: List[Dict[str, str]] = []
    seen_prompts = set()
    normalized_original_query = _normalize_surface_text(original_query)
    for prompt in [residual_query, reranker_residual_query]:
        cleaned_prompt = (prompt or "").strip()
        if len(cleaned_prompt) < 2:
            continue
        normalized_prompt = _normalize_surface_text(cleaned_prompt)
        if not normalized_prompt or normalized_prompt == normalized_original_query or normalized_prompt in seen_prompts:
            continue
        entries.append({"prompt": cleaned_prompt, "answer": supplemental_answer})
        seen_prompts.add(normalized_prompt)

    return entries

def _should_allow_partial_reuse(query: str, cached_answer: str, residual_query: str, score: float) -> Tuple[bool, str]:
    normalized_query = " ".join(query.split())
    normalized_residual = " ".join(residual_query.split())
    if not normalized_residual or normalized_residual == normalized_query:
        return False, "缺口不可分离"

    query_length = max(len(normalized_query), 1)
    residual_length = len(normalized_residual)
    residual_ratio = residual_length / query_length
    cached_answer_length = len(" ".join((cached_answer or "").split()))

    is_short_residual_candidate = (
        residual_length <= PARTIAL_REUSE_SHORT_RESIDUAL_MAX_CHARS
        and residual_ratio <= PARTIAL_REUSE_SHORT_RESIDUAL_MAX_RATIO
    )
    required_score = PARTIAL_REUSE_SHORT_RESIDUAL_MIN_SCORE if is_short_residual_candidate else PARTIAL_REUSE_MIN_SCORE

    if score < required_score:
        return False, "partial收益不足"
    if residual_length > PARTIAL_REUSE_MAX_RESIDUAL_CHARS:
        return False, "缺口过长"
    if residual_ratio > PARTIAL_REUSE_MAX_RESIDUAL_RATIO and not is_short_residual_candidate:
        return False, "缺口占比过高"
    if cached_answer_length > PARTIAL_REUSE_MAX_CACHED_ANSWER_CHARS:
        return False, "缓存答案过长"
    return True, ""

def _merge_partial_answers_without_llm(cached_answer: str, supplemental_answer: str) -> str:
    cached = (cached_answer or "").strip()
    supplemental = (supplemental_answer or "").strip()
    if not cached:
        return supplemental
    if not supplemental:
        return cached
    if supplemental in cached:
        return cached
    return f"{cached}\n\n补充说明：\n{supplemental}"

def _should_use_merge_llm(cached_answer: str, supplemental_answer: str) -> bool:
    cached_length = len((cached_answer or "").strip())
    supplemental_length = len((supplemental_answer or "").strip())
    if supplemental_length <= 120:
        return False
    if cached_length + supplemental_length <= 260:
        return False
    return True


def _coerce_tool_args(
    tool_name: str,
    tool_args: Any,
    *,
    locked_search_query: Optional[str] = None,
) -> Dict[str, Any]:
    """规范化工具参数，并在 supplement 模式下强制锁定检索词。

    正常情况下，tool_args 由 LLM 自己生成；
    但在 `research_supplement_node()` 中，我们只希望它围绕 residual_query 补缺口，
    不希望模型又把检索词扩写回原始整题。

    因此当 `locked_search_query` 存在且工具是 `search_knowledge_base` 时，
    会把实际执行的 query 强制改成 locked_search_query。
    """
    if isinstance(tool_args, dict):
        normalized_args = tool_args.copy()
    elif tool_args is None:
        normalized_args = {}
    else:
        normalized_args = {"query": str(tool_args).strip()}

    if tool_name != "search_knowledge_base" or not locked_search_query:
        return normalized_args

    coerced_args = normalized_args.copy()
    coerced_args["query"] = locked_search_query
    coerced_args.setdefault("top_k", normalized_args.get("top_k", 3))
    return coerced_args

def rerank_cache_node(state: WorkflowState) -> WorkflowState:
    """节点：对缓存命中候选做一次 LLM 语义复用裁定。

    规则：
    - 若 check_cache 没有候选，直接放行（路由器层面其实不会进到这里，这里只是双保险）。
    - 若候选存在，调用 ANALYSIS_MODEL_NAME 判定缓存答案能否复用。
    - 通过：保留 cache_hit 与 answer，由后续路由进入 synthesize_response。
    - 拒绝：等价为缓存未命中，清空 answer 并把 cache_hit 翻成 False，让流程走 RAG。
    - 异常：安全降级为拒绝，路由到 research。
    """
    start_time = time.perf_counter()
    query = state["query"]
    cache_hit = state.get("cache_hit", False)
    cached_question = state.get("cache_matched_question") or ""
    cached_answer = state.get("answer") or ""

    # 没有命中则不需要 rerank
    if not cache_hit or not cached_question:
        logger.info("   ⚪ 无缓存候选，跳过 Reranker")
        return {
            **state,
            "cache_rerank_passed": False,
            "cache_reuse_mode": "none",
            "cache_rerank_attempt": "skipped",
            "cache_rerank_score": 0.0,
            "cache_rerank_reason": "no_cache_candidate",
            "cache_reranker_reason": "",
            "cache_validation_reason": "no_cache_candidate",
            "cache_reranker_residual_query": "",
            "cache_residual_query": "",
            "execution_path": state.get("execution_path", []) + ["cache_rerank_skipped"],
        }

    logger.info(f"⚖️ 启动缓存复用裁判: '{query[:20]}...' vs 缓存问题 '{cached_question[:20]}...'")

    llm_calls = state.get("llm_calls", {}).copy()
    llm_usage = state.get("llm_usage")
    usage_lock = state.get("llm_usage_lock")
    original_residual_query = ""
    try:
        # `_invoke_reranker()` 内部可能做 primary + fallback 两轮尝试；
        # 本节点的职责是把复杂的 LLM 输出折叠成稳定的有限状态。
        result, rerank_call_count, rerank_attempt_label = _invoke_reranker(
            query,
            cached_question,
            cached_answer,
            llm_usage=llm_usage,
            usage_lock=usage_lock,
        )
        reuse_mode = _normalize_reuse_mode(result.reuse_mode)
        score = max(0.0, min(1.0, float(result.score)))
        reranker_reason = result.reason or ""
        validation_reason = ""
        reason = reranker_reason
        original_residual_query = (result.residual_query or "").strip()
        residual_query = _refine_residual_query(query, cached_question, original_residual_query)
        if original_residual_query and residual_query != original_residual_query:
            logger.info(
                "   ✂️ Residual 精简[%s]: '%s' -> '%s'",
                rerank_attempt_label,
                original_residual_query,
                residual_query,
            )
    except Exception as e:
        logger.exception(f"   ❌ Reranker 调用失败，安全降级: {e}")
        reuse_mode = "reject"
        score = 0.0
        reranker_reason = f"rerank_exception: {e}"
        validation_reason = ""
        reason = reranker_reason
        original_residual_query = ""
        residual_query = ""
        rerank_call_count = 2
        rerank_attempt_label = "failed"

    if reuse_mode == "partial_reuse":
        # partial_reuse 不会因为模型“说可以”就直接放行；
        # 这里还有一层纯规则后置校验，用来过滤收益不足或 residual 不可靠的情况。
        allow_partial_reuse, rejection_reason = _should_allow_partial_reuse(query, cached_answer, residual_query, score)
        if not allow_partial_reuse:
            validation_reason = rejection_reason or "partial收益不足"
            reuse_mode = "reject"
            reason = validation_reason
            logger.info(
                "   🧱 Partial reuse 后置校验拒绝[%s] (score=%.2f): reranker_reason=%s | validation_reason=%s | residual=%s",
                rerank_attempt_label,
                score,
                reranker_reason or "无",
                validation_reason,
                residual_query or "无",
            )

    llm_calls["analysis_llm"] = llm_calls.get("analysis_llm", 0) + rerank_call_count
    rerank_time = (time.perf_counter() - start_time) * 1000
    metrics = update_metrics(state.get("metrics", initialize_metrics()), rerank_latency=rerank_time)

    if reuse_mode == "full_reuse":
        logger.info(f"   ✅ Reranker 通过[{rerank_attempt_label}] (score={score:.2f}): {reason}")
        return {
            **state,
            "cache_rerank_passed": True,
            "cache_reuse_mode": "full_reuse",
            "cache_rerank_attempt": rerank_attempt_label,
            "cache_rerank_score": score,
            "cache_rerank_reason": reason,
            "cache_reranker_reason": reranker_reason,
            "cache_validation_reason": validation_reason,
            "cache_reranker_residual_query": original_residual_query,
            "cache_residual_query": "",
            "execution_path": state.get("execution_path", []) + ["cache_reranked_passed"],
            "llm_calls": llm_calls,
            "metrics": metrics,
        }

    if reuse_mode == "partial_reuse":
        logger.info(f"   🟡 Reranker 判定部分复用[{rerank_attempt_label}] (score={score:.2f}): {reason} | 缺口: {residual_query}")
        return {
            **state,
            "answer": "",
            "cache_hit": False,
            "cache_rerank_passed": False,
            "cache_reuse_mode": "partial_reuse",
            "cache_rerank_attempt": rerank_attempt_label,
            "cache_rerank_score": score,
            "cache_rerank_reason": reason,
            "cache_reranker_reason": reranker_reason,
            "cache_validation_reason": validation_reason,
            "cache_reranker_residual_query": original_residual_query,
            "cache_residual_query": residual_query,
            "execution_path": state.get("execution_path", []) + ["cache_reranked_partial"],
            "llm_calls": llm_calls,
            "metrics": metrics,
        }

    logger.info(f"   ⛔ Reranker 拒绝[{rerank_attempt_label}] (score={score:.2f}): {reason}，转入 RAG")
    return {
            **state,
            "answer": "",
            "cache_hit": False,
            "cache_rerank_passed": False,
            "cache_reuse_mode": "reject",
            "cache_rerank_attempt": rerank_attempt_label,
            "cache_rerank_score": score,
            "cache_rerank_reason": reason,
                "cache_reranker_reason": reranker_reason,
                "cache_validation_reason": validation_reason,
                "cache_reranker_residual_query": original_residual_query,
                "cache_residual_query": residual_query,
            "execution_path": state.get("execution_path", []) + ["cache_reranked_failed"],
            "llm_calls": llm_calls,
            "metrics": metrics,
    }

def prepare_research_messages(
    query: str,
    prompt_text: Optional[str] = None,
    llm_usage: Optional[LLMUsage] = None,
    usage_lock: Optional[Any] = None,
    locked_search_query: Optional[str] = None,
    research_mode: str = "research",
) -> Tuple[List[Any], int, bool]:
    """构造 research 阶段的消息历史。

    这里实现的是一个工程化的 ReAct 式循环：
    1. LLM 先基于当前消息判断是否需要调用工具
    2. 若调用工具，就把工具结果追加回消息历史
    3. LLM 再读取新历史继续推理
    4. 最多允许 3 轮往返，避免无限工具循环

    因此这个函数返回的不只是“最终答案前的 prompt”，
    而是一段已经混合了 system / human / ai / tool message 的完整对话历史。

    返回值：
    - messages: 已包含工具调用结果的消息序列
    - research_llm_invocations: 已发生的 research_llm 调用次数
    - needs_final_generation: 是否还需要最后一步自然语言生成
    """
    tools = [search_knowledge_base]
    research_llm_invocations = 0
    research_prompt = prompt_text or RESEARCH_PROMPT_INITIAL.format(query=query)
    llm_with_tools = get_research_llm().bind_tools(tools)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=research_prompt),
    ]

    from langchain_core.messages import ToolMessage

    # research 阶段允许一次回答内部最多 3 轮“LLM -> 工具 -> LLM”往返。
    # 这个上限是工程约束：既给模型留出补充检索空间，又避免无限工具循环。
    for _ in range(3):
        response = llm_with_tools.invoke(messages)
        _record_llm_usage(llm_usage, "research", response, usage_lock=usage_lock)
        research_llm_invocations += 1
        messages.append(response)
        if not response.tool_calls:
            # 一旦模型决定“不再调工具”，当前 response 就已经是最终答案，
            # 不需要额外的 final generation。
            return messages, research_llm_invocations, False

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            actual_tool_args = _coerce_tool_args(
                tool_name,
                tool_args,
                locked_search_query=locked_search_query,
            )
            try:
                if locked_search_query and tool_name == "search_knowledge_base":
                    planned_query = ""
                    if isinstance(tool_args, dict):
                        planned_query = str(tool_args.get("query", "") or "").strip()
                    actual_query = str(actual_tool_args.get("query", "") or "").strip()
                    if planned_query and planned_query != actual_query:
                        logger.info(
                            "   🔒 补充研究锁定检索词[%s]: planned='%s' -> actual='%s'",
                            research_mode,
                            planned_query,
                            actual_query,
                        )
                    logger.info(
                        "   🔧 执行工具[%s]: %s %s",
                        research_mode,
                        tool_name,
                        actual_tool_args,
                    )
                else:
                    logger.info(f"   🔧 执行工具: {tool_name} {actual_tool_args}")
                tool_result = tools[0].invoke(actual_tool_args)
            except Exception as e:
                tool_result = f"Error executing tool: {e}"
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))

    if isinstance(messages[-1], ToolMessage) or (hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls):
        # 如果 3 轮结束时最后一条仍是工具结果，说明模型还没来得及把检索结果“说成人话”。
        # 这里补一条人类提示，强制进入最后一步自然语言生成。
        messages.append(HumanMessage(content="检索已经结束，请根据以上的检索结果，用自然流利的语言直接给出答案，不要列出原始段落结构。"))
        return messages, research_llm_invocations, True

    return messages, research_llm_invocations, False

def execute_research(
    query: str,
    prompt_text: Optional[str] = None,
    llm_usage: Optional[LLMUsage] = None,
    usage_lock: Optional[Any] = None,
    locked_search_query: Optional[str] = None,
    research_mode: str = "research",
) -> Tuple[str, int]:
    """执行一次完整的 research 流程，返回最终答案与 LLM 调用次数。

    可以把它理解成 `prepare_research_messages()` 的执行器：
    - 先跑 ReAct 式工具循环
    - 如果工具循环结束时最后一条还不是自然语言答案，
      再补最后一轮纯生成
    - 最终统一返回答案文本和 research_llm 的调用总数
    """
    messages, research_llm_invocations, needs_final_generation = prepare_research_messages(
        query,
        prompt_text=prompt_text,
        llm_usage=llm_usage,
        usage_lock=usage_lock,
        locked_search_query=locked_search_query,
        research_mode=research_mode,
    )

    if needs_final_generation:
        # 只有当工具循环已经结束、但最后一句仍不是自然语言答案时，
        # 才补这一轮额外生成；这样可以避免无意义的多打一枪。
        response = get_research_llm().invoke(messages)
        _record_llm_usage(llm_usage, "research", response, usage_lock=usage_lock)
        research_llm_invocations += 1
        messages.append(response)

    final_answer = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
    return final_answer, research_llm_invocations

def merge_partial_answers(
    original_query: str,
    cached_answer: str,
    supplemental_answer: str,
    llm_usage: Optional[LLMUsage] = None,
    usage_lock: Optional[Any] = None,
) -> str:
    """将缓存答案与补充研究结果合并为最终回复。"""
    response = get_research_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=PARTIAL_REUSE_MERGE_PROMPT.format(
            original_query=original_query,
            cached_answer=cached_answer,
            supplemental_answer=supplemental_answer,
        )),
    ])
    _record_llm_usage(llm_usage, "research", response, usage_lock=usage_lock)
    return response.content if hasattr(response, "content") else str(response)

def research_node(state: WorkflowState) -> WorkflowState:
    """节点：执行完整 research / RAG 流程。

    这条路径通常发生在两种场景：
    - 完全未命中缓存
    - 命中了候选，但被 reranker 明确拒绝复用

    因此它代表的是“从头研究整道题”，而不是局部补充。
    """
    start_time = time.perf_counter()  # 记录节点起始时间
    query = state["query"]            # 用户提问
    iteration = state.get("research_iterations", 0) + 1  # 轮次加 1
    
    logger.info(f"🔍 正在研究: '{query}'")
    final_answer, research_llm_invocations = execute_research(
        query,
        llm_usage=state.get("llm_usage"),
        usage_lock=state.get("llm_usage_lock"),
    )
    
    # 更新 LLM 调用统计
    llm_calls = state.get("llm_calls", {}).copy()
    llm_calls["research_llm"] = llm_calls.get("research_llm", 0) + research_llm_invocations
    
    # 统计研究耗时和迭代数
    research_time = (time.perf_counter() - start_time) * 1000
    metrics = update_metrics(state.get("metrics", initialize_metrics()), research_latency=research_time, total_research_iterations=1)
    
    # 返回更新后的状态
    return {
        **state,
        "answer": final_answer,
        "research_iterations": iteration,
        "execution_path": state["execution_path"] + ["researched"],
        "llm_calls": llm_calls,
        "metrics": metrics,
    }

def research_supplement_node(state: WorkflowState) -> WorkflowState:
    """节点：只补充缓存未覆盖的缺口问题，并与缓存答案合并。

    这里有两条分支，必须分开理解：
    1. B1 缓存短路：如果 residual_query 本身已经在 L1 缓存中，完全不调用 research，
       直接复用缓存答案。这会把路径标成 `dual_subquery`。
    2. 真正的 supplement research：只有缺口问题不在缓存里时，才调用一次
       `execute_research(residual_query, locked_search_query=residual_query)`。

    所以“这是补充研究节点”与“本次没有调用 research”并不矛盾，
    因为节点名描述的是职责边界，不是每次运行都必然触发的子步骤。
    """
    start_time = time.perf_counter()
    original_query = state["query"]
    residual_query = state.get("cache_residual_query") or original_query
    reranker_residual_query = state.get("cache_reranker_residual_query") or residual_query
    cached_answer = state.get("cache_base_answer") or state.get("answer") or ""

    logger.info(f"🔎 正在补充研究缺口: '{residual_query}'")
    logger.info("   🔒 补充研究仅检索缺口问题，原问题只用于合并答案: '%s'", residual_query)

    # ── B1 缓存短路：若缺口问题本身已在 L1 缓存中，直接复用，完全跳过 research ──
    # 这就是 dual_subquery 路径的关键：复合问题的两段都已经有缓存，
    # 因而这个节点虽然叫 supplement，但本次运行中不会真正发起一次 supplement research。
    supplemental_answer: Optional[str] = None
    research_llm_invocations = 0
    _b1_from_cache = False
    if _cache_instance:
        from common.env import CACHE_DISTANCE_THRESHOLD
        residual_cache_result = _cache_instance.check(
            residual_query, distance_threshold=CACHE_DISTANCE_THRESHOLD
        )
        if residual_cache_result.matches:
            top = residual_cache_result.matches[0]
            if top.match_type in {
                "exact", "near_exact", "subquery_exact", "subquery_near_exact", "edit_distance"
            }:
                logger.info(
                    "   ⚡ 两个子问题均从缓存获取，跳过 RAG 补充缺口 '%s' [%s]",
                    residual_query,
                    top.match_type,
                )
                supplemental_answer = top.response
                _b1_from_cache = True

    if supplemental_answer is None:
        # 只有真的存在知识缺口时，才构造 supplement prompt 去做定向 research。
        # 这里会把检索词锁定为 residual_query，避免模型在补充阶段又跑回原始整题。
        supplement_prompt = RESEARCH_PROMPT_SUPPLEMENT.format(
            cached_answer=_clip_rerank_answer(cached_answer, max_chars=400),
            residual_query=residual_query,
        )
        supplemental_answer, research_llm_invocations = execute_research(
            residual_query,
            prompt_text=supplement_prompt,
            llm_usage=state.get("llm_usage"),
            usage_lock=state.get("llm_usage_lock"),
            locked_search_query=residual_query,
            research_mode="supplement",
        )

    merge_llm_invocations = 0
    normalized_supplemental_answer = supplemental_answer.strip().rstrip("。！!?")
    if normalized_supplemental_answer == "无需补充":
        final_answer = cached_answer
    elif _b1_from_cache:
        # B1: 两个子问题均来自缓存，无需 LLM 合并，直接模板拼接。
        # 这样做的核心收益不是“省一次调用”这么简单，
        # 而是避免合并 LLM 再次改写已知正确答案，降低无意义漂移。
        final_answer = _merge_partial_answers_without_llm(cached_answer, supplemental_answer)
    else:
        try:
            if _should_use_merge_llm(cached_answer, supplemental_answer):
                final_answer = merge_partial_answers(
                    original_query,
                    cached_answer,
                    supplemental_answer,
                    llm_usage=state.get("llm_usage"),
                    usage_lock=state.get("llm_usage_lock"),
                )
                merge_llm_invocations = 1
            else:
                final_answer = _merge_partial_answers_without_llm(cached_answer, supplemental_answer)
        except Exception as e:
            logger.exception(f"   ❌ 合并部分复用回答失败，降级为模板拼接: {e}")
            final_answer = f"{cached_answer}\n\n补充说明：\n{supplemental_answer}"

    cache_writeback_entries = _build_supplement_cache_writeback_entries(
        original_query=original_query,
        residual_query=residual_query,
        reranker_residual_query=reranker_residual_query,
        supplemental_answer=supplemental_answer,
    )

    llm_calls = state.get("llm_calls", {}).copy()
    llm_calls["research_llm"] = llm_calls.get("research_llm", 0) + research_llm_invocations + merge_llm_invocations

    supplement_time = (time.perf_counter() - start_time) * 1000
    metrics = update_metrics(
        state.get("metrics", initialize_metrics()),
        supplement_latency=supplement_time,
        total_research_iterations=1,
    )

    return {
        **state,
        "answer": final_answer,
        "research_iterations": state.get("research_iterations", 0) + 1,
        "cache_writeback_entries": cache_writeback_entries,
        "execution_path": state["execution_path"] + ["supplement_researched"],
        "cache_reuse_mode": "dual_subquery" if _b1_from_cache else state.get("cache_reuse_mode", "none"),
        "llm_calls": llm_calls,
        "metrics": metrics,
    }

def _store_cache_entry(prompt: str, answer: str) -> None:
    """Persist a runtime QA pair through the cache wrapper.

    Delegates to `SemanticCacheWrapper.register_entry` so all writers (FAQ seed,
    synthesize_response writeback, background subquery writeback) share one code path
    for keeping the L1 lookup maps consistent.
    """
    if not _cache_instance or not prompt or not answer:
        return

    register = getattr(_cache_instance, "register_entry", None)
    if callable(register):
        register(prompt, answer)
        return

    # Defensive fallback for any custom cache shim that lacks register_entry.
    _cache_instance.cache.store(prompt=prompt, response=answer)

def _cache_contains_prompt_variant(prompt: str) -> bool:
    if not _cache_instance or not prompt:
        return False

    contains = getattr(_cache_instance, "contains_prompt_variant", None)
    if callable(contains):
        return bool(contains(prompt))

    if hasattr(_cache_instance, 'normalize_query'):
        normalized_prompt = _cache_instance.normalize_query(prompt)
        if normalized_prompt in getattr(_cache_instance, '_normalized_question_map', {}):
            return True

    if hasattr(_cache_instance, 'normalize_surface_query'):
        normalized_surface_prompt = _cache_instance.normalize_surface_query(prompt)
        if normalized_surface_prompt in getattr(_cache_instance, '_near_exact_question_map', {}):
            return True

    return False

_SEGMENT_EXTRACT_PROMPT = (
    "以下是一个复合问题的完整回答。"
    "请只提取与子问题\"{segment}\"直接相关的那部分内容，"
    "不超过200字，直接输出答案文本，不要加任何前缀或解释。\n\n"
    "复合问题完整回答：\n{combined_answer}"
)

def _cache_segments_background(
    segments: List[str],
    combined_answer: str,
    llm_usage: Optional[LLMUsage],
    llm_calls: Optional[Dict[str, int]],
    usage_lock: Optional[Any],
) -> None:
    """后台线程：用 analysis_llm 从合并答案中提取每个子问题的精准部分并写入缓存。
    以 daemon 线程运行，不阻塞主线程/用户响应。无工具调用，成本极低。"""
    try:
        for segment in segments:
            try:
                logger.info(f"   🔍 [后台] 从合并答案中提取子问题答案: '{segment[:20]}...'")
                prompt = _SEGMENT_EXTRACT_PROMPT.format(
                    segment=segment,
                    combined_answer=_clip_rerank_answer(combined_answer, max_chars=600),
                )
                response = get_analysis_llm().invoke([HumanMessage(content=prompt)])
                _record_llm_usage(llm_usage, "analysis", response, llm_calls=llm_calls, usage_lock=usage_lock)
                segment_answer = (getattr(response, "content", "") or "").strip()
                if not segment_answer:
                    logger.warning(f"   ⚠️ [后台] 提取结果为空，跳过: '{segment[:20]}...'")
                    continue
                _store_cache_entry(segment, segment_answer)
                logger.info(f"   💾 [后台] 子问题缓存写入完成: '{segment[:20]}...'")
            except Exception as exc:
                logger.warning(f"   ⚠️ [后台] 子问题缓存失败，跳过: '{segment[:20]}...' | error={exc}")
    except Exception as outer_exc:  # pragma: no cover - defensive
        logger.exception(f"   💥 [后台] daemon 线程崩溃，后续子问题写入已中断: {outer_exc}")

def synthesize_response_node(state: WorkflowState) -> WorkflowState:
    """节点：合成最终用户响应（并执行缓存写回）。

    这是整个工作流的收口点：
    - 对用户来说，这里决定最终返回什么文本
    - 对系统来说，这里决定哪些问法 / 子问题应该被写回缓存

        关于“这里会不会调用 LLM”，要分主线程和后台线程看：
        - 主线程：不直接调用 research_llm，只做输出收口和写回决策
        - 后台线程：在 Full RAG 的复合问题场景下，可能调用 analysis_llm
            从合并答案里提取子问题答案，再异步写回缓存

    也正因为写回策略集中在这里，cache engine 本身才可以保持相对“无业务”的通用接口。
    """
    start_time = time.perf_counter()  # 记录开始时间
    
    llm_calls = state.get("llm_calls", {}).copy()
    llm_usage = state.get("llm_usage")
    background_threads = list(state.get("background_threads", []) or [])
    usage_lock = state.get("llm_usage_lock")
    
    # 获取最核心的答案呈现给用户
    final_response = state['answer']
    executed_paths = set(state.get("execution_path", []))
    # `cache_written_prompts` 是最终写回到 state 里的用户可观察列表；
    # 它会告诉前端“这次回答涉及了哪些 prompt 被写入或计划写入缓存”。
    cache_written_prompts: List[str] = []
    normalized_written_prompts = set()

    for prompt in state.get("cache_written_prompts", []) or []:
        cleaned_prompt = (prompt or "").strip()
        normalized_prompt = _normalize_surface_text(cleaned_prompt)
        if not cleaned_prompt or not normalized_prompt or normalized_prompt in normalized_written_prompts:
            continue
        cache_written_prompts.append(cleaned_prompt)
        normalized_written_prompts.add(normalized_prompt)

    def remember_written_prompt(prompt: str) -> None:
        cleaned_prompt = (prompt or "").strip()
        normalized_prompt = _normalize_surface_text(cleaned_prompt)
        if not cleaned_prompt or not normalized_prompt or normalized_prompt in normalized_written_prompts:
            return
        cache_written_prompts.append(cleaned_prompt)
        normalized_written_prompts.add(normalized_prompt)
    
    # --- 【自学习逻辑】：将研究阶段产出的新答案回填至语义缓存 ---
    # 仅当本次回答来自 research（不是缓存直出，也不是 pre_check 拦截）且答案非空时，才写回缓存。
    if (
        not state.get("cache_hit", False)
        and not state.get("intercepted", False)
        and _cache_instance
        and final_response
        and ({"researched", "supplement_researched"} & executed_paths)
    ):
        query_segments = _split_query_segments(state["query"])
        is_compound_query = len(query_segments) > 1

        # 对复合问题，默认优先缓存“稳定的子问题答案”，而不是立刻缓存整句问法。
        # 这是为了避免把一次性的合并措辞、临时拼接结构过早固化成 exact cache。
        # 单问题没有这个风险，因此可以直接写入完整 query。
        if not is_compound_query:
            logger.info(f"   💾 将研究得到的回答写入语义缓存: '{state['query'][:20]}...'")
            _store_cache_entry(state["query"], state["answer"])
            remember_written_prompt(state["query"])

        # `written_prompts` 是本函数内部使用的去重集合：
        # 它存的是归一化后的 prompt，用来保证同一轮写回过程中不会重复写入等价问法。
        # 和 `cache_written_prompts` 的区别是：
        # - `written_prompts` 偏内部控制，不直接暴露给前端
        # - `cache_written_prompts` 偏外部可观察，会进入最终 state / API 响应
        extra_cache_entries = state.get("cache_writeback_entries") or []
        written_prompts = {_normalize_surface_text(state["query"])}
        for entry in extra_cache_entries:
            prompt = (entry.get("prompt") or "").strip()
            answer = (entry.get("answer") or "").strip()
            normalized_prompt = _normalize_surface_text(prompt)
            if not prompt or not answer or normalized_prompt in written_prompts:
                continue
            logger.info(f"   💾 将补充研究得到的子问题答案写入缓存: '{prompt[:20]}...'")
            _store_cache_entry(prompt, answer)
            remember_written_prompt(prompt)
            written_prompts.add(normalized_prompt)

        # 对 Full RAG 路径，将复合问题中的子问题段落写入缓存（后台异步）。
        # 这里不是再次调用 research_llm 去重跑 RAG，
        # 而是让后台线程用 analysis_llm 从“已经生成好的完整答案”里提取每个子问题的相关部分。
        # 这样既不阻塞主线程，也能把一次复合回答拆解成更细粒度的缓存入口。
        if "researched" in executed_paths:
            pending_segments = []
            for segment in _split_query_segments(state["query"]):
                if len(segment) < 4:
                    continue
                normalized_segment = _normalize_surface_text(segment)
                if not normalized_segment or normalized_segment in written_prompts:
                    continue
                if _cache_contains_prompt_variant(segment):
                    continue
                pending_segments.append(segment)
                # 提前记录为"已写"（后台线程会很快完成），避免同会话重复调度。
                remember_written_prompt(segment)
                written_prompts.add(normalized_segment)
            if pending_segments:
                background_thread = threading.Thread(
                    target=_cache_segments_background,
                    args=(pending_segments, state["answer"], llm_usage, llm_calls, usage_lock),
                    daemon=True,
                )
                background_thread.start()
                background_threads.append(background_thread)

    if (
        state.get("cache_reuse_mode") == "dual_subquery"
        and not state.get("intercepted", False)
        and _cache_instance
        and final_response
        and not _cache_contains_prompt_variant(state["query"])
    ):
        # dual_subquery 是一个例外：
        # 既然两段子问题都已经确定来自缓存，那么当前整句复合问法也变成了“稳定问法”，
        # 这时写回整句可以让第三次相同提问直接走 exact / near_exact。
        logger.info(f"   💾 [dual_subquery] 将合并答案写入缓存，下次相同问题直接命中: '{state['query'][:30]}...'")
        _store_cache_entry(state["query"], final_response)
        remember_written_prompt(state["query"])

    if (
        state.get("cache_reuse_mode") == "full_reuse"
        and state.get("cache_rerank_passed", False)
        and not state.get("intercepted", False)
        and _cache_instance
        and final_response
        and not _cache_contains_prompt_variant(state["query"])
    ):
        # semantic / subquery 经 reranker 判定为 full_reuse 后，
        # 当前这句新问法已经被证明能安全映射到已有答案，因此值得写回为新入口。
        logger.info(f"   💾 将 full_reuse 接受后的当前问法写入缓存: '{state['query'][:20]}...'")
        _store_cache_entry(state["query"], final_response)
        remember_written_prompt(state["query"])
    
    # 合成耗时统计
    synth_time = (time.perf_counter() - start_time) * 1000
    metrics = update_metrics(state.get("metrics", initialize_metrics()), synthesis_latency=synth_time)
    
    # 返回包含最终响应的完整状态，结束工作流
    return {
        **state,
        "final_response": final_response,
        "cache_written_prompts": cache_written_prompts,
        "execution_path": state["execution_path"] + ["synthesized"],
        "llm_calls": llm_calls,
        "llm_usage": llm_usage,
        "llm_usage_lock": usage_lock,
        "background_threads": background_threads,
        "metrics": metrics,
    }