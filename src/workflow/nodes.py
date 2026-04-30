import json
import logging  # 导入日志模块，记录程序运行状态
import os       # 导入操作系统接口，用于读取环境变量
import time     # 导入时间模块，用于性能耗时统计
from datetime import datetime  # 导入日期时间模块
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict  # 导入类型提示，增强代码可读性和健壮性

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
from common.env import ARK_BASE_URL, ANALYSIS_MODEL_NAME, RESEARCH_MODEL_NAME

# 获取名为 "agentic-workflow" 的日志记录器
logger = logging.getLogger("agentic-workflow")

# 全局变量占位符，用于实现 LLM 实例的单例模式（懒加载）
_analysis_llm = None
_research_llm = None

CacheMatchType = Literal["exact", "near_exact", "subquery_exact", "subquery_near_exact", "semantic", "none"]
CacheReuseMode = Literal["full_reuse", "partial_reuse", "reject", "none"]
RerankAttempt = Literal["none", "skipped", "primary", "fallback", "failed"]

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
            max_tokens=400,                                      # 限制输出长度
            api_key=os.getenv("ARK_API_KEY"),                    # 共享 API 密钥
            base_url=ARK_BASE_URL
        )
    return _research_llm

class WorkflowMetrics(TypedDict):
    """定义工作流性能指标的字典结构"""
    total_latency: float            # 总耗时
    cache_latency: float            # 缓存检查耗时
    rerank_latency: float           # 缓存复用裁判耗时
    research_latency: float         # 知识检索耗时
    supplement_latency: float       # 补充研究耗时
    synthesis_latency: float        # 回答合成耗时
    cache_hit_rate: float           # 缓存命中率 (0 或 1)
    cache_hits_count: int           # 缓存命中计数
    total_research_iterations: int  # 总研究循环次数

class WorkflowState(TypedDict):
    """定义整个计算图共享的状态对象结构（State）"""
    query: str                          # 用户原始提问
    answer: str                         # 节点间传递的中间或最终答案
    final_response: Optional[str]       # 最终渲染给用户的文本内容
    cache_hit: bool                     # 标记是否命中缓存
    cache_matched_question: Optional[str] # 缓存中匹配到的原始问题
    cache_confidence: float             # 缓存匹配的相似度分数
    cache_seed_id: Optional[int]        # 缓存数据在数据库中的原始 ID
    cache_match_type: CacheMatchType    # 缓存候选命中类型：exact/near_exact/subquery_exact/subquery_near_exact/semantic/none
    cache_base_answer: str              # 缓存命中时保留的原始缓存答案
    cache_enabled: bool                 # 是否启用缓存开关
    intercepted: bool                   # 是否已被前置拦截器拦截
    research_iterations: int            # 当前研究的迭代轮次
    cache_rerank_passed: bool           # LLM Reranker 是否判定缓存答案可复用
    cache_reuse_mode: CacheReuseMode    # full_reuse/partial_reuse/reject/none
    cache_rerank_attempt: RerankAttempt # rerank 实际采用的判定路径
    cache_rerank_score: float           # Reranker 给出的复用置信度
    cache_rerank_reason: str            # Reranker 给出的判定理由
    cache_residual_query: str           # partial_reuse 时尚未覆盖的缺口查询
    current_research_strategy: str      # 当前采取的检索策略描述
    execution_path: List[str]           # 记录工作流经过的节点路径
    metrics: WorkflowMetrics            # 性能监控数据
    timestamp: str                      # 任务启动时间戳
    llm_calls: Dict[str, int]           # 记录各个 LLM 的调用次数

def initialize_metrics() -> WorkflowMetrics:
    """初始化指标字典的默认值"""
    return {
        "total_latency": 0.0,
        "cache_latency": 0.0,
        "rerank_latency": 0.0,
        "research_latency": 0.0,
        "supplement_latency": 0.0,
        "synthesis_latency": 0.0,
        "cache_hit_rate": 0.0,
        "cache_hits_count": 0,
        "total_research_iterations": 0,
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
        "cache_residual_query": "",
        "current_research_strategy": "",
        "execution_path": ["start"],
        "metrics": initialize_metrics(),
        "timestamp": datetime.now().isoformat(),
        "llm_calls": {},
    }

def update_metrics(metrics: WorkflowMetrics, **kwargs) -> WorkflowMetrics:
    """
    通用指标更新函数。
    如果 key 是数值型则累加，否则直接覆盖。
    """
    new_metrics = metrics.copy()  # 浅拷贝原指标，保持函数纯净
    for key, value in kwargs.items():
        if key in new_metrics and isinstance(new_metrics[key], (int, float)):
            new_metrics[key] += value  # 累加耗时或计数
        else:
            new_metrics[key] = value   # 覆盖设置非数值属性
    return new_metrics

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
    
    if is_time_sensitive or mentions_product_model or mentions_inventory:
        logger.warning(f"   ⛔ 拦截触发: 时间实体={time_entities}, 特定商品={mentions_product_model}, 库存={mentions_inventory}")
        canned_response = "抱歉，我们这个助手无法获取具体的实时信息（如动态时间查询、实时库存或某些精确具体的商品型号信息）。如需进一步确认，请咨询人工客服。"
        return {
            **state,
            "answer": canned_response,
            "final_response": canned_response,
            "intercepted": True,
            "execution_path": state.get("execution_path", []) + ["pre_check_intercepted"]
        }
    
    logger.info("   ✅ 通过前置检查，放行")
    return {
        **state,
        "intercepted": False,
        "execution_path": state.get("execution_path", []) + ["pre_check_passed"],
    }

def check_cache_node(state: WorkflowState) -> WorkflowState:
    """节点：检查语义缓存（第一道防线）"""
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
            logger.info(f"   ✅ 缓存命中[{cache_match_type}] ({cache_confidence:.3f}): '{query}' -> 匹配到了 '{cache_matched_question}'")
        else:  # 未命中
            cache_hit = False
            cache_matched_question = None
            cache_confidence = 0.0
            cache_seed_id = None
            cache_match_type = "none"
            cache_base_answer = ""
            answer = ""
            logger.info(f"   ❌ 缓存未命中: '{query}'")

    # 计算该节点耗时（毫秒）
    cache_time = (time.perf_counter() - start_time) * 1000
    
    # 更新性能指标
    metrics = state.get("metrics", initialize_metrics())
    metrics = update_metrics(
        metrics,
        cache_latency=cache_time,
        cache_hits_count=1 if cache_hit else 0,
        cache_hit_rate=1.0 if cache_hit else 0.0
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
    match = re.search(r"\{.*\}", raw_text, re.S)
    if not match:
        raise ValueError("未找到 JSON 对象")
    return match.group(0)

def _parse_reranker_fallback_response(raw_text: str) -> RerankerEvaluation:
    payload = json.loads(_extract_json_object(raw_text))
    return RerankerEvaluation(
        reuse_mode=_normalize_reuse_mode(str(payload.get("reuse_mode", "reject"))),
        score=float(payload.get("score", 0.0) or 0.0),
        reason=str(payload.get("reason", "") or ""),
        residual_query=str(payload.get("residual_query", "") or ""),
    )

def _invoke_reranker(query: str, cached_question: str, cached_answer: str) -> Tuple[RerankerEvaluation, int, str]:
    attempts = _build_rerank_attempts(query, cached_question, cached_answer)
    primary = attempts[0]
    logger.info(
        "   🧪 Reranker 输入长度[primary]: q=%s cq=%s ca=%s",
        len(primary["query"]),
        len(primary["cached_question"]),
        len(primary["cached_answer_excerpt"]),
    )

    try:
        structured_llm = get_analysis_llm().with_structured_output(RerankerEvaluation)
        result = structured_llm.invoke([
            SystemMessage(content=RERANK_SYSTEM_PROMPT),
            HumanMessage(content=RERANK_PROMPT.format(
                query=primary["query"],
                cached_question=primary["cached_question"],
                cached_answer_excerpt=primary["cached_answer_excerpt"],
            )),
        ])
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
        raw_result = get_analysis_llm().invoke([
            SystemMessage(content=RERANK_SYSTEM_PROMPT),
            HumanMessage(content=RERANK_FALLBACK_PROMPT.format(
                query=fallback["query"],
                cached_question=fallback["cached_question"],
                cached_answer_excerpt=fallback["cached_answer_excerpt"],
            )),
        ])
        try:
            return _parse_reranker_fallback_response(raw_result.content if hasattr(raw_result, "content") else str(raw_result)), 2, fallback["label"]
        except Exception as fallback_exc:
            raise RuntimeError(f"structured={primary_exc}; fallback={fallback_exc}") from fallback_exc

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

def _should_allow_partial_reuse(query: str, cached_answer: str, residual_query: str, score: float) -> Tuple[bool, str]:
    normalized_query = " ".join(query.split())
    normalized_residual = " ".join(residual_query.split())
    if not normalized_residual or normalized_residual == normalized_query:
        return False, "缺口不可分离"

    query_length = max(len(normalized_query), 1)
    residual_length = len(normalized_residual)
    residual_ratio = residual_length / query_length
    cached_answer_length = len(" ".join((cached_answer or "").split()))

    if score < PARTIAL_REUSE_MIN_SCORE:
        return False, "partial收益不足"
    if residual_length > PARTIAL_REUSE_MAX_RESIDUAL_CHARS:
        return False, "缺口过长"
    if residual_ratio > PARTIAL_REUSE_MAX_RESIDUAL_RATIO:
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
            "cache_residual_query": "",
            "execution_path": state.get("execution_path", []) + ["cache_rerank_skipped"],
        }

    logger.info(f"⚖️ 启动缓存复用裁判: '{query[:20]}...' vs 缓存问题 '{cached_question[:20]}...'")

    llm_calls = state.get("llm_calls", {}).copy()
    try:
        result, rerank_call_count, rerank_attempt_label = _invoke_reranker(query, cached_question, cached_answer)
        reuse_mode = _normalize_reuse_mode(result.reuse_mode)
        score = max(0.0, min(1.0, float(result.score)))
        reason = result.reason or ""
        residual_query = (result.residual_query or "").strip()
    except Exception as e:
        logger.exception(f"   ❌ Reranker 调用失败，安全降级: {e}")
        reuse_mode = "reject"
        score = 0.0
        reason = f"rerank_exception: {e}"
        residual_query = ""
        rerank_call_count = 2
        rerank_attempt_label = "failed"

    if reuse_mode == "partial_reuse":
        allow_partial_reuse, rejection_reason = _should_allow_partial_reuse(query, cached_answer, residual_query, score)
        if not allow_partial_reuse:
            reuse_mode = "reject"
            reason = rejection_reason or reason or "partial收益不足"
            residual_query = ""

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
            "cache_residual_query": "",
            "execution_path": state.get("execution_path", []) + ["cache_reranked_failed"],
            "llm_calls": llm_calls,
            "metrics": metrics,
    }

def prepare_research_messages(query: str, prompt_text: Optional[str] = None) -> Tuple[List[Any], int, bool]:
    """构造 research 阶段的消息历史。

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

    for _ in range(3):
        response = llm_with_tools.invoke(messages)
        research_llm_invocations += 1
        messages.append(response)
        if not response.tool_calls:
            return messages, research_llm_invocations, False

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            try:
                logger.info(f"   🔧 执行工具: {tool_name} {tool_args}")
                tool_result = tools[0].invoke(tool_args)
            except Exception as e:
                tool_result = f"Error executing tool: {e}"
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))

    if isinstance(messages[-1], ToolMessage) or (hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls):
        messages.append(HumanMessage(content="检索已经结束，请根据以上的检索结果，用自然流利的语言直接给出答案，不要列出原始段落结构。"))
        return messages, research_llm_invocations, True

    return messages, research_llm_invocations, False

def execute_research(query: str, prompt_text: Optional[str] = None) -> Tuple[str, int]:
    """执行一次 research 流程，返回最终答案与 LLM 调用次数。"""
    messages, research_llm_invocations, needs_final_generation = prepare_research_messages(query, prompt_text=prompt_text)

    if needs_final_generation:
        response = get_research_llm().invoke(messages)
        research_llm_invocations += 1
        messages.append(response)

    final_answer = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
    return final_answer, research_llm_invocations

def merge_partial_answers(original_query: str, cached_answer: str, supplemental_answer: str) -> str:
    """将缓存答案与补充研究结果合并为最终回复。"""
    response = get_research_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=PARTIAL_REUSE_MERGE_PROMPT.format(
            original_query=original_query,
            cached_answer=cached_answer,
            supplemental_answer=supplemental_answer,
        )),
    ])
    return response.content if hasattr(response, "content") else str(response)

def research_node(state: WorkflowState) -> WorkflowState:
    """节点：执行深度研究/知识库检索"""
    start_time = time.perf_counter()  # 记录节点起始时间
    query = state["query"]            # 用户提问
    iteration = state.get("research_iterations", 0) + 1  # 轮次加 1
    
    logger.info(f"🔍 正在研究: '{query}'")
    final_answer, research_llm_invocations = execute_research(query)
    
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
    """节点：只补充缓存未覆盖的缺口问题，并与缓存答案合并。"""
    start_time = time.perf_counter()
    original_query = state["query"]
    residual_query = state.get("cache_residual_query") or original_query
    cached_answer = state.get("cache_base_answer") or state.get("answer") or ""

    logger.info(f"🔎 正在补充研究缺口: '{residual_query}'")

    supplement_prompt = RESEARCH_PROMPT_SUPPLEMENT.format(
        original_query=original_query,
        cached_answer=_clip_rerank_answer(cached_answer, max_chars=400),
        residual_query=residual_query,
    )
    supplemental_answer, research_llm_invocations = execute_research(residual_query, prompt_text=supplement_prompt)

    merge_llm_invocations = 0
    normalized_supplemental_answer = supplemental_answer.strip().rstrip("。！!?")
    if normalized_supplemental_answer == "无需补充":
        final_answer = cached_answer
    else:
        try:
            if _should_use_merge_llm(cached_answer, supplemental_answer):
                final_answer = merge_partial_answers(original_query, cached_answer, supplemental_answer)
                merge_llm_invocations = 1
            else:
                final_answer = _merge_partial_answers_without_llm(cached_answer, supplemental_answer)
        except Exception as e:
            logger.exception(f"   ❌ 合并部分复用回答失败，降级为模板拼接: {e}")
            final_answer = f"{cached_answer}\n\n补充说明：\n{supplemental_answer}"

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
        "execution_path": state["execution_path"] + ["supplement_researched"],
        "llm_calls": llm_calls,
        "metrics": metrics,
    }

def synthesize_response_node(state: WorkflowState) -> WorkflowState:
    """节点：合成最终用户响应（并执行缓存写回）"""
    start_time = time.perf_counter()  # 记录开始时间
    
    llm_calls = state.get("llm_calls", {}).copy()
    
    # 获取最核心的答案呈现给用户
    final_response = state['answer']
    
    # --- 【自学习逻辑】：将研究阶段产出的新答案回填至语义缓存 ---
    # 仅当本次回答来自 research（不是缓存直出，也不是 pre_check 拦截）且答案非空时，才写回缓存。
    if (
        not state.get("cache_hit", False)
        and not state.get("intercepted", False)
        and _cache_instance
        and final_response
        and ({"researched", "supplement_researched"} & set(state.get("execution_path", [])))
    ):
        logger.info(f"   💾 将研究得到的回答写入语义缓存: '{state['query'][:20]}...'")
        _cache_instance.cache.store(prompt=state["query"], response=state["answer"])

        # 手动维护内存字典供 fuzzy_matches 短路使用
        if not hasattr(_cache_instance, '_seed_id_by_question'):
            _cache_instance._seed_id_by_question = {}
        if not hasattr(_cache_instance, '_answer_by_question'):
            _cache_instance._answer_by_question = {}
        if not hasattr(_cache_instance, '_normalized_question_map'):
            _cache_instance._normalized_question_map = {}
        if not hasattr(_cache_instance, '_near_exact_question_map'):
            _cache_instance._near_exact_question_map = {}

        _cache_instance._seed_id_by_question[state["query"]] = None
        _cache_instance._answer_by_question[state["query"]] = state["answer"]
        if hasattr(_cache_instance, 'normalize_query'):
            _cache_instance._normalized_question_map[_cache_instance.normalize_query(state["query"])] = state["query"]
        if hasattr(_cache_instance, 'normalize_surface_query'):
            _cache_instance._near_exact_question_map[_cache_instance.normalize_surface_query(state["query"])] = state["query"]
    
    # 合成耗时统计
    synth_time = (time.perf_counter() - start_time) * 1000
    metrics = update_metrics(state.get("metrics", initialize_metrics()), synthesis_latency=synth_time)
    
    # 返回包含最终响应的完整状态，结束工作流
    return {
        **state,
        "final_response": final_response,
        "execution_path": state["execution_path"] + ["synthesized"],
        "llm_calls": llm_calls,
        "metrics": metrics,
    }