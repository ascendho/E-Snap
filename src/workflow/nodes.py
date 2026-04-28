import logging  # 导入日志模块，记录程序运行状态
import os       # 导入操作系统接口，用于读取环境变量
import time     # 导入时间模块，用于性能耗时统计
from datetime import datetime  # 导入日期时间模块
from typing import Any, Dict, List, Optional, TypedDict  # 导入类型提示，增强代码可读性和健壮性

from langchain_core.messages import HumanMessage, SystemMessage  # 导入 LangChain 的消息对象
from langchain_openai import ChatOpenAI           # 导入 LangChain 的 OpenAI 兼容模型接口
from pydantic import BaseModel, Field
from workflow.tools import search_knowledge_base  # 导入自定义的知识库检索工具
from workflow.prompts import SYSTEM_PROMPT, RESEARCH_PROMPT_INITIAL, RERANK_SYSTEM_PROMPT, RERANK_PROMPT
from common.env import ARK_BASE_URL, ANALYSIS_MODEL_NAME, RESEARCH_MODEL_NAME

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
            max_tokens=800,                                      # 为结构化输出留出足够余量
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
    cache_enabled: bool                 # 是否启用缓存开关
    intercepted: bool                   # 是否已被前置拦截器拦截
    research_iterations: int            # 当前研究的迭代轮次
    cache_rerank_passed: bool           # LLM Reranker 是否判定缓存答案可复用
    cache_rerank_score: float           # Reranker 给出的复用置信度
    cache_rerank_reason: str            # Reranker 给出的判定理由
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
        "cache_enabled": True,
        "intercepted": False,
        "research_iterations": 0,
        "cache_rerank_passed": False,
        "cache_rerank_score": 0.0,
        "cache_rerank_reason": "",
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
RERANK_ANSWER_CHAR_LIMIT = 600

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
        canned_response = "抱歉，我们这个助手无法获取具体的实时信息（如动态时间查询、实时库存或某些精确具体的商品型号信息）。"
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
            answer = best_match.response
            logger.info(f"   ✅ 缓存命中 ({cache_confidence:.3f}): '{query}' -> 匹配到了 '{cache_matched_question}'")
        else:  # 未命中
            cache_hit = False
            cache_matched_question = None
            cache_confidence = 0.0
            cache_seed_id = None
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
        "execution_path": state["execution_path"] + ["cache_checked"], # 更新执行路径轨迹
        "metrics": metrics,
    }

# ---------------------------------------------------------------------------
# 缓存语义复用裁判（LLM Reranker）
# ---------------------------------------------------------------------------
class RerankerEvaluation(BaseModel):
    """LLM Reranker 的结构化输出。"""
    can_reuse_answer: bool = Field(description="缓存答案能否直接用于回答用户的新问题")
    score: float = Field(description="对该判定的置信度，范围 0.0 到 1.0")
    reason: str = Field(description="一句极短中文理由，20字以内")

def _clip_rerank_answer(answer: str, max_chars: int = RERANK_ANSWER_CHAR_LIMIT) -> str:
    """压缩缓存答案，避免 Reranker 因上下文过长而耗尽输出预算。"""
    compact = " ".join(answer.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "..."

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
            "cache_rerank_score": 0.0,
            "cache_rerank_reason": "no_cache_candidate",
            "execution_path": state.get("execution_path", []) + ["cache_rerank_skipped"],
        }

    logger.info(f"⚖️ 启动缓存复用裁判: '{query[:20]}...' vs 缓存问题 '{cached_question[:20]}...'")

    llm_calls = state.get("llm_calls", {}).copy()
    cached_answer_excerpt = _clip_rerank_answer(cached_answer)
    try:
        structured_llm = get_analysis_llm().with_structured_output(RerankerEvaluation)
        result = structured_llm.invoke([
            SystemMessage(content=RERANK_SYSTEM_PROMPT),
            HumanMessage(content=RERANK_PROMPT.format(
                query=query,
                cached_question=cached_question,
                cached_answer_excerpt=cached_answer_excerpt,
            )),
        ])
        passed = bool(result.can_reuse_answer)
        score = max(0.0, min(1.0, float(result.score)))
        reason = result.reason or ""
    except Exception as e:
        logger.exception(f"   ❌ Reranker 调用失败，安全降级: {e}")
        passed = False
        score = 0.0
        reason = f"rerank_exception: {e}"

    llm_calls["analysis_llm"] = llm_calls.get("analysis_llm", 0) + 1
    rerank_time = (time.perf_counter() - start_time) * 1000
    metrics = update_metrics(state.get("metrics", initialize_metrics()), rerank_latency=rerank_time)

    if passed:
        logger.info(f"   ✅ Reranker 通过 (score={score:.2f}): {reason}")
        return {
            **state,
            "cache_rerank_passed": True,
            "cache_rerank_score": score,
            "cache_rerank_reason": reason,
            "execution_path": state.get("execution_path", []) + ["cache_reranked_passed"],
            "llm_calls": llm_calls,
            "metrics": metrics,
        }
    else:
        logger.info(f"   ⛔ Reranker 拒绝 (score={score:.2f}): {reason}，转入 RAG")
        # 等价为缓存未命中：清空答案并转入 research，后续会把研究结果作为新答案处理。
        return {
            **state,
            "answer": "",
            "cache_hit": False,
            "cache_rerank_passed": False,
            "cache_rerank_score": score,
            "cache_rerank_reason": reason,
            "execution_path": state.get("execution_path", []) + ["cache_reranked_failed"],
            "llm_calls": llm_calls,
            "metrics": metrics,
        }

def research_node(state: WorkflowState) -> WorkflowState:
    """节点：执行深度研究/知识库检索"""
    start_time = time.perf_counter()  # 记录节点起始时间
    query = state["query"]            # 用户提问
    iteration = state.get("research_iterations", 0) + 1  # 轮次加 1
    
    logger.info(f"🔍 正在研究: '{query}'")
    
    tools = [search_knowledge_base]  # 定义可用工具列表
    research_llm_invocations = 0

    # 本项目固定为单轮研究，不再根据评分反馈重试。
    research_prompt = RESEARCH_PROMPT_INITIAL.format(query=query)

    # 绑定工具到 LLM
    llm_with_tools = get_research_llm().bind_tools(tools)
    llm_base = get_research_llm()  # 获取原始模型用于最终生成
    messages = [
        SystemMessage(content=SYSTEM_PROMPT), 
        HumanMessage(content=research_prompt)
    ] # 初始化对话列表
    
    # --- 模拟 ReAct 循环：最多允许 3 次工具调用交互 ---
    for _ in range(3):
        response = llm_with_tools.invoke(messages) # 调用 LLM 询问是否需要用工具
        research_llm_invocations += 1
        messages.append(response)
        if not response.tool_calls:                # 如果 LLM 不再需要调用工具，跳出循环
            break
        for tool_call in response.tool_calls:      # 遍历 LLM 提出的工具调用请求
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            try:
                logger.info(f"   🔧 执行工具: {tool_name} {tool_args}")
                # 执行检索工具
                tool_result = tools[0].invoke(tool_args) 
            except Exception as e:
                tool_result = f"Error executing tool: {e}"
            from langchain_core.messages import ToolMessage
            # 将工具执行结果存入对话历史
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))
            
    from langchain_core.messages import ToolMessage
    # 容错处理：如果循环结束时最后一条是工具结果，补充一步让 LLM 生成自然语言摘要
    if isinstance(messages[-1], ToolMessage) or (hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls):
        messages.append(HumanMessage(content="检索已经结束，请根据以上的检索结果，用自然流利的语言直接给出答案，不要列出原始段落结构。"))
        response = llm_base.invoke(messages)
        research_llm_invocations += 1
        messages.append(response)

    # 提取对话历史中的最终文本答案
    final_answer = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
    
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
        and "researched" in state.get("execution_path", [])
    ):
        logger.info(f"   💾 将研究得到的回答写入语义缓存: '{state['query'][:20]}...'")
        _cache_instance.cache.store(prompt=state["query"], response=state["answer"])

        # 手动维护内存字典供 fuzzy_matches 短路使用
        if not hasattr(_cache_instance, '_seed_id_by_question'):
            _cache_instance._seed_id_by_question = {}
        if not hasattr(_cache_instance, '_answer_by_question'):
            _cache_instance._answer_by_question = {}

        _cache_instance._seed_id_by_question[state["query"]] = None
        _cache_instance._answer_by_question[state["query"]] = state["answer"]
    
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