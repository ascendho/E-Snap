import logging  # 导入日志模块，记录程序运行状态
import os       # 导入操作系统接口，用于读取环境变量
import time     # 导入时间模块，用于性能耗时统计
from datetime import datetime  # 导入日期时间模块
from typing import Any, Dict, List, Optional, TypedDict  # 导入类型提示，增强代码可读性和健壮性

from langchain_core.messages import HumanMessage  # 导入 LangChain 的人类消息对象
from langchain_openai import ChatOpenAI           # 导入 LangChain 的 OpenAI 兼容模型接口
from workflow.tools import search_knowledge_base  # 导入自定义的知识库检索工具
from common.env import CACHE_DISTANCE_THRESHOLD, ARK_BASE_URL, ANALYSIS_MODEL_NAME, RESEARCH_MODEL_NAME

# 获取名为 "agentic-workflow" 的日志记录器
logger = logging.getLogger("agentic-workflow")

# 全局变量占位符，用于实现 LLM 实例的单例模式（懒加载）
_analysis_llm = None
_research_llm = None

def get_analysis_llm():
    """获取用于逻辑分析和质量评估的 LLM 实例（低随机性，重逻辑）"""
    global _analysis_llm
    # 如果尚未初始化
    if _analysis_llm is None:  
        _analysis_llm = ChatOpenAI(
            model=ANALYSIS_MODEL_NAME,
            temperature=0.1,                                     # 设置低随机性，确保评估结果稳定
            max_tokens=400,                                      # 限制输出长度
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
    research_latency: float         # 知识检索耗时
    synthesis_latency: float        # 回答合成耗时
    evaluation_latency: float       # 质量评估耗时
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
    research_iterations: int            # 当前研究的迭代轮次
    max_research_iterations: int        # 允许的最大研究轮次
    research_quality_score: float       # 研究结果的质量得分
    research_feedback: str              # 评估节点给出的改进反馈建议
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
        "research_latency": 0.0,
        "synthesis_latency": 0.0,
        "evaluation_latency": 0.0,
        "cache_hit_rate": 0.0,
        "cache_hits_count": 0,
        "total_research_iterations": 0,
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

def initialize_nodes(semantic_cache):
    """初始化节点，注入语义缓存实例"""
    global _cache_instance
    _cache_instance = semantic_cache

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

def research_node(state: WorkflowState) -> WorkflowState:
    """节点：执行深度研究/知识库检索"""
    start_time = time.perf_counter()  # 记录节点起始时间
    query = state["query"]            # 用户提问
    iteration = state.get("research_iterations", 0) + 1  # 轮次加 1
    feedback = state.get("research_feedback", "")        # 获取上一轮评估的改进意见
    
    logger.info(f"🔍 正在研究: '{query}' (轮次 {iteration})")
    
    tools = [search_knowledge_base]  # 定义可用工具列表
    
    # 如果有反馈信息，则构建一个带引导的 Prompt，要求 LLM 调整策略
    if feedback:
        from workflow.prompts import RESEARCH_PROMPT_WITH_FEEDBACK
        research_prompt = RESEARCH_PROMPT_WITH_FEEDBACK.format(query=query, feedback=feedback)
    else:
        # 初次尝试的 Prompt
        from workflow.prompts import RESEARCH_PROMPT_INITIAL
        research_prompt = RESEARCH_PROMPT_INITIAL.format(query=query)

    # 绑定工具到 LLM
    llm_with_tools = get_research_llm().bind_tools(tools)
    llm_base = get_research_llm()  # 获取原始模型用于最终生成
    messages = [HumanMessage(content=research_prompt)] # 初始化对话列表
    
    # --- 模拟 ReAct 循环：最多允许 3 次工具调用交互 ---
    for _ in range(3):
        response = llm_with_tools.invoke(messages) # 调用 LLM 询问是否需要用工具
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
            
    from langchain_core.messages import ToolMessage, AIMessage
    # 容错处理：如果循环结束时最后一条是工具结果，补充一步让 LLM 生成自然语言摘要
    if isinstance(messages[-1], ToolMessage) or (hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls):
        messages.append(HumanMessage(content="检索已经结束，请根据以上的检索结果，用自然流利的语言直接给出答案，不要列出原始段落结构。"))
        response = llm_base.invoke(messages)
        messages.append(response)

    # 提取对话历史中的最终文本答案
    final_answer = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
    
    # 更新 LLM 调用统计
    llm_calls = state.get("llm_calls", {}).copy()
    llm_calls["research_llm"] = llm_calls.get("research_llm", 0) + 1
    
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

from pydantic import BaseModel, Field

class QualityEvaluation(BaseModel):
    score: float = Field(description="研究结果质量分数，从 0.0 到 1.0")
    feedback: str = Field(description="如果分数低于0.7，给出如何改进搜索策略的建议，否则写'OK'")

def evaluate_quality_node(state: WorkflowState) -> WorkflowState:
    """节点：对研究结果进行质量评估"""
    start_time = time.perf_counter()  # 记录开始时间
    query = state["query"]            # 用户提问
    answer = state["answer"]          # 检索出来的答案
    
    # 构建评估 Prompt，要求输出特定的结构化格式
    from workflow.prompts import EVAL_PROMPT
    eval_prompt = EVAL_PROMPT.format(query=query, answer=answer)
    
    try:
        # 调用分析专用 LLM，使用 structured output 强类型约束
        structured_llm = get_analysis_llm().with_structured_output(QualityEvaluation)
        result = structured_llm.invoke([HumanMessage(content=eval_prompt)])
        score = result.score
        feedback = result.feedback
    except Exception as e:
        logger.error(f"结构化解析或大模型调用失败: {e}")
        # 解析失败时的兜底策略 (安全悲观降级)
        score = 0.0
        feedback = "解析异常，需要重试"
        
    logger.info(f"   ✅ '{query[:20]}...' - 质量评估分数: {score:.2f} ({'达标' if score>=0.7 else '未达标'})")

    # 更新调用次数和性能指标
    llm_calls = state.get("llm_calls", {}).copy()
    llm_calls["analysis_llm"] = llm_calls.get("analysis_llm", 0) + 1
    
    eval_time = (time.perf_counter() - start_time) * 1000
    metrics = update_metrics(state.get("metrics", initialize_metrics()), evaluation_latency=eval_time)
    
    # 返回评估结果存入状态
    return {
        **state,
        "research_quality_score": score,
        "research_feedback": feedback,
        "execution_path": state["execution_path"] + ["quality_evaluated"],
        "llm_calls": llm_calls,
        "metrics": metrics,
    }

def synthesize_response_node(state: WorkflowState) -> WorkflowState:
    """节点：合成最终用户响应（并执行缓存写回）"""
    start_time = time.perf_counter()  # 记录开始时间
    
    llm_calls = state.get("llm_calls", {}).copy()
    
    # 拼接最终呈现给用户的文案格式
    final_response = f"针对您的问题：“{state['query']}”\n\n解答如下：\n{state['answer']}"
    
    # --- 【自学习逻辑】：将高质量研究结果回填至语义缓存 ---
    # 只有当原本未命中缓存，且当前研究结果质量达标时，才执行存入 Redis 操作
    if not state.get("cache_hit", False) and _cache_instance:
        if state.get("research_quality_score", 1.0) >= 0.7:
            logger.info(f"   💾 将高质量的回答写入语义缓存: '{state['query'][:20]}...'")
            # 下次如果有类似问题，就能直接从缓存拿，不再消耗大模型 token
            _cache_instance.cache.store(prompt=state["query"], response=state["answer"])
    
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