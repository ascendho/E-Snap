import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from agent.tools import search_knowledge_base

logger = logging.getLogger("agentic-workflow")

_analysis_llm = None
_research_llm = None

def get_analysis_llm():
    global _analysis_llm
    if _analysis_llm is None:
        _analysis_llm = ChatOpenAI(
            model="ep-m-20260411093114-9hftc",
            temperature=0.1,
            max_tokens=400,
            api_key=os.getenv("ARK_API_KEY"),
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )
    return _analysis_llm

def get_research_llm():
    global _research_llm
    if _research_llm is None:
        _research_llm = ChatOpenAI(
            model="deepseek-v3-2-251201",
            temperature=0.2,
            max_tokens=400,
            api_key=os.getenv("ARK_API_KEY"),
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )
    return _research_llm

class WorkflowMetrics(TypedDict):
    total_latency: float
    cache_latency: float
    research_latency: float
    synthesis_latency: float
    evaluation_latency: float
    cache_hit_rate: float
    cache_hits_count: int
    total_research_iterations: int

class WorkflowState(TypedDict):
    query: str
    answer: str
    final_response: Optional[str]
    cache_hit: bool
    cache_matched_question: Optional[str]
    cache_confidence: float
    cache_seed_id: Optional[int]
    cache_enabled: bool
    research_iterations: int
    max_research_iterations: int
    research_quality_score: float
    research_feedback: str
    current_research_strategy: str
    execution_path: List[str]
    metrics: WorkflowMetrics
    timestamp: str
    llm_calls: Dict[str, int]

def initialize_metrics() -> WorkflowMetrics:
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
    new_metrics = metrics.copy()
    for key, value in kwargs.items():
        if key in new_metrics and isinstance(new_metrics[key], (int, float)):
            new_metrics[key] += value
        else:
            new_metrics[key] = value
    return new_metrics

_cache_instance = None

def initialize_nodes(semantic_cache):
    global _cache_instance
    _cache_instance = semantic_cache

def check_cache_node(state: WorkflowState) -> WorkflowState:
    start_time = time.perf_counter()
    query = state["query"]
    
    logger.info(f"🔍 开始检查语义缓存: '{query}'")
    
    if not state.get("cache_enabled", True) or not _cache_instance:
        logger.info("   ⚠️ 缓存未启用或未初始化")
        cache_hit = False
        cache_matched_question = None
        cache_confidence = 0.0
        cache_seed_id = None
        answer = ""
    else:
        results = _cache_instance.check(query, distance_threshold=0.2)
        if results.matches:
            best_match = results.matches[0]
            cache_hit = True
            cache_matched_question = best_match.prompt
            cache_confidence = best_match.cosine_similarity
            cache_seed_id = best_match.seed_id
            answer = best_match.response
            logger.info(f"   ✅ 缓存命中 ({cache_confidence:.3f}): '{query}' -> 匹配到了 '{cache_matched_question}'")
        else:
            cache_hit = False
            cache_matched_question = None
            cache_confidence = 0.0
            cache_seed_id = None
            answer = ""
            logger.info(f"   ❌ 缓存未命中: '{query}'")

    cache_time = (time.perf_counter() - start_time) * 1000
    
    metrics = state.get("metrics", initialize_metrics())
    metrics = update_metrics(
        metrics,
        cache_latency=cache_time,
        cache_hits_count=1 if cache_hit else 0,
        cache_hit_rate=1.0 if cache_hit else 0.0
    )
    
    return {
        **state,
        "answer": answer,
        "cache_hit": cache_hit,
        "cache_matched_question": cache_matched_question,
        "cache_confidence": cache_confidence,
        "cache_seed_id": cache_seed_id,
        "execution_path": state["execution_path"] + ["cache_checked"],
        "metrics": metrics,
    }

def research_node(state: WorkflowState) -> WorkflowState:
    start_time = time.perf_counter()
    query = state["query"]
    iteration = state.get("research_iterations", 0) + 1
    feedback = state.get("research_feedback", "")
    
    logger.info(f"🔍 正在研究: '{query}' (轮次 {iteration})")
    
    tools = [search_knowledge_base]
    if feedback:
        research_prompt = f"""
请针对以下问题进行检索和回答：
问题：{query}

之前尝试检索失败或效果不佳，收到了以下反馈：
{feedback}

请调整你的搜索关键词或策略，如果需要请多次分步骤调用重试，利用知识库尽力搜寻信息，并给出完整准确的答案。
如果确实找不到相关规定，就回答“目前知识库中没有记录关于此问题的具体规定”。
        """
    else:
        research_prompt = f"""
请针对以下问题进行研究：
问题：{query}

请使用提供的检索工具从知识库查找信息并给出准确完整的答案。如果相关政策有前提条件或特例，请一并说明。
        """

    llm_with_tools = get_research_llm().bind_tools(tools)
    llm_base = get_research_llm()
    messages = [HumanMessage(content=research_prompt)]
    
    # 模拟简单的 Tool Call 循环
    for _ in range(3):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            try:
                logger.info(f"   🔧 执行工具: {tool_name} {tool_args}")
                tool_result = tools[0].invoke(tool_args)
            except Exception as e:
                tool_result = f"Error executing tool: {e}"
            from langchain_core.messages import ToolMessage
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))
            
    from langchain_core.messages import ToolMessage, AIMessage
    # 如果最后一条消息仍是工具返回的结果（即因为达到了最大循环次数，LLM还没来得及给出自然语言回答）
    if isinstance(messages[-1], ToolMessage) or (hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls):
        messages.append(HumanMessage(content="检索已经结束，请根据以上的检索结果，用自然流利的语言直接给出答案，不要列出原始段落结构。"))
        response = llm_base.invoke(messages)
        messages.append(response)

    final_answer = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
    
    llm_calls = state.get("llm_calls", {}).copy()
    llm_calls["research_llm"] = llm_calls.get("research_llm", 0) + 1
    
    research_time = (time.perf_counter() - start_time) * 1000
    metrics = update_metrics(state.get("metrics", initialize_metrics()), research_latency=research_time, total_research_iterations=1)
    
    return {
        **state,
        "answer": final_answer,
        "research_iterations": iteration,
        "execution_path": state["execution_path"] + ["researched"],
        "llm_calls": llm_calls,
        "metrics": metrics,
    }

def evaluate_quality_node(state: WorkflowState) -> WorkflowState:
    start_time = time.perf_counter()
    query = state["query"]
    answer = state["answer"]
    
    eval_prompt = f"""
请评估以下回答是否充分解答了问题。
如果知识库明确涵盖了这个问题并给出了答案，那么是合格的。
如果回答是"我在知识库里没找到"，也是合格的（说明确实没信息，无需再找）。

问题：{query}
生成的回答：{answer}

请严格按如下格式输出单行结果：
SCORE: [0.0-1.0的分数]
FEEDBACK: [如果分数低于0.7，给出如何改进搜索策略的建议，否则写OK]
"""
    
    response = get_analysis_llm().invoke([HumanMessage(content=eval_prompt)])
    content = response.content.strip()
    
    score = 1.0
    feedback = ""
    try:
        lines = content.split('\n')
        for line in lines:
            if line.startswith("SCORE:"):
                score = float(line.split("SCORE:")[1].strip())
            elif line.startswith("FEEDBACK:"):
                feedback = line.split("FEEDBACK:")[1].strip()
    except Exception:
        score = 0.8
        feedback = "解析异常，假设合格"
        
    logger.info(f"   ✅ '{query[:20]}...' - 质量评估分数: {score:.2f} ({'达标' if score>=0.7 else '未达标'})")

    llm_calls = state.get("llm_calls", {}).copy()
    llm_calls["analysis_llm"] = llm_calls.get("analysis_llm", 0) + 1
    
    eval_time = (time.perf_counter() - start_time) * 1000
    metrics = update_metrics(state.get("metrics", initialize_metrics()), evaluation_latency=eval_time)
    
    return {
        **state,
        "research_quality_score": score,
        "research_feedback": feedback,
        "execution_path": state["execution_path"] + ["quality_evaluated"],
        "llm_calls": llm_calls,
        "metrics": metrics,
    }

def synthesize_response_node(state: WorkflowState) -> WorkflowState:
    start_time = time.perf_counter()
    
    llm_calls = state.get("llm_calls", {}).copy()
    
    final_response = f"针对您的问题：“{state['query']}”\n\n解答如下：\n{state['answer']}"
    
    # 【非常关键的遗漏修复】：把研究出来的好答案存回缓存！否则只有首批自带的 FAQ 起作用
    if not state.get("cache_hit", False) and _cache_instance:
        # 只缓存通过了评估(达到及格线)的靠谱答案，避免脏数据污染
        if state.get("research_quality_score", 1.0) >= 0.7:
            logger.info(f"   💾 将高质量的回答写入语义缓存: '{state['query'][:20]}...'")
            _cache_instance.cache.store(prompt=state["query"], response=state["answer"])
    
    synth_time = (time.perf_counter() - start_time) * 1000
    metrics = update_metrics(state.get("metrics", initialize_metrics()), synthesis_latency=synth_time)
    
    return {
        **state,
        "final_response": final_response,
        "execution_path": state["execution_path"] + ["synthesized"],
        "llm_calls": llm_calls,
        "metrics": metrics,
    }
