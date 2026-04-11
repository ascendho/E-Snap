"""
Deep research Agent 工作流包。

该包提供了带语义缓存的完整 Agent 工作流实现，包含查询拆解、研究检索、
质量评估与条件路由。
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any

# 配置日志记录器
logger = logging.getLogger("agentic-workflow")

# 从子模块导入核心组件
from .nodes import (
    WorkflowState,
    initialize_nodes,
    initialize_metrics,
    decompose_question_node,
    decompose_query_node,
    check_cache_node,
    synthesize_response_node,
    evaluate_quality_node,
    research_node,
)

from .edges import (
    initialize_edges,
    route_after_cache_check,
    route_after_quality_evaluation,
    cache_validated_research,
)

from .tools import (
    initialize_tools,
    search_knowledge_base,
)

from .demo import (
    create_demo,
    launch_demo,
    ResearchDemo,
)

from .knowledge_base_utils import (
    KnowledgeBaseManager,
    create_knowledge_base_from_texts,
)


def initialize_agent(semantic_cache, knowledge_base_index, openai_embeddings):
    """
    初始化 Agent 所需的核心组件与依赖。

    Args:
        semantic_cache: 语义缓存实例，用于子问题级别缓存命中判断。
        knowledge_base_index: 知识库对应的 Redis 向量检索索引名。
        openai_embeddings: 向量嵌入模型实例，用于查询向量化。
    """
    # 初始化节点、边以及检索工具，并注入共享的依赖
    initialize_nodes(semantic_cache)
    initialize_edges(semantic_cache)
    initialize_tools(knowledge_base_index, openai_embeddings)


def run_agent(agent, query: str, enable_caching: bool = True) -> Dict[str, Any]:
    """
    执行一次查询并返回完整监控信息（路径、时延、缓存命中等）。
    
    Args:
        agent: 已编译的工作流对象（通常是 LangGraph 实例）。
        query: 用户的原始输入问题。
        enable_caching: 是否启用语义缓存功能。
        
    Returns:
        包含执行结果、性能指标和执行路径的字典。
    """
    start_time = time.perf_counter()

    # 初始化工作流状态（State）：这是整个 Graph 执行过程中流转的数据结构
    initial_state: WorkflowState = {
        "original_query": query,               # 原始问题
        "sub_questions": [],                  # 拆解后的子问题列表
        "sub_answers": {},                    # 子问题的答案存储
        "cache_hits": {},                     # 每个子问题的缓存命中情况
        "cache_confidences": {},              # 缓存命中的置信度
        "cache_enabled": enable_caching,      # 是否启用缓存
        "research_iterations": {},            # 记录子问题的研究迭代次数
        "max_research_iterations": 2,         # 最大迭代次数，防止死循环
        "research_quality_scores": {},        # 检索质量评分
        "research_feedback": {},              # 针对质量不合格的反馈建议
        "current_research_strategy": {},      # 当前采用的搜索/研究策略
        "final_response": None,               # 最终生成的汇总回答
        "execution_path": [],                 # 记录节点执行顺序的路径
        "active_sub_question": None,          # 当前正在处理的子问题
        "metrics": initialize_metrics(),      # 初始化性能度量指标
        "timestamp": datetime.now().isoformat(),
        "comparison_mode": False,             # 是否对比模式
        "llm_calls": {},                      # 统计不同 LLM 的调用次数
        "llm_usage": [],                      # 记录每次 LLM 调用 token 元数据
    }

    logger.info("=" * 80)

    try:
        # 执行工作流：agent.invoke 会根据定义好的节点和边进行逻辑流转
        final_state = agent.invoke(initial_state)

        # 计算总时延（毫秒）
        total_latency = (time.perf_counter() - start_time) * 1000
        
        # 整理最终指标
        final_metrics = final_state["metrics"].copy()
        final_metrics["total_latency"] = total_latency
        final_metrics["execution_path"] = " → ".join(final_state["execution_path"])

        # 封装最终返回给前端/用户的 JSON 结果
        result = {
            "original_query": query,
            "sub_questions": final_state["sub_questions"],
            "sub_answers": final_state["sub_answers"],
            "cache_hits": final_state["cache_hits"],
            "cache_confidences": final_state["cache_confidences"],
            "final_response": final_state["final_response"],
            "execution_path": final_metrics["execution_path"],
            "total_latency": f"{total_latency:.2f}ms",
            "metrics": final_metrics,
            "timestamp": final_state["timestamp"],
            "llm_calls": final_state.get("llm_calls", {}),
            "llm_usage": final_state.get("llm_usage", []),
        }

        logger.info("=" * 80)
        logger.info(f"✅ 工作流执行完成，总耗时 {total_latency:.2f}ms")

        return result

    except Exception as e:
        logger.error(f"❌ 工作流执行失败: {e}")
        return {
            "original_query": query,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def display_results(result):
    """
    在 Notebook 或终端中格式化展示一次请求的执行路径、缓存状态与最终回答。
    """
    from IPython.display import display, Markdown

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    print(f"\\n🎯 语义缓存工作流分析")
    print("=" * 80)

    # 展示原始问题
    print(
        f"📝 **原始问题:** {result['original_query'][:100]}{'...' if len(result['original_query']) > 100 else ''}"
    )
    # 展示节点流转路径
    execution_path = result.get("execution_path", "")
    print(f"🔄 **执行路径:** {execution_path}")

    sub_questions = result.get("sub_questions", [])
    cache_hits = result.get("cache_hits", {})
    cache_confidences = result.get("cache_confidences", {})

    # 展示子问题的拆解与缓存命中细节
    if sub_questions:
        print(f"\\n🧠 **问题拆解:** 共 {len(sub_questions)} 个子问题")
        for i, sq in enumerate(sub_questions, 1):
            hit_status = cache_hits.get(sq, False)
            confidence = cache_confidences.get(sq, 0.0)

            if hit_status:
                print(f"   {i}. ✅ **缓存命中** ({confidence:.3f}): {sq}")
            else:
                print(f"   {i}. 🔍 **需要检索**: {sq}")

    print(f"\\n📊 **性能指标:**")

    # 计算并展示缓存命中率
    if cache_hits:
        hits = sum(1 for hit in cache_hits.values() if hit)
        hit_rate = hits / len(cache_hits) if cache_hits else 0
        print(
            f"   💾 缓存命中率: **{hit_rate:.1%}** ({hits}/{len(cache_hits)} 个子问题)"
        )

    # 展示不同模型的调用统计
    llm_calls = result.get("llm_calls", {})
    if llm_calls:
        analysis_calls = llm_calls.get("analysis_llm", 0)
        research_calls = llm_calls.get("research_llm", 0)
        total_calls = analysis_calls + research_calls
        print(
            f"   🤖 LLM 调用次数: **{total_calls}** (DeepSeek-V3: {analysis_calls}, Doubao-Lite: {research_calls})"
        )

    # 基于每次调用的 token 元数据，展示本次请求的费用摘要。
    llm_usage = result.get("llm_usage", [])
    if llm_usage:
        try:
            from cache.evals import PerfEval, format_cost

            perf = PerfEval()
            for call in llm_usage:
                perf.record_llm_call(
                    model=call.get("model", "unknown-model"),
                    provider=call.get("provider", "openai"),
                    input_tokens=int(call.get("input_tokens", 0) or 0),
                    output_tokens=int(call.get("output_tokens", 0) or 0),
                )
            perf.set_total_queries(1)
            costs = perf.get_costs()

            total_in = sum(int(c.get("input_tokens", 0) or 0) for c in llm_usage)
            total_out = sum(int(c.get("output_tokens", 0) or 0) for c in llm_usage)

            print(
                f"   💰 本次成本: **{format_cost(costs.get('total_cost', 0.0), costs.get('currency', 'CNY'))}** "
                f"(币种: {costs.get('currency', 'CNY')}, in/out tokens: {total_in}/{total_out})"
            )
        except Exception as e:
            # 成本展示失败不影响主流程结果展示。
            print(f"   💰 本次成本: 无法计算（{e}）")

    # 细化展示各阶段耗时
    metrics = result.get("metrics", {})
    total_latency = result.get("total_latency", "0ms")
    print(f"   ⚡ 总时延: **{total_latency}**")

    if metrics:
        cache_latency = metrics.get("cache_latency", 0)
        research_latency = metrics.get("research_latency", 0)
        print(f"   ⏱️  缓存阶段: {cache_latency:.0f}ms, 检索阶段: {research_latency:.0f}ms")

    # 使用 Markdown 渲染最终答案
    final_response = result.get("final_response", "")
    if final_response:
        print(f"\\n📋 **最终回答:**")
        print("-" * 80)
        display(Markdown(final_response))
        print("-" * 80)

    print("=" * 80)


def analyze_agent_results(results):
    """
    汇总多轮场景（Scenario）执行结果，并生成缓存效果与耗时的可视化图表。
    用于展示随着对话进行，语义缓存如何显著降低耗时和成本。
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.display import display
    
    # 定义模拟的三个阶段场景
    scenario_names = [
        "Enterprise Evaluation", # 企业评估阶段
        "Implementation Planning", # 实施规划阶段
        "Final Review",            # 最终审查阶段
    ]

    scenario_data = []
    cumulative_questions = 0
    cumulative_hits = 0

    # 提取多轮测试的数据
    for i, (result, name) in enumerate(zip(results, scenario_names), 1):
        cache_hits = result.get("cache_hits", {})
        sub_questions = result.get("sub_questions", [])
        llm_calls = result.get("llm_calls", {})
        metrics = result.get("metrics", {})
        
        hits = sum(1 for hit in cache_hits.values() if hit)
        total_qs = len(sub_questions)
        hit_rate = (hits / total_qs * 100) if total_qs > 0 else 0

        # 计算累积命中率
        cumulative_questions += total_qs
        cumulative_hits += hits
        cumulative_rate = (
            (cumulative_hits / cumulative_questions * 100)
            if cumulative_questions > 0
            else 0
        )

        analysis_calls = llm_calls.get("analysis_llm", 0)
        research_calls = llm_calls.get("research_llm", 0)
        total_calls = analysis_calls + research_calls

        # 解析时延字符串
        total_latency_str = result.get("total_latency", "0ms")
        total_latency = float(total_latency_str.replace("ms", "")) if isinstance(total_latency_str, str) else total_latency_str
        cache_latency = metrics.get("cache_latency", 0)
        research_latency = metrics.get("research_latency", 0)

        scenario_data.append({
            'name': name,
            'scenario_num': i,
            'total_questions': total_qs,
            'cache_hits': hits,
            'hit_rate': hit_rate,
            'cumulative_hit_rate': cumulative_rate,
            'total_llm_calls': total_calls,
            'analysis_calls': analysis_calls,
            'research_calls': research_calls,
            'total_latency': total_latency,
            'cache_latency': cache_latency,
            'research_latency': research_latency,
        })

    # --- 开始绘图 ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("🎯 Semantic Caching Performance Analysis", fontsize=16, fontweight="bold", y=0.98)

    scenarios = [d['scenario_num'] for d in scenario_data]
    hit_rates = [d['hit_rate'] for d in scenario_data]
    cumulative_rates = [d['cumulative_hit_rate'] for d in scenario_data]
    
    # 图表 1: 缓存命中率演进（柱状图 + 趋势线）
    ax1.bar(scenarios, hit_rates, alpha=0.7, color='skyblue', label='Per-Scenario Hit Rate')
    ax1.plot(scenarios, cumulative_rates, marker='o', color='darkblue', linewidth=2, label='Cumulative Hit Rate')
    ax1.set_title("💾 Cache Hit Rate Evolution", fontweight="bold")
    ax1.set_xlabel("Scenario")
    ax1.set_ylabel("Hit Rate (%)")
    ax1.set_xticks(scenarios)
    ax1.set_xticklabels([f"S{i}" for i in scenarios])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for i, (rate, cum_rate) in enumerate(zip(hit_rates, cumulative_rates)):
        ax1.text(scenarios[i], rate + 2, f'{rate:.0f}%', ha='center', fontsize=9)
        ax1.text(scenarios[i], cum_rate + 5, f'{cum_rate:.1f}%', ha='center', fontsize=9, color='darkblue')

    # 图表 2: LLM 调用分类统计
    analysis_calls = [d['analysis_calls'] for d in scenario_data]
    research_calls = [d['research_calls'] for d in scenario_data]
    
    width = 0.35
    x_pos = np.array(scenarios)
    ax2.bar(x_pos - width/2, analysis_calls, width, label='DeepSeek-V3 (Analysis)', color='lightcoral')
    ax2.bar(x_pos + width/2, research_calls, width, label='Doubao-Lite (Research)', color='lightgreen')
    
    ax2.set_title("🤖 LLM Calls by Type", fontweight="bold")
    ax2.set_xlabel("Scenario")
    ax2.set_ylabel("Number of Calls")
    ax2.set_xticks(scenarios)
    ax2.set_xticklabels([f"S{i}" for i in scenarios])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    for i, (a_calls, r_calls) in enumerate(zip(analysis_calls, research_calls)):
        ax2.text(scenarios[i] - width/2, a_calls + 0.1, str(a_calls), ha='center', fontsize=9)
        ax2.text(scenarios[i] + width/2, r_calls + 0.1, str(r_calls), ha='center', fontsize=9)

    # 图表 3: 耗时分解（对数坐标，用于处理缓存和检索之间的巨大差异）
    cache_latencies = [d['cache_latency'] for d in scenario_data]
    research_latencies = [d['research_latency'] for d in scenario_data]
    
    ax3.bar(x_pos - width/2, cache_latencies, width, label='Cache Latency', color='gold', alpha=0.8)
    ax3.bar(x_pos + width/2, research_latencies, width, label='Research Latency', color='orange', alpha=0.8)
    ax3.set_yscale('log') # 使用对数刻度
    
    ax3.set_title("⚡ Latency Breakdown (Log Scale)", fontweight="bold")
    ax3.set_xlabel("Scenario")
    ax3.set_ylabel("Latency (ms) - Log Scale")
    ax3.set_xticks(scenarios)
    ax3.set_xticklabels([f"S{i}" for i in scenarios])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    for i, (cache_lat, research_lat) in enumerate(zip(cache_latencies, research_latencies)):
        cache_text_y = cache_lat * 1.5 if cache_lat > 0 else 0.1
        ax3.text(scenarios[i] - width/2, cache_text_y, f'{cache_lat:.0f}ms', ha='center', fontsize=9, rotation=90)
        
        research_text_y = research_lat * 1.2 if research_lat > 0 else 0.1
        ax3.text(scenarios[i] + width/2, research_text_y, f'{research_lat:.0f}ms', ha='center', fontsize=9)

    # 图表 4: 文本总结框
    ax4.axis('off') # 隐藏坐标轴，仅显示文本
    
    total_questions_all = sum(d['total_questions'] for d in scenario_data)
    total_hits_all = sum(d['cache_hits'] for d in scenario_data)
    overall_hit_rate = (total_hits_all / total_questions_all * 100) if total_questions_all > 0 else 0
    total_llm_calls_all = sum(d['total_llm_calls'] for d in scenario_data)
    avg_latency = np.mean([d['total_latency'] for d in scenario_data])
    
    summary_text = f"""
📊 OVERALL PERFORMANCE SUMMARY

📝 Total Questions Processed: {total_questions_all}
💾 Total Cache Hits: {total_hits_all} ({overall_hit_rate:.1f}%)
🤖 Total LLM Calls: {total_llm_calls_all}
⚡ Average Latency: {avg_latency:.0f}ms

🚀 Research Calls Avoided: {total_hits_all}
🎯 Key Insight: Cache effectiveness increases with each interaction!
✅ Result: Semantic caching delivers progressive intelligence and cost savings

💡 Cache Evolution:
  • Scenario 1: {scenario_data[0]['hit_rate']:.0f}% hit rate
  • Scenario 2: {scenario_data[1]['hit_rate']:.0f}% hit rate  
  • Scenario 3: {scenario_data[2]['hit_rate']:.0f}% hit rate
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 自动保存分析结果图表
    import os
    os.makedirs("outputs", exist_ok=True)
    save_path = "outputs/performance_analysis.png"
    plt.savefig(save_path, dpi=300)
    logger.info(f"📊 可视化图已保存到: {save_path}")

    return cumulative_questions, cumulative_hits

# 定义包暴露的公开接口
__all__ = [
    "WorkflowState",
    "initialize_agent",
    "run_agent",
    "analyze_agent_results",
    "decompose_question_node",
    "check_cache_node",
    "synthesize_response_node",
    "evaluate_quality_node",
    "research_node",
    "process_query_node",
    "route_after_cache_check",
    "route_after_quality_evaluation",
    "cache_validated_research",
    "search_knowledge_base",
    "create_demo",
    "launch_demo",
    "ResearchDemo",
    "KnowledgeBaseManager", 
    "create_knowledge_base_from_texts",
]