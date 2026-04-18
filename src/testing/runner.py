from typing import Dict, Any
def run_agent(workflow_app, query: str) -> Dict[str, Any]:
    from workflow.nodes import initialize_metrics
    import time
    from datetime import datetime
    
    start_time = time.perf_counter()
    initial_state = {
        "query": query,
        "answer": "",
        "final_response": "",
        "cache_hit": False,
        "cache_confidence": 0.0,
        "cache_seed_id": None,
        "cache_enabled": True,
        "research_iterations": 0,
        "max_research_iterations": 3,
        "research_quality_score": 0.0,
        "research_feedback": "",
        "current_research_strategy": "",
        "execution_path": ["start"],
        "metrics": initialize_metrics(),
        "timestamp": datetime.now().isoformat(),
        "llm_calls": {},
    }
    
    final_state = workflow_app.invoke(initial_state)
    total_time = (time.perf_counter() - start_time) * 1000
    if isinstance(final_state, dict) and "metrics" in final_state:
        final_state["metrics"]["total_latency"] = total_time
        
    return final_state

def display_results(result: Dict[str, Any]) -> None:
    query = result.get("query", "")
    metrics = result.get("metrics", {})
    cache_hit = result.get("cache_hit", False)
    cache_conf = result.get("cache_confidence", 0.0)

    print("\n" + "=" * 60)
    print(f"🧐 查询: {query}")
    print("-" * 60)
    
    if cache_hit:
        print(f"🟢 缓存状态: 命中 (置信度: {cache_conf:.2f})")
    else:
        print("🔴 缓存状态: 未命中")

    print("-" * 60)
    print("📈 性能指标:")
    print(f"  • 词频总耗时: {metrics.get('total_latency', 0):.0f}ms")
    print(f"  • 节点执行路径: {' -> '.join(result.get('execution_path', []))}")
    print(f"  • 研究轮次: {result.get('research_iterations', 0)}")
    
    print("-" * 60)
    print("🤖 最终回答:")
    print(result.get("final_response", ""))
    print("=" * 60 + "\n")

def analyze_agent_results(results: list) -> tuple:
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt

    total_queries = len(results)
    total_cache_hits = sum(1 for r in results if r.get("cache_hit", False))

    logger.info(f"📊 分析 {total_queries} 个查询的性能数据...")

    analysis_data = []
    
    try:
        from testing.scenarios import SCENARIO_RUNS
        scenario_names = [s.get("title", f"Scenario_{i}") for i, s in enumerate(SCENARIO_RUNS)]
    except ImportError:
        scenario_names = [f"Scenario_{i+1}" for i in range(len(results))]

    if len(scenario_names) < len(results):
        scenario_names.extend([f"Scenario_{i+1}" for i in range(len(scenario_names), len(results))])
        
    for i, res in enumerate(results):
        scenario = scenario_names[i].split("-")[0] if "-" in scenario_names[i] else scenario_names[i]
        metrics = res.get("metrics", {})
        
        row = {
            "scenario": scenario,
            "query": res.get("query", ""),
            "cache_hit": "命中" if res.get("cache_hit", False) else "未命中",
            "latency": metrics.get("total_latency", 0),
            "research_latency": metrics.get("research_latency", 0),
            "research_iterations": res.get("research_iterations", 0),
            "cache_confidence": res.get("cache_confidence", 0.0)
        }
        analysis_data.append(row)

    df = pd.DataFrame(analysis_data)
    
    if df.empty:
        logger.warning("没有足够的数据生成图表。")
        return total_queries, total_cache_hits

    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Semantic Cache Workflow 分析报告", fontsize=16)

        cache_counts = df["cache_hit"].value_counts()
        colors = ["#2ecc71" if i == "命中" else "#e74c3c" for i in cache_counts.index]
        if not cache_counts.empty:
            axes[0, 0].pie(cache_counts.values, labels=cache_counts.index, autopct="%1.1f%%", colors=colors, startangle=90)
            axes[0, 0].set_title("缓存命中分布")

        sns.barplot(data=df, x="scenario", y="latency", hue="cache_hit", palette={"命中": "#2ecc71", "未命中": "#e74c3c"}, ax=axes[0, 1])
        axes[0, 1].set_title("各场景端到端延迟")
        axes[0, 1].set_ylabel("毫秒 (ms)")
        axes[0, 1].tick_params(axis='x', rotation=45)

        if "research_iterations" in df.columns:
            sns.barplot(data=df, x="scenario", y="research_iterations", hue="cache_hit", palette={"命中": "#2ecc71", "未命中": "#e74c3c"}, ax=axes[1, 0])
            axes[1, 0].set_title("按场景和缓存状态的研究迭代次数")
            axes[1, 0].set_ylabel("迭代次数")
            axes[1, 0].tick_params(axis='x', rotation=45)
            
        if "cache_confidence" in df.columns:
            sns.scatterplot(data=df, x="cache_confidence", y="latency", hue="cache_hit", palette={"命中": "#2ecc71", "未命中": "#e74c3c"}, s=100, ax=axes[1, 1])
            axes[1, 1].set_title("响应延迟 vs 缓存置信度")
            axes[1, 1].set_xlabel("置信度分数")
            axes[1, 1].set_ylabel("延迟 (ms)")

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.savefig("outputs/workflow_analysis.png", dpi=300, bbox_inches="tight")
        logger.info("✅ 分析图表已保存至 outputs/workflow_analysis.png")
        
        summary_text = f"""
================ 分析摘要 ================
总查询数: {total_queries}
缓存命中数: {total_cache_hits}
缓存命中率: {(total_cache_hits/total_queries)*100:.1f}%

性能对比 (按命中状态分组的平均指标):
{df.groupby('cache_hit')[['latency', 'research_iterations']].mean().round(2).to_string()}
=========================================
"""
        logger.info(summary_text)

    except Exception as e:
        logger.error(f"图表生成失败: {str(e)}")

    return total_queries, total_cache_hits


from typing import Any, Dict

from common.logger import setup_logging
logger = setup_logging()

try:
    from testing.scenarios import SCENARIO_RUNS
except ImportError:
    SCENARIO_RUNS = []
from testing.scenarios import SCENARIO_RUNS
from testing.exporter import export_results


def run_workflow_scenarios(
    workflow_app: Any,
    logger,
    show_console_results: bool = False,
) -> Dict[str, Any]:
    """运行场景、导出结果并输出分析日志。"""
    all_results = []
    for scenario in SCENARIO_RUNS:
        logger.info("运行场景：%s", scenario["title"])
        result = run_agent(workflow_app, scenario["query"])
        all_results.append(result)
        if show_console_results:
            display_results(result)

    export_paths = export_results(all_results)
    logger.info("结果已导出 | 场景汇总=%s", export_paths["summary_csv"])
    logger.info("结果已导出 | 全量JSON=%s", export_paths["result_json"])

    logger.info("分析 Agent 性能...")
    total_questions, total_cache_hits = analyze_agent_results(all_results)
    logger.info("子问题总数: %s, 缓存命中总数: %s", total_questions, total_cache_hits)

    return {
        "results": all_results,
        "exports": export_paths,
        "total_questions": total_questions,
        "total_cache_hits": total_cache_hits,
    }
