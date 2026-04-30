from typing import Dict, Any
def run_agent(workflow_app, query: str) -> Dict[str, Any]:
    from workflow.nodes import build_initial_state
    import time
    
    start_time = time.perf_counter()
    initial_state = build_initial_state(query)
    
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
    cache_match_type = result.get("cache_match_type", "none")
    cache_reuse_mode = result.get("cache_reuse_mode", "none")

    print("\n" + "=" * 60)
    print(f"🧐 查询: {query}")
    print("-" * 60)
    
    if cache_hit:
        print(f"🟢 缓存状态: 命中[{cache_match_type}] (置信度: {cache_conf:.2f})")
    elif cache_reuse_mode == "partial_reuse":
        print(f"🟡 缓存状态: 部分复用[{cache_match_type}] (置信度: {cache_conf:.2f})")
    else:
        print("🔴 缓存状态: 未命中")

    print("-" * 60)
    print("📈 性能指标:")
    print(f"  • 词频总耗时: {metrics.get('total_latency', 0):.0f}ms")
    print(f"  • 节点执行路径: {' -> '.join(result.get('execution_path', []))}")
    
    print("-" * 60)
    print("🤖 最终回答:")
    print(result.get("final_response", ""))
    print("=" * 60 + "\n")

def classify_result_path(result: Dict[str, Any]) -> str:
    if result.get("intercepted", False):
        return "拦截"
    if result.get("cache_hit", False) and result.get("cache_match_type") == "exact":
        return "精确缓存直出"
    if result.get("cache_hit", False) and result.get("cache_match_type") == "near_exact":
        return "近精确缓存直出"
    if result.get("cache_reuse_mode") == "full_reuse":
        return "Reranker完整复用"
    if result.get("cache_reuse_mode") == "partial_reuse":
        return "部分复用+补充研究"
    if result.get("cache_matched_question"):
        return "Reranker拒绝后研究"
    return "完整研究"

def analyze_agent_results(results: list) -> tuple:
    total_queries = len(results)
    total_cache_hits = sum(1 for r in results if r.get("cache_hit", False))

    logger.info(f"📊 分析 {total_queries} 个查询的性能数据...")

    try:
        import pandas as pd
        
        analysis_data = []
        for i, res in enumerate(results):
            metrics = res.get("metrics", {})
            row = {
                "path_type": classify_result_path(res),
                "latency": metrics.get("total_latency", 0),
                "total_llm_calls": sum(res.get("llm_calls", {}).values()),
            }
            analysis_data.append(row)

        df = pd.DataFrame(analysis_data)
        if not df.empty:
            summary_text = f"""
================ 分析摘要 ================
总查询数: {total_queries}
缓存命中数: {total_cache_hits}
缓存命中率: {(total_cache_hits/total_queries)*100 if total_queries > 0 else 0:.1f}%

性能对比 (按路径分组的平均指标):
{df.groupby('path_type')[['latency', 'total_llm_calls']].mean().round(2).to_string()}
=========================================
"""
            logger.info(summary_text)
    except Exception as e:
        logger.error(f"分析摘要生成失败: {str(e)}")

    return total_queries, total_cache_hits


from typing import Any, Dict

from common.logger import setup_logging
logger = setup_logging()

try:
    from tests.scenarios import SCENARIO_RUNS
except ImportError:
    SCENARIO_RUNS = []
from tests.scenarios import SCENARIO_RUNS
from tests.exporter import export_results
import time


def run_workflow_scenarios(
    workflow_app: Any,
    logger,
    show_console_results: bool = False,
) -> Dict[str, Any]:
    """运行场、导出结果并输出分析日志。"""
    start_time = time.perf_counter()
    all_results = []
    for scenario in SCENARIO_RUNS:
        logger.info("运行场景：%s", scenario["title"])
        result = run_agent(workflow_app, scenario["query"])
        all_results.append(result)
        if show_console_results:
            display_results(result)

    total_wall_time = time.perf_counter() - start_time
    export_paths = export_results(all_results, total_wall_time)
    logger.info("结果已导出 | 场景明细=%s", export_paths["summary_csv"])
    logger.info("结果已导出 | 宏观性能=%s", export_paths["perf_metrics_txt"])

    logger.info("分析 Agent 性能...")
    total_questions, total_cache_hits = analyze_agent_results(all_results)
    logger.info("子问题总数: %s, 缓存命中总数: %s, 吞吐量 QPS: %.2f，总耗时 %.2fs", 
                total_questions, total_cache_hits, len(all_results)/total_wall_time, total_wall_time)

    return {
        "results": all_results,
        "exports": export_paths,
        "total_questions": total_questions,
        "total_cache_hits": total_cache_hits,
    }
