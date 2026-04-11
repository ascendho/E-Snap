from typing import Any, Dict

from agent import analyze_agent_results, display_results, run_agent
from cache.evals import PerfEval, format_cost
from .scenarios import SCENARIO_RUNS
from utility.results_export import export_results


def run_workflow_scenarios(
    workflow_app: Any,
    logger,
    show_console_results: bool = False,
) -> Dict[str, Any]:
    """运行场景、统计成本、导出结果并输出分析日志。"""
    all_results = []
    for scenario in SCENARIO_RUNS:
        logger.info("运行场景：%s", scenario["title"])
        result = run_agent(workflow_app, scenario["query"])
        all_results.append(result)
        if show_console_results:
            display_results(result)

    perf = PerfEval()
    for result in all_results:
        for call in result.get("llm_usage", []):
            perf.record_llm_call(
                model=call.get("model", "unknown-model"),
                provider=call.get("provider", "openai"),
                input_tokens=int(call.get("input_tokens", 0) or 0),
                output_tokens=int(call.get("output_tokens", 0) or 0),
            )

    perf.set_total_queries(len(all_results))
    costs = perf.get_costs()

    logger.info(
        "成本汇总 | 币种=%s | 总成本=%s | 单问题=%s | 单调用=%s | 调用次数=%d",
        costs.get("currency", "CNY"),
        format_cost(costs.get("total_cost", 0.0), costs.get("currency", "CNY")),
        format_cost(costs.get("avg_cost_per_query", 0.0), costs.get("currency", "CNY")),
        format_cost(costs.get("avg_cost_per_call", 0.0), costs.get("currency", "CNY")),
        costs.get("calls", 0),
    )

    export_paths = export_results(all_results, costs)
    logger.info("结果已导出 | 场景汇总=%s", export_paths["summary_csv"])
    logger.info("结果已导出 | 调用明细=%s", export_paths["usage_csv"])
    logger.info("结果已导出 | 全量JSON=%s", export_paths["result_json"])

    logger.info("分析 Agent 性能...")
    total_questions, total_cache_hits = analyze_agent_results(all_results)
    logger.info("子问题总数: %s, 缓存命中总数: %s", total_questions, total_cache_hits)

    return {
        "results": all_results,
        "costs": costs,
        "exports": export_paths,
        "total_questions": total_questions,
        "total_cache_hits": total_cache_hits,
    }
