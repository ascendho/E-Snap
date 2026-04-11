import csv
import json
import os
from typing import Dict, List

from cache.evals import PerfEval


def export_results(all_results: List[Dict], total_costs: Dict, output_dir: str = "outputs") -> Dict[str, str]:
    """
    将运行结果导出为 CSV + JSON 文件，便于后续分析与归档。

    Returns:
        导出文件路径字典。
    """
    os.makedirs(output_dir, exist_ok=True)

    summary_csv = os.path.join(output_dir, "run_summary.csv")
    usage_csv = os.path.join(output_dir, "llm_usage.csv")
    result_json = os.path.join(output_dir, "run_results.json")

    # 场景级汇总：每条主查询一行。
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario_index",
                "original_query",
                "sub_questions_count",
                "cache_hits",
                "cache_hit_rate",
                "analysis_llm_calls",
                "research_llm_calls",
                "total_latency_ms",
                "cost",
                "currency",
                "final_response",
            ],
        )
        writer.writeheader()

        for idx, result in enumerate(all_results, 1):
            sub_questions = result.get("sub_questions", [])
            cache_hits_map = result.get("cache_hits", {})
            hits = sum(1 for v in cache_hits_map.values() if v)
            total_sq = len(sub_questions)
            hit_rate = (hits / total_sq) if total_sq else 0.0

            llm_calls = result.get("llm_calls", {})
            total_latency = str(result.get("total_latency", "0ms")).replace("ms", "")

            per_perf = PerfEval()
            for call in result.get("llm_usage", []):
                per_perf.record_llm_call(
                    model=call.get("model", "unknown-model"),
                    provider=call.get("provider", "openai"),
                    input_tokens=int(call.get("input_tokens", 0) or 0),
                    output_tokens=int(call.get("output_tokens", 0) or 0),
                )
            per_perf.set_total_queries(1)
            per_cost = per_perf.get_costs()

            writer.writerow(
                {
                    "scenario_index": idx,
                    "original_query": result.get("original_query", ""),
                    "sub_questions_count": total_sq,
                    "cache_hits": hits,
                    "cache_hit_rate": f"{hit_rate:.4f}",
                    "analysis_llm_calls": llm_calls.get("analysis_llm", 0),
                    "research_llm_calls": llm_calls.get("research_llm", 0),
                    "total_latency_ms": total_latency,
                    "cost": per_cost.get("total_cost", 0.0),
                    "currency": per_cost.get("currency", total_costs.get("currency", "CNY")),
                    "final_response": result.get("final_response", ""),
                }
            )

    # 调用级明细：每次 LLM 调用一行。
    with open(usage_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario_index",
                "provider",
                "model",
                "input_tokens",
                "output_tokens",
            ],
        )
        writer.writeheader()

        for idx, result in enumerate(all_results, 1):
            for call in result.get("llm_usage", []):
                writer.writerow(
                    {
                        "scenario_index": idx,
                        "provider": call.get("provider", "openai"),
                        "model": call.get("model", "unknown-model"),
                        "input_tokens": int(call.get("input_tokens", 0) or 0),
                        "output_tokens": int(call.get("output_tokens", 0) or 0),
                    }
                )

    payload = {
        "cost_summary": total_costs,
        "results": all_results,
    }
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        "summary_csv": summary_csv,
        "usage_csv": usage_csv,
        "result_json": result_json,
    }
