import csv
import json
import os
from typing import Dict, List


def export_results(all_results: List[Dict], total_wall_time_sec: float, output_dir: str = "outputs") -> Dict[str, str]:
    """
    将运行结果导出为纯净的统报表CSV文件，分别提供明细以及对命中率、吞吐量等指标的性能聚合(TXT)。
    """
    os.makedirs(output_dir, exist_ok=True)

    summary_csv = os.path.join(output_dir, "run_summary.csv")
    perf_metrics_txt = os.path.join(output_dir, "performance_report.txt")

    def execution_path(result: Dict) -> List[str]:
        return result.get("execution_path", []) or []

    def total_llm_calls(result: Dict) -> int:
        return sum(result.get("llm_calls", {}).values())

    def avg_latency(subset: List[Dict]) -> float:
        if not subset:
            return 0.0
        return sum(r.get("metrics", {}).get("total_latency", 0) for r in subset) / len(subset)

    def avg_metric(subset: List[Dict], metric_name: str) -> float:
        if not subset:
            return 0.0
        return sum(r.get("metrics", {}).get(metric_name, 0) for r in subset) / len(subset)

    def avg_unattributed_overhead(subset: List[Dict]) -> float:
        if not subset:
            return 0.0
        total = 0.0
        for result in subset:
            metrics = result.get("metrics", {})
            attributed = sum(
                float(metrics.get(metric_name, 0) or 0)
                for metric_name in [
                    "precheck_latency",
                    "cache_latency",
                    "rerank_latency",
                    "research_latency",
                    "supplement_latency",
                    "synthesis_latency",
                ]
            )
            total += max(float(metrics.get("total_latency", 0) or 0) - attributed, 0.0)
        return total / len(subset)

    def avg_state_value(subset: List[Dict], field_name: str) -> float:
        if not subset:
            return 0.0
        return sum(float(r.get(field_name, 0) or 0) for r in subset) / len(subset)

    def avg_llm_calls(subset: List[Dict]) -> float:
        if not subset:
            return 0.0
        return sum(total_llm_calls(r) for r in subset) / len(subset)

    def sum_usage_value(subset: List[Dict], field_name: str) -> float:
        if not subset:
            return 0.0
        return sum(float(r.get("llm_usage", {}).get(field_name, 0) or 0) for r in subset)

    def avg_usage_value(subset: List[Dict], field_name: str) -> float:
        if not subset:
            return 0.0
        return sum_usage_value(subset, field_name) / len(subset)

    def is_exact_bypass(result: Dict) -> bool:
        return result.get("cache_hit", False) and result.get("cache_match_type") == "exact"

    def is_near_exact_bypass(result: Dict) -> bool:
        return result.get("cache_hit", False) and result.get("cache_match_type") == "near_exact"

    def is_edit_distance_bypass(result: Dict) -> bool:
        return result.get("cache_hit", False) and result.get("cache_match_type") == "edit_distance"

    def is_rerank_candidate(result: Dict) -> bool:
        return (
            bool(result.get("cache_matched_question"))
            and result.get("cache_match_type") not in {"exact", "near_exact", "none"}
            and result.get("cache_rerank_attempt") not in {"skipped", "none"}
        )

    def is_reranked_full_reuse(result: Dict) -> bool:
        return is_rerank_candidate(result) and result.get("cache_reuse_mode") == "full_reuse"

    def is_partial_reuse(result: Dict) -> bool:
        return result.get("cache_reuse_mode") == "partial_reuse" or "supplement_researched" in execution_path(result)

    def is_reranker_exception(result: Dict) -> bool:
        return str(result.get("cache_rerank_reason", "")).startswith("rerank_exception:")

    def classify_path(result: Dict) -> str:
        if result.get("intercepted", False):
            return "拦截"
        if is_exact_bypass(result):
            return "精确缓存直出"
        if is_near_exact_bypass(result):
            return "近精确缓存直出"
        if is_edit_distance_bypass(result):
            return "编辑距离缓存直出"
        if is_reranked_full_reuse(result):
            return "Reranker完整复用"
        if is_partial_reuse(result):
            return "部分复用+补充研究"
        if is_rerank_candidate(result):
            return "Reranker拒绝后研究"
        return "完整研究"

    # 1. 测试项汇总：每条主查询一行。
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "test_index",
                "original_query",
                "intercepted",
                "cache_hit",
                "cache_candidate_found",
                "cache_match_type",
                "cache_reuse_mode",
                "cache_matched_question",
                "cache_confidence",
                "cache_rerank_passed",
                "cache_rerank_attempt",
                "cache_rerank_score",
                "cache_rerank_reason",
                "cache_reranker_reason",
                "cache_validation_reason",
                "cache_residual_query",
                "cache_written_prompts",
                "analysis_llm_calls",
                "research_llm_calls",
                "total_llm_calls",
                "analysis_input_tokens",
                "analysis_output_tokens",
                "analysis_cached_input_tokens",
                "research_input_tokens",
                "research_output_tokens",
                "research_cached_input_tokens",
                "total_input_tokens",
                "total_output_tokens",
                "total_cached_input_tokens",
                "analysis_cost_rmb",
                "research_cost_rmb",
                "total_cost_rmb",
                "precheck_latency_ms",
                "cache_latency_ms",
                "rerank_latency_ms",
                "research_latency_ms",
                "supplement_latency_ms",
                "synthesis_latency_ms",
                "total_latency_ms",
                "final_response",
            ],
        )
        writer.writeheader()

        for idx, result in enumerate(all_results, 1):
            intercepted = result.get("intercepted", False)
            cache_hit = result.get("cache_hit", False)
            cache_candidate_found = bool(result.get("cache_matched_question"))
            cache_confidence = result.get("cache_confidence", 0.0)

            llm_calls = result.get("llm_calls", {})
            llm_usage = result.get("llm_usage", {})
            metrics = result.get("metrics", {})
            total_latency = f"{metrics.get('total_latency', 0):.0f}"

            writer.writerow(
                {
                    "test_index": idx,
                    "original_query": result.get("query", ""),
                    "intercepted": str(intercepted),
                    "cache_hit": str(cache_hit),
                    "cache_candidate_found": str(cache_candidate_found),
                    "cache_match_type": result.get("cache_match_type", "none"),
                    "cache_reuse_mode": result.get("cache_reuse_mode", "none"),
                    "cache_matched_question": result.get("cache_matched_question", ""),
                    "cache_confidence": f"{cache_confidence:.4f}",
                    "cache_rerank_passed": str(result.get("cache_rerank_passed", False)),
                    "cache_rerank_attempt": result.get("cache_rerank_attempt", "none"),
                    "cache_rerank_score": f"{result.get('cache_rerank_score', 0.0):.4f}",
                    "cache_rerank_reason": result.get("cache_rerank_reason", ""),
                    "cache_reranker_reason": result.get("cache_reranker_reason", ""),
                    "cache_validation_reason": result.get("cache_validation_reason", ""),
                    "cache_residual_query": result.get("cache_residual_query", ""),
                    "cache_written_prompts": json.dumps(result.get("cache_written_prompts", []) or [], ensure_ascii=False),
                    "analysis_llm_calls": llm_calls.get("analysis_llm", 0),
                    "research_llm_calls": llm_calls.get("research_llm", 0),
                    "total_llm_calls": total_llm_calls(result),
                    "analysis_input_tokens": int(llm_usage.get("analysis_input_tokens", 0) or 0),
                    "analysis_output_tokens": int(llm_usage.get("analysis_output_tokens", 0) or 0),
                    "analysis_cached_input_tokens": int(llm_usage.get("analysis_cached_input_tokens", 0) or 0),
                    "research_input_tokens": int(llm_usage.get("research_input_tokens", 0) or 0),
                    "research_output_tokens": int(llm_usage.get("research_output_tokens", 0) or 0),
                    "research_cached_input_tokens": int(llm_usage.get("research_cached_input_tokens", 0) or 0),
                    "total_input_tokens": int(llm_usage.get("total_input_tokens", 0) or 0),
                    "total_output_tokens": int(llm_usage.get("total_output_tokens", 0) or 0),
                    "total_cached_input_tokens": int(llm_usage.get("total_cached_input_tokens", 0) or 0),
                    "analysis_cost_rmb": f"{float(llm_usage.get('analysis_cost_rmb', 0) or 0):.6f}",
                    "research_cost_rmb": f"{float(llm_usage.get('research_cost_rmb', 0) or 0):.6f}",
                    "total_cost_rmb": f"{float(llm_usage.get('total_cost_rmb', 0) or 0):.6f}",
                    "precheck_latency_ms": f"{metrics.get('precheck_latency', 0):.0f}",
                    "cache_latency_ms": f"{metrics.get('cache_latency', 0):.0f}",
                    "rerank_latency_ms": f"{metrics.get('rerank_latency', 0):.0f}",
                    "research_latency_ms": f"{metrics.get('research_latency', 0):.0f}",
                    "supplement_latency_ms": f"{metrics.get('supplement_latency', 0):.0f}",
                    "synthesis_latency_ms": f"{metrics.get('synthesis_latency', 0):.0f}",
                    "total_latency_ms": total_latency,
                    "final_response": result.get("final_response", ""),
                }
            )

    # 2. 计算并聚合性能级指标报表
    total_queries = len(all_results)
    intercepted_paths = [r for r in all_results if r.get("intercepted", False)]
    eligible_queries = [r for r in all_results if not r.get("intercepted", False)]
    direct_cache_reuse_paths = [r for r in eligible_queries if r.get("cache_hit", False)]
    research_paths = [r for r in eligible_queries if "researched" in execution_path(r)]
    partial_reuse_paths = [r for r in eligible_queries if is_partial_reuse(r)]
    cache_candidates = [r for r in eligible_queries if r.get("cache_matched_question")]
    rerank_candidates = [r for r in eligible_queries if is_rerank_candidate(r)]
    reranked_full_reuse_paths = [r for r in rerank_candidates if r.get("cache_reuse_mode") == "full_reuse"]
    rerank_reusable_paths = [r for r in rerank_candidates if r.get("cache_reuse_mode") in {"full_reuse", "partial_reuse"}]
    reranker_reject_paths = [r for r in rerank_candidates if r.get("cache_reuse_mode") == "reject"]
    reranker_exception_paths = [r for r in reranker_reject_paths if is_reranker_exception(r)]
    reranker_fallback_paths = [r for r in rerank_candidates if r.get("cache_rerank_attempt") == "fallback"]
    deterministic_partial_paths = [r for r in partial_reuse_paths if r.get("cache_rerank_attempt") == "skipped"]
    reranked_partial_paths = [r for r in partial_reuse_paths if r.get("cache_rerank_attempt") not in {"skipped", "none"}]
    exact_bypass_paths = [r for r in eligible_queries if is_exact_bypass(r)]
    near_exact_bypass_paths = [r for r in eligible_queries if is_near_exact_bypass(r)]
    edit_distance_bypass_paths = [r for r in eligible_queries if is_edit_distance_bypass(r)]
    semantic_full_reuse_paths = [r for r in reranked_full_reuse_paths if r.get("cache_match_type") == "semantic"]
    subquery_full_reuse_paths = [r for r in reranked_full_reuse_paths if str(r.get("cache_match_type", "")).startswith("subquery_")]

    eligible_count = len(eligible_queries)
    direct_reuse_count = len(direct_cache_reuse_paths)
    candidate_count = len(cache_candidates)
    rerank_candidate_count = len(rerank_candidates)
    intercept_count = len(intercepted_paths)
    direct_cache_reuse_rate = direct_reuse_count / eligible_count if eligible_count > 0 else 0.0
    hybrid_reuse_rate = len(partial_reuse_paths) / eligible_count if eligible_count > 0 else 0.0
    candidate_hit_rate = candidate_count / eligible_count if eligible_count > 0 else 0.0
    rerank_reuse_rate = len(rerank_reusable_paths) / rerank_candidate_count if rerank_candidate_count > 0 else 0.0
    throughput = total_queries / total_wall_time_sec if total_wall_time_sec > 0 else 0.0

    research_latency = avg_latency(research_paths)
    direct_cache_reuse_latency = avg_latency(direct_cache_reuse_paths)
    intercepted_latency = avg_latency(intercepted_paths)
    avg_precheck_latency = avg_metric(all_results, "precheck_latency")
    avg_cache_check_latency = avg_metric(eligible_queries, "cache_latency")
    avg_rerank_latency = avg_metric(rerank_candidates, "rerank_latency")
    avg_research_stage_latency = avg_metric(research_paths, "research_latency")
    avg_supplement_latency = avg_metric(partial_reuse_paths, "supplement_latency")
    avg_synthesis_latency = avg_metric(all_results, "synthesis_latency")
    avg_other_overhead_latency = avg_unattributed_overhead(all_results)

    research_call_cost = avg_llm_calls(research_paths)
    direct_cache_reuse_call_cost = avg_llm_calls(direct_cache_reuse_paths)
    partial_reuse_call_cost = avg_llm_calls(partial_reuse_paths)
    actual_eligible_call_cost = avg_llm_calls(eligible_queries)
    llm_savings_per_request = max(research_call_cost - actual_eligible_call_cost, 0.0)

    total_analysis_input_tokens = sum_usage_value(all_results, "analysis_input_tokens")
    total_analysis_output_tokens = sum_usage_value(all_results, "analysis_output_tokens")
    total_analysis_cached_input_tokens = sum_usage_value(all_results, "analysis_cached_input_tokens")
    total_research_input_tokens = sum_usage_value(all_results, "research_input_tokens")
    total_research_output_tokens = sum_usage_value(all_results, "research_output_tokens")
    total_research_cached_input_tokens = sum_usage_value(all_results, "research_cached_input_tokens")
    total_input_tokens = sum_usage_value(all_results, "total_input_tokens")
    total_output_tokens = sum_usage_value(all_results, "total_output_tokens")
    total_cached_input_tokens = sum_usage_value(all_results, "total_cached_input_tokens")

    total_analysis_cost_rmb = sum_usage_value(all_results, "analysis_cost_rmb")
    total_research_cost_rmb = sum_usage_value(all_results, "research_cost_rmb")
    actual_total_cost_rmb = sum_usage_value(all_results, "total_cost_rmb")
    actual_eligible_total_cost_rmb = sum_usage_value(eligible_queries, "total_cost_rmb")
    pure_rag_avg_cost_rmb = avg_usage_value(research_paths, "total_cost_rmb")
    direct_cache_reuse_avg_cost_rmb = avg_usage_value(direct_cache_reuse_paths, "total_cost_rmb")
    partial_reuse_avg_cost_rmb = avg_usage_value(partial_reuse_paths, "total_cost_rmb")
    actual_eligible_avg_cost_rmb = avg_usage_value(eligible_queries, "total_cost_rmb")

    direct_reuse_saved_latency = max(research_latency - direct_cache_reuse_latency, 0.0) * direct_reuse_count
    intercept_saved_latency = max(research_latency - intercepted_latency, 0.0) * intercept_count
    partial_reuse_penalty_latency = max(avg_latency(partial_reuse_paths) - research_latency, 0.0) * len(partial_reuse_paths)

    full_reuse_saved_calls = max(research_call_cost - direct_cache_reuse_call_cost, 0.0) * direct_reuse_count
    partial_reuse_added_calls = max(partial_reuse_call_cost - research_call_cost, 0.0) * len(partial_reuse_paths)
    net_saved_calls_total = max(full_reuse_saved_calls - partial_reuse_added_calls, 0.0)

    theory_total_cost_without_cache_rmb = pure_rag_avg_cost_rmb * eligible_count
    direct_reuse_saved_cost_rmb = max(pure_rag_avg_cost_rmb - direct_cache_reuse_avg_cost_rmb, 0.0) * direct_reuse_count
    partial_reuse_extra_cost_rmb = max(partial_reuse_avg_cost_rmb - pure_rag_avg_cost_rmb, 0.0) * len(partial_reuse_paths)
    net_saved_cost_rmb = max(theory_total_cost_without_cache_rmb - actual_eligible_total_cost_rmb, 0.0)
    cost_reduction_ratio = ((theory_total_cost_without_cache_rmb - actual_eligible_total_cost_rmb) / theory_total_cost_without_cache_rmb * 100.0) if theory_total_cost_without_cache_rmb > 0 else 0.0

    intercepted_total_time = sum(r.get("metrics", {}).get("total_latency", 0) for r in intercepted_paths)
    theory_total_time_without_cache = research_latency * eligible_count + intercepted_total_time
    actual_total_time = sum(r.get("metrics", {}).get("total_latency", 0) for r in all_results)

    if theory_total_time_without_cache > 0:
        latency_reduction = ((theory_total_time_without_cache - actual_total_time) / theory_total_time_without_cache) * 100
    else:
        latency_reduction = 0.0

    # 吞吐量指标
    baseline_qps = total_queries / (theory_total_time_without_cache / 1000.0) if theory_total_time_without_cache > 0 else 0.0
    max_qps = 1000.0 / direct_cache_reuse_latency if direct_cache_reuse_latency > 0 else 0.0
    rerank_latency_text = f"{avg_rerank_latency:.0f} ms ({rerank_candidate_count} 候选)" if rerank_candidate_count > 0 else "N/A (未触发)"
    supplement_latency_text = f"{avg_supplement_latency:.0f} ms ({len(partial_reuse_paths)} 条路径)" if partial_reuse_paths else "N/A (未触发)"

    report_text = f"""======================================================
         AGENT CACHE PERFORMANCE REPORT
======================================================

1. 总体概况 (Overview)
------------------------------------------------------
测试集总请求数 : {total_queries} 次
前置拦截数     : {intercept_count} 次
可参与缓存查询数 : {eligible_count} 次
缓存候选数     : {candidate_count} 次
需要 Reranker 的候选数 : {rerank_candidate_count} 次
直接缓存复用数（仅 cache_hit=True） : {direct_reuse_count} 次
exact 旁路数   : {len(exact_bypass_paths)} 次
near_exact 旁路数 : {len(near_exact_bypass_paths)} 次
edit_distance 旁路数 : {len(edit_distance_bypass_paths)} 次
Reranker full_reuse 数 : {len(reranked_full_reuse_paths)} 次
    其中 semantic full_reuse 数 : {len(semantic_full_reuse_paths)} 次
    其中 subquery full_reuse 数 : {len(subquery_full_reuse_paths)} 次
partial_reuse 数（单列统计，不计入 direct cache hit） : {len(partial_reuse_paths)} 次
    其中 deterministic subquery partial 数 : {len(deterministic_partial_paths)} 次
    其中 reranker partial 数 : {len(reranked_partial_paths)} 次
Reranker reject 数 : {len(reranker_reject_paths)} 次
Reranker exception 数 : {len(reranker_exception_paths)} 次
Reranker fallback 数 : {len(reranker_fallback_paths)} 次
缓存候选率     : {candidate_hit_rate * 100:.2f}%
直接缓存复用率（仅 cache_hit=True） : {direct_cache_reuse_rate * 100:.2f}%
partial_reuse 率 : {hybrid_reuse_rate * 100:.2f}%
测试总墙上时间 : {total_wall_time_sec:.2f} 秒

2. 路径延迟拆分 (Path Latency Breakdown)
------------------------------------------------------
- 平均 Pre-check 耗时         : {avg_precheck_latency:.0f} ms
- 平均缓存检查耗时           : {avg_cache_check_latency:.0f} ms
- 平均 Reranker 耗时         : {rerank_latency_text}
- 平均 Research 阶段耗时     : {avg_research_stage_latency:.0f} ms
- 平均 Supplement 阶段耗时   : {supplement_latency_text}
- 平均 Synthesis 阶段耗时    : {avg_synthesis_latency:.0f} ms
- 平均其他未拆分开销         : {avg_other_overhead_latency:.0f} ms
- 直接缓存复用路径平均总耗时 : {direct_cache_reuse_latency:.0f} ms
- RAG 路径平均总耗时         : {research_latency:.0f} ms
- 前置拦截路径平均总耗时     : {intercepted_latency:.0f} ms

3. Reranker 效果
------------------------------------------------------
Reranker 可复用判定数        : {len(rerank_reusable_paths)} 次
Reranker 可复用率            : {rerank_reuse_rate * 100:.2f}%
平均 Reranker 置信度         : {avg_state_value(rerank_candidates, 'cache_rerank_score'):.2f}

4. 吞吐量对比 (Throughput / QPS)
------------------------------------------------------
估算无缓存基线 QPS           : {baseline_qps:.2f} 请求/秒
加入缓存后实测 QPS            : {throughput:.2f} 请求/秒
直接缓存复用路径理论峰值 QPS  : {max_qps:.2f} 请求/秒
吞吐量提升倍数                : {throughput / baseline_qps if baseline_qps > 0 else 0:.2f} 倍

5. 延迟总降低 (Total Latency Reduction)
------------------------------------------------------
估算无缓存基线总耗时          : {theory_total_time_without_cache:.0f} ms
实际执行总耗时                : {actual_total_time:.0f} ms
full_reuse 节省总耗时         : {direct_reuse_saved_latency:.0f} ms
intercept 节省总耗时          : {intercept_saved_latency:.0f} ms
partial_reuse 额外耗时         : {partial_reuse_penalty_latency:.0f} ms
📌 Latency Reduction         : {latency_reduction:.2f}% 

(说明: baseline 为估算值，保留前置拦截路径，仅将可参与缓存的请求替换为 RAG 路径平均耗时)

6. Token 消耗 (Token Breakdown)
------------------------------------------------------
Analysis 输入 tokens 总数        : {total_analysis_input_tokens:.0f}
Analysis 输出 tokens 总数        : {total_analysis_output_tokens:.0f}
Analysis 缓存命中输入 tokens 总数 : {total_analysis_cached_input_tokens:.0f}
Research 输入 tokens 总数        : {total_research_input_tokens:.0f}
Research 输出 tokens 总数        : {total_research_output_tokens:.0f}
Research 缓存命中输入 tokens 总数 : {total_research_cached_input_tokens:.0f}
总输入 tokens                  : {total_input_tokens:.0f}
总输出 tokens                  : {total_output_tokens:.0f}
总缓存命中输入 tokens           : {total_cached_input_tokens:.0f}

7. 真实金额成本 (RMB Cost Breakdown)
------------------------------------------------------
Analysis 成本总计              : ￥{total_analysis_cost_rmb:.6f}
Research 成本总计              : ￥{total_research_cost_rmb:.6f}
全部请求总成本                : ￥{actual_total_cost_rmb:.6f}
纯 RAG 路径平均成本           : ￥{pure_rag_avg_cost_rmb:.6f}
直接缓存复用平均成本          : ￥{direct_cache_reuse_avg_cost_rmb:.6f}
partial_reuse 平均成本        : ￥{partial_reuse_avg_cost_rmb:.6f}
可参与缓存请求平均实际成本     : ￥{actual_eligible_avg_cost_rmb:.6f}
估算无缓存基线总成本          : ￥{theory_total_cost_without_cache_rmb:.6f}
实际可参与缓存总成本          : ￥{actual_eligible_total_cost_rmb:.6f}
full_reuse 节省金额总计       : ￥{direct_reuse_saved_cost_rmb:.6f}
partial_reuse 额外成本总计     : ￥{partial_reuse_extra_cost_rmb:.6f}
净节省金额总计                : ￥{net_saved_cost_rmb:.6f}
📌 金额节省比例               : {cost_reduction_ratio:.2f}%

(说明: 金额基于 API 返回的真实 usage metadata 与当前配置单价；仅在 cached input token > 0 时按厂商缓存命中单价计费)

8. LLM 调用次数（调试）
------------------------------------------------------
纯 RAG 路径平均 LLM 调用数    : {research_call_cost:.2f} 次
直接缓存复用平均 LLM 调用数   : {direct_cache_reuse_call_cost:.2f} 次
partial_reuse 平均 LLM 调用数 : {partial_reuse_call_cost:.2f} 次
可参与缓存请求平均实际调用数  : {actual_eligible_call_cost:.2f} 次
full_reuse 节省调用总数       : {full_reuse_saved_calls:.2f} 次
partial_reuse 额外调用总数    : {partial_reuse_added_calls:.2f} 次
净节省调用总数                : {net_saved_calls_total:.2f} 次
📌 调用节省（相对纯RAG基线）   : {llm_savings_per_request:.2f} 次 / 请求

(说明: 该节仅作为调试辅助，真实成本请以第 7 节人民币金额为准)
======================================================
"""
    with open(perf_metrics_txt, "w", encoding="utf-8") as f:
        f.write(report_text)

    # 尝试删除旧的csv指标文件
    old_csv_path = os.path.join(output_dir, "performance_metrics.csv")
    if os.path.exists(old_csv_path):
        try:
            os.remove(old_csv_path)
        except OSError:
            pass

    return {
        "summary_csv": summary_csv,
        "perf_metrics_txt": perf_metrics_txt,
    }
