"""End-to-end exporter contract tests.

Validates the CSV column set is exactly SUMMARY_FIELDNAMES, in order, and that
TXT report aggregation does not crash on representative inputs.
"""

import csv
import os
import tempfile
import unittest

from tests.exporter import export_results
from tests.summary_schema import SUMMARY_FIELDNAMES


def _intercepted_result():
    return {
        "query": "intercepted query",
        "intercepted": True,
        "execution_path": ["pre_check"],
        "llm_calls": {},
        "llm_usage": {},
        "metrics": {"precheck_latency": 5, "total_latency": 5},
        "final_response": "blocked",
    }


def _exact_cache_result():
    return {
        "query": "exact cached",
        "intercepted": False,
        "cache_hit": True,
        "cache_match_type": "exact",
        "cache_reuse_mode": "full_reuse",
        "cache_matched_question": "exact cached",
        "cache_confidence": 1.0,
        "cache_rerank_attempt": "skipped",
        "execution_path": ["pre_check", "cache_hit"],
        "llm_calls": {},
        "llm_usage": {"total_cost_rmb": 0.0},
        "metrics": {
            "precheck_latency": 4,
            "cache_latency": 8,
            "synthesis_latency": 2,
            "total_latency": 14,
        },
        "final_response": "cached answer",
    }


def _full_research_result():
    return {
        "query": "deep research",
        "intercepted": False,
        "cache_hit": False,
        "cache_match_type": "none",
        "cache_reuse_mode": "none",
        "execution_path": ["pre_check", "researched"],
        "llm_calls": {"analysis_llm": 1, "research_llm": 2},
        "llm_usage": {
            "analysis_input_tokens": 100,
            "analysis_output_tokens": 50,
            "research_input_tokens": 300,
            "research_output_tokens": 200,
            "total_input_tokens": 400,
            "total_output_tokens": 250,
            "total_cost_rmb": 0.012345,
            "research_cost_rmb": 0.011,
            "analysis_cost_rmb": 0.001345,
        },
        "metrics": {
            "precheck_latency": 4,
            "cache_latency": 8,
            "research_latency": 1500,
            "synthesis_latency": 100,
            "total_latency": 1700,
        },
        "final_response": "researched answer",
    }


def _reranked_full_reuse_result():
    return {
        "query": "semantic reused",
        "intercepted": False,
        "cache_hit": False,
        "cache_match_type": "semantic",
        "cache_reuse_mode": "full_reuse",
        "cache_matched_question": "semantic cached",
        "cache_confidence": 0.99,
        "cache_rerank_attempt": "semantic",
        "execution_path": ["pre_check", "rerank_reused"],
        "llm_calls": {"analysis_llm": 1},
        "llm_usage": {"analysis_cost_rmb": 0.002, "total_cost_rmb": 0.002},
        "metrics": {
            "precheck_latency": 2,
            "cache_latency": 3,
            "rerank_latency": 10,
            "synthesis_latency": 5,
            "total_latency": 300,
        },
        "final_response": "reranked answer",
    }


def _partial_reuse_result():
    return {
        "query": "partial reuse",
        "intercepted": False,
        "cache_hit": False,
        "cache_match_type": "subquery_near_exact",
        "cache_reuse_mode": "partial_reuse",
        "cache_matched_question": "cached segment",
        "cache_confidence": 0.98,
        "cache_rerank_attempt": "skipped",
        "execution_path": ["pre_check", "supplement_researched"],
        "llm_calls": {"research_llm": 1},
        "llm_usage": {"research_cost_rmb": 0.004, "total_cost_rmb": 0.004},
        "metrics": {
            "precheck_latency": 3,
            "cache_latency": 4,
            "supplement_latency": 400,
            "synthesis_latency": 50,
            "total_latency": 500,
        },
        "final_response": "partial answer",
    }


def _dual_subquery_result():
    return {
        "query": "dual subquery",
        "intercepted": False,
        "cache_hit": False,
        "cache_match_type": "subquery_near_exact",
        "cache_reuse_mode": "dual_subquery",
        "cache_matched_question": "cached segment",
        "cache_confidence": 0.99,
        "cache_rerank_attempt": "skipped",
        "execution_path": ["pre_check", "supplement_researched"],
        "llm_calls": {},
        "llm_usage": {"total_cost_rmb": 0.0},
        "metrics": {
            "precheck_latency": 2,
            "cache_latency": 3,
            "supplement_latency": 20,
            "synthesis_latency": 10,
            "total_latency": 40,
        },
        "final_response": "dual cached answer",
    }


class ExporterContractTests(unittest.TestCase):
    def test_csv_columns_match_schema_exactly(self):
        results = [_intercepted_result(), _exact_cache_result(), _full_research_result()]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_results(results, total_wall_time_sec=12.5, output_dir=tmpdir)
            csv_path = paths["summary_csv"]
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)
        self.assertEqual(tuple(header), SUMMARY_FIELDNAMES)
        self.assertEqual(len(rows), 3)
        # Each row must have the same column count as the header.
        for row in rows:
            self.assertEqual(len(row), len(SUMMARY_FIELDNAMES))

    def test_empty_results_writes_header_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_results([], total_wall_time_sec=0.0, output_dir=tmpdir)
            with open(paths["summary_csv"], "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                rest = list(reader)
        self.assertEqual(tuple(header), SUMMARY_FIELDNAMES)
        self.assertEqual(rest, [])

    def test_perf_report_contains_known_sections(self):
        results = [_intercepted_result(), _exact_cache_result(), _full_research_result()]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_results(results, total_wall_time_sec=10.0, output_dir=tmpdir)
            with open(paths["perf_metrics_txt"], "r", encoding="utf-8") as f:
                text = f.read()
        for section in [
            "1. 总体概况",
            "2. 路径延迟拆分",
            "3. Reranker 效果",
            "4. 吞吐量对比",
            "5. 延迟总降低",
            "6. Token 消耗",
            "7. 真实金额成本",
            "8. LLM 调用次数",
        ]:
            self.assertIn(section, text)
        self.assertIn("测试集总请求数 : 3", text)
        self.assertIn("前置拦截数     : 1", text)

    def test_perf_report_full_reuse_savings_include_reranked_reuse(self):
        full_research = {
            "query": "research baseline",
            "intercepted": False,
            "cache_hit": False,
            "cache_match_type": "none",
            "cache_reuse_mode": "none",
            "execution_path": ["pre_check", "researched"],
            "llm_calls": {"analysis_llm": 1, "research_llm": 2},
            "llm_usage": {"analysis_cost_rmb": 0.001, "research_cost_rmb": 0.009, "total_cost_rmb": 0.01},
            "metrics": {"precheck_latency": 1, "cache_latency": 1, "research_latency": 900, "synthesis_latency": 100, "total_latency": 1000},
            "final_response": "researched",
        }
        direct_reuse = {
            "query": "direct reuse",
            "intercepted": False,
            "cache_hit": True,
            "cache_match_type": "exact",
            "cache_reuse_mode": "full_reuse",
            "cache_matched_question": "direct reuse",
            "cache_confidence": 1.0,
            "cache_rerank_attempt": "skipped",
            "execution_path": ["pre_check", "cache_hit"],
            "llm_calls": {},
            "llm_usage": {"total_cost_rmb": 0.0},
            "metrics": {"precheck_latency": 1, "cache_latency": 1, "total_latency": 100},
            "final_response": "cached",
        }
        reranked_reuse = {
            "query": "semantic reuse",
            "intercepted": False,
            "cache_hit": False,
            "cache_match_type": "semantic",
            "cache_reuse_mode": "full_reuse",
            "cache_matched_question": "semantic reuse",
            "cache_confidence": 0.99,
            "cache_rerank_attempt": "semantic",
            "execution_path": ["pre_check", "rerank_reused"],
            "llm_calls": {"analysis_llm": 1},
            "llm_usage": {"analysis_cost_rmb": 0.002, "total_cost_rmb": 0.002},
            "metrics": {"precheck_latency": 1, "cache_latency": 1, "rerank_latency": 10, "total_latency": 300},
            "final_response": "reranked",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_results([direct_reuse, reranked_reuse, full_research], total_wall_time_sec=3.0, output_dir=tmpdir)
            with open(paths["perf_metrics_txt"], "r", encoding="utf-8") as f:
                text = f.read()

        self.assertIn("full_reuse 节省总耗时         : 1600 ms", text)
        self.assertIn("full_reuse 节省金额总计       : ￥0.018000", text)
        self.assertIn("full_reuse 节省调用总数       : 5.00 次", text)

    def test_perf_report_falls_back_to_partial_reuse_sample_when_research_sample_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_results([_partial_reuse_result(), _exact_cache_result()], total_wall_time_sec=2.0, output_dir=tmpdir)
            with open(paths["perf_metrics_txt"], "r", encoding="utf-8") as f:
                text = f.read()

        self.assertIn("估算无缓存基线总耗时          : 1000 ms", text)
        self.assertIn("估算无缓存基线总成本          : ￥0.008000", text)

    def test_perf_report_keeps_dual_subquery_out_of_partial_reuse_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_results(
                [_partial_reuse_result(), _dual_subquery_result(), _exact_cache_result()],
                total_wall_time_sec=3.0,
                output_dir=tmpdir,
            )
            with open(paths["perf_metrics_txt"], "r", encoding="utf-8") as f:
                text = f.read()

        self.assertIn("partial_reuse 数（单列统计，不计入 direct cache hit） : 1 次", text)
        self.assertIn("dual_subquery 数（双子问题均来自缓存） : 1 次", text)
        self.assertIn("- 平均 Supplement 阶段耗时   : 400 ms (1 条路径)", text)
        self.assertIn("- 平均 Dual Subquery 补充阶段耗时 : 20 ms (1 条路径)", text)


if __name__ == "__main__":
    unittest.main()
