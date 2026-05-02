"""Tests for tests.result_classifiers — single source of truth for path labels."""

import unittest

from tests.result_classifiers import (
    classify_path,
    is_edit_distance_bypass,
    is_dual_subquery_reuse,
    is_exact_bypass,
    is_near_exact_bypass,
    is_partial_reuse,
    is_rerank_candidate,
    is_reranked_full_reuse,
    is_reranker_exception,
)


def _result(**overrides):
    base = {
        "intercepted": False,
        "cache_hit": False,
        "cache_match_type": "none",
        "cache_reuse_mode": "none",
        "cache_matched_question": "",
        "cache_rerank_attempt": "none",
        "cache_rerank_reason": "",
        "execution_path": [],
    }
    base.update(overrides)
    return base


class ClassifyPathTests(unittest.TestCase):
    """覆盖 8 类预期路径标签。"""

    def test_intercepted(self):
        r = _result(intercepted=True)
        self.assertEqual(classify_path(r), "拦截")

    def test_exact_bypass(self):
        r = _result(cache_hit=True, cache_match_type="exact")
        self.assertEqual(classify_path(r), "精确缓存直出")
        self.assertTrue(is_exact_bypass(r))

    def test_near_exact_bypass(self):
        r = _result(cache_hit=True, cache_match_type="near_exact")
        self.assertEqual(classify_path(r), "近精确缓存直出")
        self.assertTrue(is_near_exact_bypass(r))

    def test_edit_distance_bypass(self):
        r = _result(cache_hit=True, cache_match_type="edit_distance")
        self.assertEqual(classify_path(r), "编辑距离缓存直出")
        self.assertTrue(is_edit_distance_bypass(r))

    def test_reranker_full_reuse(self):
        r = _result(
            cache_matched_question="similar?",
            cache_match_type="semantic",
            cache_rerank_attempt="primary",
            cache_reuse_mode="full_reuse",
        )
        self.assertTrue(is_rerank_candidate(r))
        self.assertTrue(is_reranked_full_reuse(r))
        self.assertEqual(classify_path(r), "Reranker完整复用")

    def test_partial_reuse_via_mode(self):
        r = _result(
            cache_matched_question="similar?",
            cache_match_type="semantic",
            cache_rerank_attempt="primary",
            cache_reuse_mode="partial_reuse",
        )
        self.assertTrue(is_partial_reuse(r))
        self.assertEqual(classify_path(r), "部分复用+补充研究")

    def test_partial_reuse_via_execution_path(self):
        r = _result(execution_path=["pre_check", "supplement_researched"])
        self.assertTrue(is_partial_reuse(r))
        self.assertEqual(classify_path(r), "部分复用+补充研究")

    def test_dual_subquery_is_not_partial_reuse_even_with_supplement_path(self):
        r = _result(
            cache_reuse_mode="dual_subquery",
            execution_path=["pre_check", "supplement_researched"],
        )
        self.assertTrue(is_dual_subquery_reuse(r))
        self.assertFalse(is_partial_reuse(r))
        self.assertEqual(classify_path(r), "Dual Subquery Cache Hit")

    def test_reranker_reject(self):
        r = _result(
            cache_matched_question="similar?",
            cache_match_type="semantic",
            cache_rerank_attempt="primary",
            cache_reuse_mode="reject",
        )
        self.assertTrue(is_rerank_candidate(r))
        self.assertFalse(is_reranked_full_reuse(r))
        self.assertEqual(classify_path(r), "Reranker拒绝后研究")

    def test_full_research_default(self):
        r = _result()
        self.assertEqual(classify_path(r), "完整研究")

    def test_reranker_exception_predicate(self):
        r = _result(cache_rerank_reason="rerank_exception: timeout")
        self.assertTrue(is_reranker_exception(r))

    def test_intercept_short_circuits_other_signals(self):
        r = _result(
            intercepted=True,
            cache_hit=True,
            cache_match_type="exact",
        )
        self.assertEqual(classify_path(r), "拦截")

    def test_rerank_candidate_excludes_skipped(self):
        r = _result(
            cache_matched_question="similar?",
            cache_match_type="semantic",
            cache_rerank_attempt="skipped",
        )
        self.assertFalse(is_rerank_candidate(r))


if __name__ == "__main__":
    unittest.main()
