"""Tests for src/cache/engine.py pure helpers and registration logic.

These tests avoid Redis by exercising static normalizers and by constructing a fake
wrapper instance with the relevant maps pre-populated.
"""

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from cache.engine import SemanticCacheWrapper  # noqa: E402


class NormalizationTests(unittest.TestCase):
    def test_normalize_query_lowercases_and_collapses_whitespace(self):
        self.assertEqual(SemanticCacheWrapper.normalize_query("  Hello   World  "), "hello world")

    def test_normalize_surface_query_strips_punctuation_and_nfkc(self):
        # Full-width characters and trailing punctuation should be removed.
        self.assertEqual(
            SemanticCacheWrapper.normalize_surface_query("Hello, World!"),
            "helloworld",
        )

    def test_split_query_segments_uses_min_length_2(self):
        segments = SemanticCacheWrapper.split_query_segments("a？bb？cccc？dddddd")
        self.assertIn("bb", segments)
        self.assertIn("cccc", segments)
        self.assertIn("dddddd", segments)
        self.assertNotIn("a", segments)


def _make_fake_wrapper():
    """Build a stand-in object that exposes the same maps + bound methods.

    Avoids actually connecting to Redis. We bind the unbound functions onto a fresh
    SimpleNamespace so the contains/register methods resolve identically to
    `SemanticCacheWrapper` instances.
    """
    fake = types.SimpleNamespace()
    fake._seed_id_by_question = {}
    fake._answer_by_question = {}
    fake._normalized_question_map = {}
    fake._near_exact_question_map = {}
    fake._stored_normalized_question_map = {}
    fake._stored_near_exact_question_map = {}
    fake._pinned_l1_questions = set()
    fake._semantic_hit_counts = {}
    fake._l1_promotion_enabled = True
    fake._l1_promotion_threshold = 2
    fake._l1_max_entries = 128
    fake.semantic_results = []

    captured_writes = []

    class _FakeCache:
        def __init__(self, owner):
            self.owner = owner

        def store(self, prompt, response):
            captured_writes.append((prompt, response))

        def check(self, query, distance_threshold=None, num_results=1):
            return list(self.owner.semantic_results)

        @staticmethod
        def clear():
            return None

    fake.cache = _FakeCache(fake)
    fake.captured_writes = captured_writes

    # Bind the unbound methods.
    fake.normalize_query = SemanticCacheWrapper.normalize_query
    fake.normalize_surface_query = SemanticCacheWrapper.normalize_surface_query
    fake.split_query_segments = SemanticCacheWrapper.split_query_segments
    fake._levenshtein_distance_with_limit = SemanticCacheWrapper._levenshtein_distance_with_limit
    fake.find_subquery_candidate = SemanticCacheWrapper.find_subquery_candidate.__get__(fake)
    fake.find_edit_distance_candidate = SemanticCacheWrapper.find_edit_distance_candidate.__get__(fake)
    fake.register_entry = SemanticCacheWrapper.register_entry.__get__(fake)
    fake.store_runtime_entry = SemanticCacheWrapper.store_runtime_entry.__get__(fake)
    fake.contains_prompt_variant = SemanticCacheWrapper.contains_prompt_variant.__get__(fake)
    fake.check = SemanticCacheWrapper.check.__get__(fake)
    fake.get_l1_stats = SemanticCacheWrapper.get_l1_stats.__get__(fake)
    return fake


class RegisterEntryTests(unittest.TestCase):
    def test_register_entry_populates_all_four_maps(self):
        wrapper = _make_fake_wrapper()
        wrapper.register_entry("How to ship?", "Use SF Express", seed_id=42)

        self.assertEqual(wrapper._seed_id_by_question["How to ship?"], 42)
        self.assertEqual(wrapper._answer_by_question["How to ship?"], "Use SF Express")
        self.assertEqual(wrapper._normalized_question_map["how to ship?"], "How to ship?")
        # surface key strips the trailing question mark
        self.assertIn("howtoship", wrapper._near_exact_question_map)
        self.assertEqual(wrapper.captured_writes, [("How to ship?", "Use SF Express")])

    def test_register_entry_skips_empty_inputs(self):
        wrapper = _make_fake_wrapper()
        wrapper.register_entry("", "answer")
        wrapper.register_entry("prompt", "")
        self.assertEqual(wrapper.captured_writes, [])
        self.assertEqual(wrapper._answer_by_question, {})


class ContainsPromptVariantTests(unittest.TestCase):
    def test_exact_variant_detected(self):
        wrapper = _make_fake_wrapper()
        wrapper.register_entry("Refund policy", "...")
        self.assertTrue(wrapper.contains_prompt_variant("refund   policy"))

    def test_surface_variant_detected_through_punctuation(self):
        wrapper = _make_fake_wrapper()
        wrapper.register_entry("Refund policy?", "...")
        self.assertTrue(wrapper.contains_prompt_variant("Refund   Policy!"))

    def test_unknown_prompt_returns_false(self):
        wrapper = _make_fake_wrapper()
        wrapper.register_entry("Refund policy", "...")
        self.assertFalse(wrapper.contains_prompt_variant("shipping speed"))


class RuntimePromotionTests(unittest.TestCase):
    def test_runtime_entry_stays_out_of_l1_until_it_becomes_hot(self):
        wrapper = _make_fake_wrapper()

        wrapper.store_runtime_entry("Refund timeline", "Refunds usually arrive in 7 days.")

        self.assertEqual(wrapper._answer_by_question, {})
        self.assertTrue(wrapper.contains_prompt_variant("refund   timeline"))
        self.assertEqual(
            wrapper.captured_writes,
            [("Refund timeline", "Refunds usually arrive in 7 days.")],
        )

    def test_semantic_hits_promote_runtime_entry_after_threshold(self):
        wrapper = _make_fake_wrapper()
        wrapper._l1_promotion_threshold = 2
        wrapper.store_runtime_entry("Refund timeline", "Refunds usually arrive in 7 days.")
        wrapper.semantic_results = [{
            "prompt": "Refund timeline",
            "response": "Refunds usually arrive in 7 days.",
            "vector_distance": 0.1,
        }]

        wrapper.check("When will my refund arrive?")
        self.assertNotIn("Refund timeline", wrapper._answer_by_question)

        wrapper.check("How long does a refund take?")

        self.assertEqual(
            wrapper._answer_by_question["Refund timeline"],
            "Refunds usually arrive in 7 days.",
        )
        self.assertEqual(wrapper.get_l1_stats()["promoted_entries"], 1)
        self.assertEqual(wrapper.get_l1_stats()["promotion_count"], 1)

    def test_lru_evicts_only_runtime_promotions_and_keeps_pinned_faq(self):
        wrapper = _make_fake_wrapper()
        wrapper._l1_promotion_threshold = 1
        wrapper._l1_max_entries = 2
        wrapper.register_entry("How to ship?", "Use SF Express", seed_id=42)

        wrapper.store_runtime_entry("Refund timeline", "Refunds usually arrive in 7 days.")
        wrapper.semantic_results = [{
            "prompt": "Refund timeline",
            "response": "Refunds usually arrive in 7 days.",
            "vector_distance": 0.1,
        }]
        wrapper.check("When will my refund arrive?")
        self.assertIn("Refund timeline", wrapper._answer_by_question)

        wrapper.store_runtime_entry("Overseas shipping", "We support overseas delivery.")
        wrapper.semantic_results = [{
            "prompt": "Overseas shipping",
            "response": "We support overseas delivery.",
            "vector_distance": 0.1,
        }]
        wrapper.check("Do you ship abroad?")

        self.assertIn("How to ship?", wrapper._answer_by_question)
        self.assertNotIn("Refund timeline", wrapper._answer_by_question)
        self.assertIn("Overseas shipping", wrapper._answer_by_question)
        self.assertTrue(wrapper.contains_prompt_variant("refund timeline"))
        self.assertEqual(wrapper.get_l1_stats()["eviction_count"], 1)


if __name__ == "__main__":
    unittest.main()
