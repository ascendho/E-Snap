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

    captured_writes = []

    class _FakeCache:
        @staticmethod
        def store(prompt, response):
            captured_writes.append((prompt, response))

    fake.cache = _FakeCache()
    fake.captured_writes = captured_writes

    # Bind the unbound methods.
    fake.normalize_query = SemanticCacheWrapper.normalize_query
    fake.normalize_surface_query = SemanticCacheWrapper.normalize_surface_query
    fake.register_entry = SemanticCacheWrapper.register_entry.__get__(fake)
    fake.contains_prompt_variant = SemanticCacheWrapper.contains_prompt_variant.__get__(fake)
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


if __name__ == "__main__":
    unittest.main()
