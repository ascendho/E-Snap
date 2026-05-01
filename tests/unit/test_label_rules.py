"""Tests for src/api/server.py label rules and /labels contract."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Importing api.server starts FastAPI app construction but no network I/O.
from api.server import (  # noqa: E402
    DEFAULT_LABEL,
    LABEL_RULES,
    build_label_metadata,
    resolve_label,
)


def _state(**overrides):
    base = {
        "intercepted": False,
        "cache_hit": False,
        "cache_match_type": "none",
        "cache_reuse_mode": "none",
    }
    base.update(overrides)
    return base


class LabelRulesPriorityTests(unittest.TestCase):
    def test_intercept_wins_over_cache(self):
        s = _state(intercepted=True, cache_hit=True, cache_match_type="exact")
        self.assertEqual(resolve_label(s), ("zero_intercept", "Zero-Layer Intercept"))

    def test_exact_hit(self):
        s = _state(cache_hit=True, cache_match_type="exact")
        self.assertEqual(resolve_label(s)[0], "cache_exact")

    def test_near_exact_hit(self):
        s = _state(cache_hit=True, cache_match_type="near_exact")
        self.assertEqual(resolve_label(s)[0], "cache_near_exact")

    def test_edit_distance_hit(self):
        s = _state(cache_hit=True, cache_match_type="edit_distance")
        self.assertEqual(resolve_label(s)[0], "cache_edit_distance")

    def test_full_reuse(self):
        s = _state(cache_reuse_mode="full_reuse")
        self.assertEqual(resolve_label(s)[0], "cache_semantic_reuse")

    def test_dual_subquery(self):
        s = _state(cache_reuse_mode="dual_subquery")
        self.assertEqual(resolve_label(s)[0], "cache_dual_subquery")

    def test_partial_reuse(self):
        s = _state(cache_reuse_mode="partial_reuse")
        self.assertEqual(resolve_label(s)[0], "cache_partial_reuse")

    def test_default_fallback(self):
        self.assertEqual(resolve_label(_state()), DEFAULT_LABEL)

    def test_full_reuse_does_not_override_direct_hit(self):
        # cache_hit + exact match must take precedence over reuse_mode signal.
        s = _state(cache_hit=True, cache_match_type="exact", cache_reuse_mode="full_reuse")
        self.assertEqual(resolve_label(s)[0], "cache_exact")


class LabelMetadataShapeTests(unittest.TestCase):
    def test_metadata_keys_match_chat_response_contract(self):
        meta = build_label_metadata(_state(cache_hit=True, cache_match_type="near_exact"))
        self.assertEqual(
            set(meta.keys()),
            {
                "cache_hit",
                "intercepted",
                "cache_match_type",
                "cache_reuse_mode",
                "label_key",
                "label_text",
            },
        )
        self.assertEqual(meta["label_key"], "cache_near_exact")
        self.assertEqual(meta["label_text"], "Near-Exact Cache Hit")


class LabelRulesTableTests(unittest.TestCase):
    def test_keys_are_unique(self):
        keys = [key for key, _, _ in LABEL_RULES]
        self.assertEqual(len(keys), len(set(keys)))

    def test_default_key_not_in_rules(self):
        keys = {key for key, _, _ in LABEL_RULES}
        self.assertNotIn(DEFAULT_LABEL[0], keys)


if __name__ == "__main__":
    unittest.main()
