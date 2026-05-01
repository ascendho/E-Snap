"""Regression tests for offline runner metric accounting."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.runner import run_agent  # noqa: E402


class RunnerMetricsTests(unittest.TestCase):
    def test_run_agent_total_latency_includes_background_wait(self):
        workflow_app = SimpleNamespace(invoke=lambda initial_state: {"metrics": {}, **initial_state})

        with patch("workflow.nodes.build_initial_state", return_value={"query": "Q", "metrics": {}}):
            with patch("workflow.nodes.wait_for_background_tasks", side_effect=lambda state: state) as mocked_wait:
                with patch("tests.runner.time.perf_counter", side_effect=[10.0, 10.75]):
                    final_state = run_agent(workflow_app, "Q")

        mocked_wait.assert_called_once()
        self.assertEqual(final_state["metrics"]["total_latency"], 750.0)


if __name__ == "__main__":
    unittest.main()