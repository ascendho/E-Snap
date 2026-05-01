"""Tests for the streaming completion contract in src/api/server.py."""

import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from api.server import (  # noqa: E402
    ChatRequest,
    RouteTarget,
    _build_stream_ready_final_event,
    _finalize_total_latency,
    _request_disconnected,
    chat_stream_endpoint,
)
from workflow.state import initialize_llm_usage  # noqa: E402


def _state(**overrides):
    base = {
        "final_response": "**代理合作**\n\n请发送邮件联系我们。",
        "cache_written_prompts": ["我想做代理商，怎么和你们合作？"],
        "intercepted": False,
        "cache_hit": False,
        "cache_match_type": "none",
        "cache_reuse_mode": "none",
        "metrics": {},
        "background_threads": [object()],
    }
    base.update(overrides)
    return base


class StreamReadyFinalEventTests(unittest.TestCase):
    def test_stream_ready_final_event_waits_for_background_tasks(self):
        state = _state(cache_reuse_mode="partial_reuse")
        with patch("api.server.wait_for_background_tasks") as mocked_wait:
            with patch("api.server.time.time", return_value=10.25):
                event = _build_stream_ready_final_event(state, 10.0, state["final_response"])

        mocked_wait.assert_called_once_with(state)
        self.assertEqual(event["answer"], state["final_response"])
        self.assertEqual(event["latency_ms"], 250.0)
        self.assertEqual(event["cache_written_prompts"], state["cache_written_prompts"])
        self.assertEqual(event["label_key"], "cache_partial_reuse")
        self.assertEqual(event["label_text"], "Partial Cache Reuse + RAG")
        self.assertEqual(state["metrics"]["total_latency"], 250.0)

    def test_stream_ready_final_event_initializes_total_latency(self):
        state = _state()
        with patch("api.server.wait_for_background_tasks") as mocked_wait:
            with patch("api.server.time.time", return_value=4.5):
                _build_stream_ready_final_event(state, 4.0, state["final_response"])

        mocked_wait.assert_called_once_with(state)
        self.assertEqual(state["metrics"]["total_latency"], 500.0)


class FinalizeTotalLatencyTests(unittest.TestCase):
    def test_finalize_total_latency_waits_and_persists_metric(self):
        state = _state(metrics={})
        with patch("api.server.wait_for_background_tasks") as mocked_wait:
            with patch("api.server.time.time", return_value=20.75):
                latency = _finalize_total_latency(state, 20.0)

        mocked_wait.assert_called_once_with(state)
        self.assertEqual(latency, 750.0)
        self.assertEqual(state["metrics"]["total_latency"], 750.0)

    def test_finalize_total_latency_initializes_metrics_if_missing(self):
        state = _state(metrics=None)
        with patch("api.server.wait_for_background_tasks"):
            with patch("api.server.time.time", return_value=8.01):
                latency = _finalize_total_latency(state, 8.0)

        self.assertEqual(latency, 10.0)
        self.assertEqual(state["metrics"]["total_latency"], 10.0)


class RequestDisconnectedTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_false_without_request(self):
        self.assertFalse(await _request_disconnected(None))

    async def test_returns_false_when_checker_missing(self):
        self.assertFalse(await _request_disconnected(SimpleNamespace()))

    async def test_returns_checker_value(self):
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=True))
        self.assertTrue(await _request_disconnected(request))

    async def test_checker_exception_treated_as_disconnected(self):
        request = SimpleNamespace(is_disconnected=AsyncMock(side_effect=RuntimeError("closed")))
        self.assertTrue(await _request_disconnected(request))


class ChatStreamRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_partial_reuse_route_enters_research_supplement(self):
        payload = ChatRequest(query="怎么联系人工？支持海外发货吗？", access_code="HIRE_ME_2026")
        request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
        initial_state = {"query": payload.query, "execution_path": []}
        checked_state = {
            **initial_state,
            "cache_hit": False,
            "cache_match_type": "subquery_near_exact",
            "cache_reuse_mode": "partial_reuse",
        }
        supplemented_state = {
            **checked_state,
            "answer": "客服热线 400-820-2026。\n\n补充说明：\n目前主要支持国内发货。",
            "cache_reuse_mode": "dual_subquery",
            "execution_path": ["cache_checked", "supplement_researched"],
            "llm_calls": {"research_llm": 0},
        }
        final_state = {
            **supplemented_state,
            "final_response": supplemented_state["answer"],
            "cache_written_prompts": ["支持海外发货吗", payload.query],
        }

        with patch("api.server.validate_chat_request"):
            with patch("api.server._request_disconnected", new=AsyncMock(return_value=False)):
                with patch("api.server.build_initial_state", return_value=initial_state):
                    with patch("api.server.pre_check_node", return_value=initial_state):
                        with patch("api.server.check_cache_node", return_value=checked_state):
                            with patch("api.server.cache_router", return_value=RouteTarget.RESEARCH_SUPPLEMENT):
                                with patch("api.server.research_supplement_node", return_value=supplemented_state):
                                    with patch("api.server.synthesize_response_node", return_value=final_state):
                                        response = await chat_stream_endpoint(payload, request)
                                        events = []
                                        async for chunk in response.body_iterator:
                                            text = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                                            for line in text.splitlines():
                                                if line.strip():
                                                    events.append(json.loads(line))

        status_stages = [event.get("stage") for event in events if event.get("type") == "status"]
        self.assertIn(RouteTarget.RESEARCH_SUPPLEMENT, status_stages)
        self.assertNotIn(RouteTarget.RESEARCH, status_stages)
        final_event = next(event for event in events if event.get("type") == "final")
        self.assertEqual(final_event["cache_reuse_mode"], "dual_subquery")
        self.assertEqual(final_event["label_key"], "cache_dual_subquery")

    async def test_research_route_records_usage_and_waits_before_final(self):
        payload = ChatRequest(query="订单什么时候发货？", access_code="HIRE_ME_2026")
        request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
        initial_state = {
            "query": payload.query,
            "execution_path": [],
            "metrics": {},
            "llm_calls": {},
            "llm_usage": initialize_llm_usage(),
            "llm_usage_lock": None,
            "background_threads": [object()],
        }
        captured_states = []
        stream_chunks = [
            SimpleNamespace(content="预计", usage_metadata={}),
            SimpleNamespace(
                content="明天发货。",
                usage_metadata={"input_tokens": 12, "output_tokens": 6, "input_token_details": {"cache_read": 0}},
            ),
        ]

        def synthesize_side_effect(state):
            captured_states.append(state)
            return {**state, "final_response": state["answer"]}

        with patch("api.server.validate_chat_request"):
            with patch("api.server._request_disconnected", new=AsyncMock(return_value=False)):
                with patch("api.server.build_initial_state", return_value=initial_state):
                    with patch("api.server.pre_check_node", return_value=initial_state):
                        with patch("api.server.check_cache_node", return_value=initial_state):
                            with patch("api.server.cache_router", return_value=RouteTarget.RESEARCH):
                                with patch("api.server.prepare_research_messages", return_value=(["msg"], 1, True)) as mocked_prepare:
                                    with patch("api.server.get_research_llm", return_value=SimpleNamespace(stream=lambda messages: iter(stream_chunks))):
                                        with patch("api.server.wait_for_background_tasks") as mocked_wait:
                                            with patch("api.server.synthesize_response_node", side_effect=synthesize_side_effect):
                                                response = await chat_stream_endpoint(payload, request)
                                                events = []
                                                async for chunk in response.body_iterator:
                                                    text = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                                                    for line in text.splitlines():
                                                        if line.strip():
                                                            events.append(json.loads(line))

        mocked_prepare.assert_called_once()
        self.assertIs(mocked_prepare.call_args.kwargs["llm_usage"], initial_state["llm_usage"])
        self.assertIs(mocked_prepare.call_args.kwargs["usage_lock"], initial_state["llm_usage_lock"])
        mocked_wait.assert_called_once()
        self.assertEqual(len(captured_states), 1)
        streamed_state = captured_states[0]
        self.assertEqual(streamed_state["answer"], "预计明天发货。")
        self.assertEqual(streamed_state["llm_calls"]["research_llm"], 2)
        self.assertEqual(streamed_state["llm_usage"]["research_calls"], 1)
        self.assertEqual(streamed_state["llm_usage"]["research_input_tokens"], 12)
        self.assertEqual(streamed_state["llm_usage"]["research_output_tokens"], 6)
        self.assertGreaterEqual(streamed_state["metrics"]["research_latency"], 0.0)
        final_event = next(event for event in events if event.get("type") == "final")
        self.assertGreaterEqual(final_event["latency_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()