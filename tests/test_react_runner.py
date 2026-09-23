import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk

from app.agents.specs import ReactAgentSpec
from app.runtime.runner import ReactAgentRunner, UnavailableAgentRunner


class FakeCompiledAgent:
    def __init__(self) -> None:
        self.config: dict[str, Any] | None = None

    async def astream_events(
        self,
        inputs: dict[str, Any],
        *,
        config: dict[str, Any],
        version: str,
    ) -> AsyncIterator[dict[str, Any]]:
        assert inputs["messages"][0]["content"] == "hello"
        assert version == "v2"
        self.config = config
        yield {
            "event": "on_tool_start",
            "name": "clock",
            "run_id": "run-1",
            "data": {"input": {"timezone": "Asia/Shanghai"}},
        }
        yield {
            "event": "on_tool_end",
            "name": "clock",
            "run_id": "run-1",
            "data": {"output": "2026-09-17T12:00:00+08:00"},
        }
        yield {
            "event": "on_chat_model_stream",
            "name": "fake-model",
            "run_id": "model-1",
            "data": {"chunk": AIMessageChunk(content="hello back")},
        }


class CancelledCompiledAgent:
    async def astream_events(
        self,
        inputs: dict[str, Any],
        *,
        config: dict[str, Any],
        version: str,
    ) -> AsyncIterator[dict[str, Any]]:
        del inputs, config, version
        if False:
            yield {}
        raise asyncio.CancelledError


@pytest.mark.asyncio
async def test_react_runner_translates_framework_events() -> None:
    agent = FakeCompiledAgent()
    spec = ReactAgentSpec(name="test", system_prompt="test", recursion_limit=8)
    runner = ReactAgentRunner(agent=agent, spec=spec)

    events = [event async for event in runner.stream("hello", "conversation-1")]

    assert [event.type for event in events] == [
        "thinking",
        "tool_start",
        "tool_end",
        "text",
        "complete",
    ]
    assert events[1].tool_name == "clock"
    assert events[3].content == "hello back"
    assert agent.config == {
        "configurable": {"thread_id": "conversation-1"},
        "recursion_limit": 8,
    }


@pytest.mark.asyncio
async def test_unavailable_runner_returns_stable_error() -> None:
    runner = UnavailableAgentRunner("missing key")

    events = [event async for event in runner.stream("hello", "conversation-1")]

    assert [event.type for event in events] == ["error", "complete"]
    assert events[0].message == "missing key"
    assert events[0].code == "AGENT_NOT_CONFIGURED"


@pytest.mark.asyncio
async def test_react_runner_propagates_cancellation_without_complete_event() -> None:
    spec = ReactAgentSpec(name="cancelled", system_prompt="test")
    runner = ReactAgentRunner(agent=CancelledCompiledAgent(), spec=spec)

    with pytest.raises(asyncio.CancelledError):
        [event async for event in runner.stream("hello", "conversation-1")]
