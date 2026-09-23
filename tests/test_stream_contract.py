import json
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from app.main import app
from app.models.events import StreamEvent, StreamEventType


class StubAgentRunner:
    async def stream(self, query: str, conversation_id: str) -> AsyncIterator[StreamEvent]:
        assert query == "测试"
        assert conversation_id == "conversation-test"
        yield StreamEvent(type=StreamEventType.THINKING, content="正在处理...\n")
        yield StreamEvent(
            type=StreamEventType.TOOL_START,
            tool_name="test_tool",
            tool_call_id="tool-1",
        )
        yield StreamEvent(type=StreamEventType.TEXT, content="真实运行器的替身响应")
        yield StreamEvent(
            type=StreamEventType.TOOL_END,
            tool_name="test_tool",
            tool_call_id="tool-1",
        )
        yield StreamEvent(type=StreamEventType.COMPLETE)


def _decode_sse_events(body: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for block in body.strip().split("\n\n"):
        assert block.startswith("data: ")
        events.append(json.loads(block.removeprefix("data: ")))
    return events


def test_chat_stream_preserves_legacy_event_contract() -> None:
    original_runner = app.state.chat_runner
    app.state.chat_runner = StubAgentRunner()
    try:
        with TestClient(app) as client:
            response = client.get(
                "/agent/chat/stream",
                params={"query": "测试", "conversationId": "conversation-test"},
            )
    finally:
        app.state.chat_runner = original_runner

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _decode_sse_events(response.text)
    assert [event["type"] for event in events] == [
        "thinking",
        "tool_start",
        "text",
        "tool_end",
        "complete",
    ]
    assert events[1]["toolName"] == "test_tool"
    assert events[1]["toolCallId"] == "tool-1"
    assert events[2]["content"] == "真实运行器的替身响应"


def test_blank_query_is_rejected_before_streaming() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/agent/chat/stream",
            params={"query": "   ", "conversationId": "conversation-test"},
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "查询参数不能为空"


def test_stop_endpoint_keeps_java_response_shape() -> None:
    with TestClient(app) as client:
        response = client.get("/agent/stop", params={"conversationId": "not-running"})
    assert response.json() == {
        "success": False,
        "message": "没有找到正在执行的任务或已停止",
    }


def test_stop_endpoint_awaits_task_manager_result() -> None:
    class StubStopRunner:
        async def stop(self, conversation_id: str) -> bool:
            assert conversation_id == "running"
            return True

    original_runner = app.state.chat_runner
    app.state.chat_runner = StubStopRunner()
    try:
        with TestClient(app) as client:
            response = client.get("/agent/stop", params={"conversationId": "running"})
    finally:
        app.state.chat_runner = original_runner

    assert response.json() == {"success": True, "message": "已停止执行"}
