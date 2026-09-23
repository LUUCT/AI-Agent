import json
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from app.core.config import Settings
from app.main import app
from app.web_search import WebSearchRunner


class FakeSessionRepository:
    def __init__(self) -> None:
        self.turns: dict[str, list[dict]] = {}
        self.deleted: list[str] = []

    async def recent(self, conversation_id: str, limit: int = 30) -> list[tuple[str, str]]:
        return [
            (turn["question"], turn.get("answer", ""))
            for turn in self.turns.get(conversation_id, [])[-limit:]
        ]

    async def create(self, conversation_id: str, question: str) -> int:
        turns = self.turns.setdefault(conversation_id, [])
        turns.append({"id": len(turns) + 1, "question": question})
        return len(turns)

    async def finish(self, record_id: int, **fields: object) -> None:
        for turns in self.turns.values():
            for turn in turns:
                if turn["id"] == record_id:
                    turn.update(fields)
                    return

    async def list_page(self, page_num: int, page_size: int) -> dict:
        records = [
            {"conversationId": key, "question": turns[0]["question"], "agentType": "chat"}
            for key, turns in self.turns.items()
        ]
        return {
            "pageNum": page_num,
            "pageSize": page_size,
            "total": len(records),
            "records": records[(page_num - 1) * page_size : page_num * page_size],
        }

    async def detail(self, conversation_id: str) -> dict | None:
        turns = self.turns.get(conversation_id)
        return (
            {"conversationId": conversation_id, "agentType": "chat", "messages": turns}
            if turns
            else None
        )

    async def delete(self, conversation_id: str) -> bool:
        self.deleted.append(conversation_id)
        return self.turns.pop(conversation_id, None) is not None


class FakeModel:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.inputs: list[list] = []

    def bind_tools(self, tools: list) -> "FakeModel":
        return self

    async def astream(self, messages: list) -> AsyncIterator[AIMessageChunk]:
        self.inputs.append(messages)
        yield AIMessageChunk(content=self.answer)

    async def ainvoke(self, messages: list) -> AIMessage:
        return AIMessage(content='["继续", "来源？", "总结？"]')


def test_java_jdbc_defaults_translate_to_asyncmy_without_embedded_password() -> None:
    from app.persistence.sessions import mysql_async_url

    url = mysql_async_url(Settings(_env_file=None))
    assert url.drivername == "mysql+asyncmy"
    assert (url.host, url.port, url.database, url.username) == ("127.0.0.1", 3307, "dodo", "root")
    assert url.password == ""


@pytest.mark.asyncio
async def test_web_search_reads_and_saves_history_through_repository() -> None:
    repository = FakeSessionRepository()
    first_model = FakeModel("第一轮答案")
    first = WebSearchRunner(first_model, [], repository)
    first_types = [event.type.value async for event in first.stream("第一问", "c1")]
    assert first_types == ["text", "recommend"]
    assert repository.turns["c1"][0]["answer"] == "第一轮答案"

    # A new runner represents a process restart; history is recovered from the repository.
    second_model = FakeModel("第二轮答案")
    second = WebSearchRunner(second_model, [], repository)
    events = [event async for event in second.stream("第二问", "c1")]
    assert [event.type.value for event in events] == ["text", "recommend"]
    assert any(message.content == "第一轮答案" for message in second_model.inputs[0])
    saved = repository.turns["c1"][1]
    assert saved["answer"] == "第二轮答案"
    assert json.loads(saved["recommend"]) == ["继续", "来源？", "总结？"]
    assert isinstance(saved["total_response_time"], int)


@pytest.mark.asyncio
async def test_closing_stream_saves_answer_and_releases_task() -> None:
    repository = FakeSessionRepository()
    runner = WebSearchRunner(FakeModel("已输出的答案"), [], repository)
    stream = runner.stream("问题", "closed")
    event = await anext(stream)
    assert event.content == "已输出的答案"
    await stream.aclose()
    assert repository.turns["closed"][0]["answer"] == "已输出的答案"
    assert "closed" not in runner._active


def test_legacy_session_routes_keep_java_envelope() -> None:
    repository = FakeSessionRepository()
    repository.turns["c1"] = [{"id": 1, "question": "你好", "answer": "你好！"}]
    original = app.state.session_repository
    app.state.session_repository = repository
    try:
        with TestClient(app) as client:
            listed = client.get("/session/list", params={"pageNum": 1, "pageSize": 100})
            detail = client.get("/session/c1")
            missing = client.get("/session/missing")
            deleted = client.delete("/session/c1")
    finally:
        app.state.session_repository = original
    assert listed.json()["data"]["records"][0]["conversationId"] == "c1"
    assert detail.json()["data"]["messages"][0]["answer"] == "你好！"
    assert missing.json() == {"code": 500, "message": "会话不存在", "data": None}
    assert deleted.json() == {"code": 200, "message": "会话删除成功", "data": None}
