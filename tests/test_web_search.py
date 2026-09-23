import json

import pytest
from fastmcp import FastMCP
from langchain.mcp import MCPAdapter
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from app.web_search import WebSearchRunner
from app.web_search.prompts import FORCE_FINAL_PROMPT, WEB_SEARCH_PROMPT
from app.web_search.references import parse_tavily_references


@pytest.mark.asyncio
async def test_mcp_adapter_discovers_real_schema_without_hardcoding() -> None:
    server = FastMCP("fixture")

    @server.tool(name="tavily_search")
    def search(query: str) -> str:
        return query

    async with MCPAdapter(server) as adapter:
        tools = await adapter.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "tavily_search"
        assert "query" in tools[0].args_schema["properties"]


class FakeTool:
    name = "tavily_search"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def ainvoke(self, args: dict) -> list[dict]:
        self.calls.append(args)
        return [
            {
                "text": json.dumps(
                    {
                        "results": [
                            {"url": "https://example.org", "title": "标题", "content": "摘要"}
                        ]
                    }
                )
            }
        ]


class FakeModel:
    def __init__(self, rounds: list[list[AIMessageChunk]]) -> None:
        self.rounds = rounds
        self.inputs: list[list] = []
        self.bound_tool_names: list[str] = []

    def bind_tools(self, tools: list) -> "FakeModel":
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    async def astream(self, messages: list):
        self.inputs.append(list(messages))
        for chunk in self.rounds.pop(0):
            yield chunk

    async def ainvoke(self, messages: list) -> AIMessage:
        return AIMessage(content='["接下来呢？", "还有什么？", "如何验证？"]')


def _call(index: int) -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"name": "tavily_search", "args": '{"query":"测试"}', "id": f"call-{index}", "index": 0}
        ],
    )


@pytest.mark.asyncio
async def test_fragmented_tool_arguments_keep_original_call_id() -> None:
    first = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"name": "tavily_search", "args": '{"query":"', "id": "original-id", "index": 0}
        ],
    )
    second = AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": None, "args": '分片"}', "id": None, "index": 0}],
    )
    model = FakeModel([[first, second], [AIMessageChunk(content="答案")]])
    tool = FakeTool()
    runner = WebSearchRunner(model, [tool])

    events = [event async for event in runner.stream("问题", "fragmented")]

    assert tool.calls == [{"query": "分片"}]
    assert model.inputs[1][-1].tool_call_id == "original-id"
    assert [event.type.value for event in events] == ["thinking", "text", "reference", "recommend"]


@pytest.mark.asyncio
async def test_search_follows_java_prompt_and_emits_reference_then_recommend() -> None:
    tool = FakeTool()
    model = FakeModel([[_call(1)], [AIMessageChunk(content="查到结果。")]])
    runner = WebSearchRunner(model, [tool])

    events = [event async for event in runner.stream("今天有什么新闻", "session-1")]

    assert model.bound_tool_names == ["tavily_search"]
    assert model.inputs[0][0].content.startswith(WEB_SEARCH_PROMPT.split("{current_time}")[0])
    assert model.inputs[0][-1].content == "<question>今天有什么新闻</question>"
    assert isinstance(model.inputs[1][-1], ToolMessage)
    assert model.inputs[1][-1].tool_call_id == "call-1"
    assert tool.calls == [{"query": "测试"}]
    assert [event.type.value for event in events] == ["thinking", "text", "reference", "recommend"]
    assert events[0].content == "🔍 正在搜索信息: 测试\n"
    assert events[2].count == 1
    assert json.loads(events[2].content) == [
        {"url": "https://example.org", "title": "标题", "content": "摘要"}
    ]


@pytest.mark.asyncio
async def test_fifth_tool_round_is_skipped_and_forced_to_finish() -> None:
    tool = FakeTool()
    model = FakeModel([[_call(i)] for i in range(5)] + [[AIMessageChunk(content="最终答案")]])
    runner = WebSearchRunner(model, [tool])

    events = [event async for event in runner.stream("问题", "session-2")]

    assert len(tool.calls) == 4
    assert model.inputs[-1][-1].content == FORCE_FINAL_PROMPT
    assert any(event.type.value == "text" and event.content == "最终答案" for event in events)


@pytest.mark.parametrize("as_string", [True, False])
def test_reference_parser_matches_java_nested_response(as_string: bool) -> None:
    payload = {
        "results": [
            {"url": "https://a.example", "title": "A", "content": "C", "score": 1},
            {"url": " ", "title": "ignore"},
        ]
    }
    raw = [{"text": json.dumps(payload) if as_string else payload}]

    assert parse_tavily_references(raw) == [
        {"url": "https://a.example", "title": "A", "content": "C"}
    ]
