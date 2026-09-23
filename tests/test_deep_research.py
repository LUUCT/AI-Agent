import json
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from app.deep_research import DeepResearchRunner
from app.deep_research.models import PlanTask
from app.main import app
from app.models.events import StreamEvent, StreamEventType


class FakeResearchTool:
    """返回 Java 解析器支持的 Tavily MCP 结构，并记录每次搜索参数。"""

    name = "tavily_search"
    description = "搜索互联网"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def ainvoke(self, args: dict) -> list[dict]:
        """记录搜索并返回一个固定来源，供任务答案和 reference 事件使用。"""
        self.calls.append(args)
        return [
            {
                "text": json.dumps(
                    {
                        "results": [
                            {
                                "url": "https://example.org/research",
                                "title": "研究来源",
                                "content": "检索事实",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            }
        ]


class ScriptedResearchModel:
    """按系统提示词识别研究阶段，使测试覆盖完整 Plan-Execute 调用链。"""

    def __init__(self, *, needs_clarification: bool = False) -> None:
        self.needs_clarification = needs_clarification
        self.inputs: list[list] = []
        self.plan_calls = 0

    def bind_tools(self, tools: list) -> "ScriptedResearchModel":
        """记录绑定动作后返回自身；测试模型根据消息决定是否产生 ToolCall。"""
        self.tools = tools
        return self

    async def astream(self, messages: list) -> AsyncIterator[AIMessageChunk]:
        """为澄清、主题和报告阶段提供流式片段。"""
        self.inputs.append(list(messages))
        system = str(messages[0].content)
        if "需求分析专家" in system:
            text = (
                "【需要补充信息】请说明研究对象。"
                if self.needs_clarification
                else "<think>判断中</think>【开始研究】研究测试主题。"
            )
        elif "分析点规划专家" in system:
            text = "1. 事实背景\n2. 最新进展\n3. 证据来源"
        elif "结果总结专家" in system:
            text = "# 测试主题分析报告\n\n基于检索事实形成报告。"
        else:
            raise AssertionError(f"unexpected streaming phase: {system[:80]}")
        yield AIMessageChunk(content=text)

    async def ainvoke(self, messages: list) -> AIMessage:
        """为规划、搜索子任务、评审和可能的压缩阶段返回确定性结果。"""
        self.inputs.append(list(messages))
        system = str(messages[0].content)
        if "执行计划规划专家" in system:
            self.plan_calls += 1
            return AIMessage(
                content='[{"id":"task-1","instruction":"调用搜索工具查询事实","order":1}]'
            )
        if "研究评审专家" in system:
            return AIMessage(content='{"passed":true,"feedback":"证据充分"}')
        if "工具执行与结果整理专家" in system:
            if any(isinstance(message, ToolMessage) for message in messages):
                return AIMessage(content="搜索任务事实整理")
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "tavily_search",
                        "args": {"query": "测试事实"},
                        "id": "call-1",
                    }
                ],
            )
        if "上下文内容压缩器" in system:
            return AIMessage(content="压缩状态")
        raise AssertionError(f"unexpected invoke phase: {system[:80]}")


class FakeResearchSessions:
    """以内存记录模拟 MySQL 会话仓储，并保留 create/finish 的全部字段。"""

    def __init__(self) -> None:
        self.turns: list[dict] = []

    async def recent(self, conversation_id: str, limit: int = 30) -> list[tuple[str, str]]:
        """返回同一会话最近的问答；本测试从空历史开始。"""
        return []

    async def create(
        self,
        conversation_id: str,
        question: str,
        *,
        agent_type: str | None = "chat",
    ) -> int:
        """创建待回填记录，并检查研究路径保持 Java 的空 agent_type。"""
        self.turns.append(
            {
                "id": len(self.turns) + 1,
                "conversation_id": conversation_id,
                "question": question,
                "agent_type": agent_type,
            }
        )
        return len(self.turns)

    async def finish(self, record_id: int, **fields: object) -> None:
        """将研究流累计结果回填到对应内存记录。"""
        self.turns[record_id - 1].update(fields)


@pytest.mark.asyncio
async def test_deep_research_runs_java_flow_and_persists_exact_report() -> None:
    """完整研究应执行搜索、评审和报告，并在报告后发送 Java 格式来源。"""
    model = ScriptedResearchModel()
    tool = FakeResearchTool()
    sessions = FakeResearchSessions()
    runner = DeepResearchRunner(model, [tool], sessions)

    events = [event async for event in runner.stream("研究测试主题", "deep-1")]

    types = [event.type.value for event in events]
    assert types[-2:] == ["text", "reference"]
    assert "complete" not in types
    assert tool.calls == [{"query": "测试事实"}]
    assert model.plan_calls == 1
    assert json.loads(events[-1].content) == [
        {
            "url": "https://example.org/research",
            "title": "研究来源",
            "content": "检索事实",
        }
    ]
    saved = sessions.turns[0]
    assert saved["agent_type"] is None
    assert saved["answer"] == "# 测试主题分析报告\n\n基于检索事实形成报告。"
    assert saved["tools"] == ""
    assert saved["recommend"] is None
    assert json.loads(saved["reference"])["type"] == "reference"


@pytest.mark.asyncio
async def test_clarification_pause_stops_before_planning() -> None:
    """问题不清时应输出暂停文本、保存它，并且不生成主题、计划或报告。"""
    model = ScriptedResearchModel(needs_clarification=True)
    sessions = FakeResearchSessions()
    runner = DeepResearchRunner(model, [], sessions)

    events = [event async for event in runner.stream("帮我研究一下", "deep-pause")]

    assert events[-1].type is StreamEventType.TEXT
    assert events[-1].content == "⏸【暂停深入研究】请说明研究对象。"
    assert model.plan_calls == 0
    assert sessions.turns[0]["answer"] == events[-1].content


class FiveRoundWorkerModel:
    """连续五轮请求工具，第六次调用才返回强制收尾答案。"""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools: list) -> "FiveRoundWorkerModel":
        """测试替身无需真实 schema，直接保留自身。"""
        return self

    async def ainvoke(self, messages: list) -> AIMessage:
        """前五次返回 ToolCall，验证第五批工具不会像普通搜索 Runner 那样被跳过。"""
        self.calls += 1
        if self.calls <= 5:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "tavily_search",
                        "args": {"query": str(self.calls)},
                        "id": f"call-{self.calls}",
                    }
                ],
            )
        return AIMessage(content="强制收尾")


@pytest.mark.asyncio
async def test_worker_executes_fifth_tool_round_before_forced_answer() -> None:
    """研究子任务应执行五轮工具，再进行一次禁用工具语义的最终回答。"""
    model = FiveRoundWorkerModel()
    tool = FakeResearchTool()
    runner = DeepResearchRunner(model, [tool])

    result, references = await runner._execute_worker(PlanTask("task-1", "连续搜索", 1), "无\n")

    assert result.success
    assert result.output == "强制收尾"
    assert len(tool.calls) == 5
    assert len(references) == 5


class StubDeepRunner:
    """用于验证路由参数和旧 SSE 编码，不执行真实模型调用。"""

    async def stream(self, query: str, conversation_id: str) -> AsyncIterator[StreamEvent]:
        """检查路由传参后返回一条报告事件。"""
        assert query == "测试深度研究"
        assert conversation_id == "deep-route"
        yield StreamEvent(type=StreamEventType.TEXT, content="研究报告")


def test_deep_route_preserves_legacy_sse_shape() -> None:
    """旧前端路由应返回 data JSON SSE，并且不自行添加 complete 事件。"""
    original = app.state.deep_runner
    app.state.deep_runner = StubDeepRunner()
    try:
        with TestClient(app) as client:
            response = client.get(
                "/agent/deep/stream",
                params={"query": "测试深度研究", "conversationId": "deep-route"},
            )
    finally:
        app.state.deep_runner = original

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert json.loads(response.text.strip().removeprefix("data: ")) == {
        "type": "text",
        "content": "研究报告",
    }


def test_deep_route_rejects_blank_query() -> None:
    """空白问题应在建立研究流之前返回 400。"""
    with TestClient(app) as client:
        response = client.get(
            "/agent/deep/stream",
            params={"query": "   ", "conversationId": "deep-route"},
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "查询参数不能为空"


class ThinkingReportModel(ScriptedResearchModel):
    """在最终报告阶段分块返回 think 标签，验证 Java 的流式分类语义。"""

    async def astream(self, messages: list) -> AsyncIterator[AIMessageChunk]:
        """报告阶段分开发送推理和正文，其他阶段沿用完整流程替身。"""
        system = str(messages[0].content)
        if "结果总结专家" in system:
            self.inputs.append(list(messages))
            yield AIMessageChunk(content="<think>报告推理")
            yield AIMessageChunk(content="</think># 可见报告")
            return
        async for chunk in super().astream(messages):
            yield chunk


@pytest.mark.asyncio
async def test_report_routes_think_content_without_polluting_saved_answer() -> None:
    """报告推理应进入 thinking，标签外正文进入 text 和数据库 answer。"""
    model = ThinkingReportModel()
    sessions = FakeResearchSessions()
    runner = DeepResearchRunner(model, [FakeResearchTool()], sessions)

    events = [event async for event in runner.stream("研究测试主题", "deep-think")]

    thinking = "".join(event.content for event in events if event.type is StreamEventType.THINKING)
    answer = "".join(event.content for event in events if event.type is StreamEventType.TEXT)
    assert "报告推理" in thinking
    assert answer == "# 可见报告"
    assert sessions.turns[0]["answer"] == answer


class UnpassedResearchModel(ScriptedResearchModel):
    """让三轮评审全部失败，并记录压缩后的最终报告输入。"""

    async def ainvoke(self, messages: list) -> AIMessage:
        """评审始终不通过，其他阶段沿用完整研究替身。"""
        system = str(messages[0].content)
        if "研究评审专家" in system:
            self.inputs.append(list(messages))
            return AIMessage(content='{"passed":false,"feedback":"继续补充"}')
        return await super().ainvoke(messages)


@pytest.mark.asyncio
async def test_three_round_limit_keeps_all_results_after_compression() -> None:
    """三轮均未通过时仍应报告，并且压缩不能删掉已完成任务证据。"""
    model = UnpassedResearchModel()
    runner = DeepResearchRunner(
        model,
        [FakeResearchTool()],
        context_char_limit=1,
    )

    events = [event async for event in runner.stream("研究测试主题", "deep-three-rounds")]

    assert model.plan_calls == 3
    assert any(event.type is StreamEventType.TEXT for event in events)
    report_input = next(
        messages[-1].content
        for messages in reversed(model.inputs)
        if "结果总结专家" in str(messages[0].content)
    )
    assert str(report_input).count("【Completed Task Result】") == 3
