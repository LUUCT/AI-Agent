import re
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.tools import BaseTool

from app.models.events import StreamEvent, StreamEventType


def _text(content: Any) -> str:
    """把模型返回的字符串或文本块列表合成普通文本；非文本内容不进入最终答案。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else str(block.get("text") or "")
            for block in content
            if isinstance(block, (str, dict))
        )
    return ""


def _answer_events(answer: str) -> list[StreamEvent]:
    """把最终文本里的 <think> 标签拆成 thinking/text，供旧前端分别展示。"""
    events: list[StreamEvent] = []
    for part in re.split(r"(<think>.*?</think>)", answer, flags=re.DOTALL):
        if not part:
            continue
        if part.startswith("<think>") and part.endswith("</think>"):
            content = part[len("<think>") : -len("</think>")]
            if content:
                events.append(StreamEvent(type=StreamEventType.THINKING, content=content))
        else:
            events.append(StreamEvent(type=StreamEventType.TEXT, content=part))
    return events


class SearchModel:
    """合并模型流分片，并在收尾调用时关闭工具绑定。"""

    def __init__(self, model: Any, tools: Sequence[BaseTool]) -> None:
        """保存原模型并仅绑定一次 MCP 工具及其动态 schema。"""
        self._model = model
        self._bound_model = model.bind_tools(list(tools))

    async def model_round(self, messages: list[BaseMessage], *, allow_tools: bool) -> AIMessage:
        """输入对话消息，合并流式分片，输出完整回复及工具调用列表。"""
        # 最后收尾时禁用工具，使模型直接回答，不再发起新搜索。
        model = self._bound_model if allow_tools else self._model
        merged: AIMessageChunk | None = None
        # 工具参数可能分散在多个 AIMessageChunk；相加后才能获得完整参数和调用 ID。
        async for chunk in model.astream(messages):
            if not isinstance(chunk, AIMessageChunk):
                raise TypeError("模型流未返回 AIMessageChunk")
            merged = chunk if merged is None else merged + chunk
        if merged is None:
            return AIMessage(content="")
        return AIMessage(content=merged.content, tool_calls=merged.tool_calls)
