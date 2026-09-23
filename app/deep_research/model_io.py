from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from app.deep_research.parsing import _content_text, _parse_think_chunk


class ModelIO:
    """集中处理模型调用和 think 片段拆分，供各研究阶段复用。"""

    def __init__(self, model: Any, tools: Sequence[BaseTool]) -> None:
        """保存原模型并仅绑定一次工具，后续调用沿用同一模型实例。"""
        self.model = model
        self.bound_model = model.bind_tools(list(tools))

    async def stream_segments(self, messages: list[BaseMessage]) -> AsyncIterator[tuple[bool, str]]:
        """流式拆分模型的 think 与普通文本片段，保持 Java 的跨片状态语义。"""
        in_think = False
        async for chunk in self.model.astream(messages):
            segments, in_think = _parse_think_chunk(_content_text(chunk.content), in_think)
            for segment in segments:
                yield segment

    async def invoke_text(self, messages: list[BaseMessage]) -> str:
        """使用未绑定工具的模型执行一次非流式调用，返回回复文本。"""
        response = await self.model.ainvoke(messages)
        return _content_text(response.content)
