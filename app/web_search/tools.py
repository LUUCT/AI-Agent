import asyncio
import json
import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class ToolExecutor:
    """按模型返回的工具名调用 MCP 工具，保留原始 call_id。"""

    def __init__(self, tools: Sequence[BaseTool]) -> None:
        """建立工具名索引；未知名称在调用时返回错误消息。"""
        self._tools = {tool.name: tool for tool in tools}

    async def call_tool(self, call: dict[str, Any]) -> ToolMessage:
        """执行一条模型工具调用，输出带原始 call_id 的 ToolMessage 供下一轮模型读取。"""
        name = str(call.get("name") or "")
        call_id = str(call.get("id") or "")
        tool = self._tools.get(name)
        if tool is None:
            result: Any = {"error": f"工具未找到：{name}"}
        else:
            try:
                result = await tool.ainvoke(call.get("args") or {})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("MCP tool failed: %s", name, exc_info=True)
                result = {"error": f"tool execution failed: {exc}"}
        # 工具失败也转为结果消息，让模型在下一轮解释错误或改用其他方式回答。
        if isinstance(result, ToolMessage):
            content = result.content
        elif isinstance(result, str):
            content = result
        else:
            content = json.dumps(result, ensure_ascii=False, default=str)
        return ToolMessage(content=content, name=name, tool_call_id=call_id)
