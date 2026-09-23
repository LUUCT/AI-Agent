import asyncio
import json
import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.deep_research.model_io import ModelIO
from app.deep_research.models import PlanTask, TaskResult
from app.deep_research.parsing import _content_text
from app.deep_research.prompts import (
    EXECUTE,
    FORCE_FINAL_PROMPT,
    REACT_AGENT_SYSTEM_PROMPT,
    current_time,
)
from app.web_search.references import parse_tavily_references

logger = logging.getLogger(__name__)


class SearchWorker:
    """执行计划中的搜索任务，并收集工具返回的引用来源。"""

    def __init__(self, model_io: ModelIO, tools: Sequence[BaseTool], max_rounds: int) -> None:
        """保存模型、可用工具与单任务工具调用轮数上限。"""
        self._model_io = model_io
        self._tools = {tool.name: tool for tool in tools}
        self._worker_max_rounds = max_rounds

    async def _call_tool(self, call: dict[str, Any]) -> tuple[ToolMessage, list[dict[str, str]]]:
        """执行一个模型 ToolCall，返回可回填消息及从原始 Tavily 结果解析的来源。"""
        name = str(call.get("name") or "")
        call_id = str(call.get("id") or "")
        tool = self._tools.get(name)
        if tool is None:
            raw: Any = {"error": f"工具未找到：{name}"}
        else:
            try:
                raw = await tool.ainvoke(call.get("args") or {})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Deep research tool failed: %s", name, exc_info=True)
                raw = {"error": f"工具执行失败：{exc}"}
        references = parse_tavily_references(raw)
        if isinstance(raw, ToolMessage):
            content = raw.content
        elif isinstance(raw, str):
            content = raw
        else:
            content = json.dumps(raw, ensure_ascii=False, default=str)
        return ToolMessage(content=content, name=name, tool_call_id=call_id), references

    async def execute_worker(
        self, task: PlanTask, dependency_context: str
    ) -> tuple[TaskResult, list[dict[str, str]]]:
        """执行一个最多五轮的搜索 ReAct 子任务；子任务不读写整场会话记忆。"""
        context = (
            f"【Available Results】\n{dependency_context}\n\n【Current Task】\n{task.instruction}\n"
        )
        messages: list[BaseMessage] = [
            SystemMessage(content=f"{current_time()}\n\n{REACT_AGENT_SYSTEM_PROMPT}\n\n{EXECUTE}"),
            HumanMessage(content=f"<question>{context}</question>"),
        ]
        references: list[dict[str, str]] = []

        # Java 在五轮内仍会执行第五轮工具；超过上限后才追加强制收尾消息。
        for _ in range(self._worker_max_rounds):
            response = await self._model_io.bound_model.ainvoke(messages)
            calls = list(getattr(response, "tool_calls", None) or [])
            if not calls:
                return TaskResult(task.id or "", True, _content_text(response.content)), references
            messages.append(response)
            for call in calls:
                tool_message, found = await self._call_tool(call)
                messages.append(tool_message)
                references.extend(found)

        messages.append(HumanMessage(content=FORCE_FINAL_PROMPT))
        answer = await self._model_io.invoke_text(messages)
        return TaskResult(task.id or "", True, answer), references

    async def run_task(
        self,
        task: PlanTask,
        dependency_context: str,
        semaphore: asyncio.Semaphore,
    ) -> tuple[TaskResult, list[dict[str, str]]]:
        """在请求级三许可限制内执行任务；异常转换为失败结果，不重试 Java 未实现的重试。"""
        async with semaphore:
            try:
                return await self.execute_worker(task, dependency_context)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Deep research task failed: %s", task.id, exc_info=True)
                return TaskResult(task.id or "", False, error=str(exc)), []
