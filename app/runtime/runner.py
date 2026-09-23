import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any, Protocol

from app.agents.specs import ReactAgentSpec
from app.models.events import StreamEvent, StreamEventType

logger = logging.getLogger(__name__)


class AgentStreamRunner(Protocol):
    """运行器统一接口：输入问题与会话 ID，异步产出前端事件。"""

    async def stream(self, query: str, conversation_id: str) -> AsyncIterator[StreamEvent]: ...


class ReactAgentRunner:
    """执行通用 ReAct Agent，并把框架原始事件转换为本项目的前端事件。"""

    def __init__(self, agent: Any, spec: ReactAgentSpec) -> None:
        self._agent = agent
        self._spec = spec

    async def stream(self, query: str, conversation_id: str) -> AsyncIterator[StreamEvent]:
        """输入用户问题与会话 ID；输出模型文本、工具状态及完成/错误事件。"""
        yield StreamEvent(
            type=StreamEventType.THINKING,
            content="正在分析问题并选择合适的工具...\n",
        )

        # LangGraph 用 thread_id 将同一会话的多次请求关联到 checkpoint。
        config = {
            "configurable": {"thread_id": conversation_id},
            "recursion_limit": self._spec.recursion_limit,
        }

        try:
            async for raw_event in self._agent.astream_events(
                {"messages": [{"role": "user", "content": query}]},
                config=config,
                version="v2",
            ):
                # 框架事件结构属于内部实现；先转换，再交给 API 输出 SSE。
                event = self._translate_event(raw_event)
                if event is not None:
                    yield event
        except asyncio.CancelledError:
            logger.info("Agent run cancelled: conversation_id=%s", conversation_id)
            raise
        except Exception:
            logger.exception("Agent run failed: conversation_id=%s", conversation_id)
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message="Agent 执行失败，请稍后重试。",
                code="AGENT_EXECUTION_ERROR",
            )

        yield StreamEvent(type=StreamEventType.COMPLETE)

    @classmethod
    def _translate_event(cls, raw_event: dict[str, Any]) -> StreamEvent | None:
        """把模型分片和工具开始/结束事件映射到统一事件；无关事件返回 None。"""
        event_name = raw_event.get("event")
        data = raw_event.get("data") or {}

        if event_name == "on_chat_model_stream":
            chunk = data.get("chunk")
            content = cls._extract_text(getattr(chunk, "content", ""))
            if content:
                return StreamEvent(type=StreamEventType.TEXT, content=content)

        if event_name == "on_tool_start":
            return StreamEvent(
                type=StreamEventType.TOOL_START,
                tool_name=str(raw_event.get("name") or "unknown"),
                tool_call_id=str(raw_event.get("run_id") or ""),
            )

        if event_name == "on_tool_end":
            return StreamEvent(
                type=StreamEventType.TOOL_END,
                tool_name=str(raw_event.get("name") or "unknown"),
                tool_call_id=str(raw_event.get("run_id") or ""),
            )

        return None

    @staticmethod
    def _extract_text(content: Any) -> str:
        """兼容字符串和结构化文本块，提取可直接显示给用户的文本。"""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""

        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            if block.get("type") not in {"text", "output_text"}:
                continue
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)
            elif isinstance(text, dict) and isinstance(text.get("value"), str):
                text_parts.append(text["value"])
        return "".join(text_parts)


class UnavailableAgentRunner:
    """缺少运行配置时返回固定错误事件，避免请求直接抛出初始化异常。"""

    def __init__(self, message: str) -> None:
        self._message = message

    async def stream(self, query: str, conversation_id: str) -> AsyncIterator[StreamEvent]:
        del query, conversation_id
        yield StreamEvent(
            type=StreamEventType.ERROR,
            message=self._message,
            code="AGENT_NOT_CONFIGURED",
        )
        yield StreamEvent(type=StreamEventType.COMPLETE)
