"""文件问答 ReAct 循环；复用 LangChain 模型/工具协议和现有 SSE 事件。"""

import asyncio
import logging
from collections.abc import AsyncIterator
from time import monotonic
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from app.models.events import StreamEvent, StreamEventType
from app.rag.prompts import (
    FORCE_FINAL_PROMPT,
    LOAD_CONTENT_DESCRIPTION,
    file_system_prompt,
)
from app.web_search.model_io import _answer_events, _text

logger = logging.getLogger(__name__)


class FileChatRunner:
    def __init__(
        self,
        model: Any,
        files: Any,
        sessions: Any,
        task_manager: Any | None = None,
        max_rounds: int = 24,
    ) -> None:
        self._model = model
        self._files = files
        self._sessions = sessions
        self._task_manager = task_manager
        self._max_rounds = max_rounds
        self._active: dict[str, asyncio.Task] = {}

    async def stop(self, conversation_id: str) -> bool:
        """输入会话 ID，取消当前文件问答；输出是否真的找到运行任务。"""
        if self._task_manager is not None:
            return await self._task_manager.stop_task(conversation_id)
        task = self._active.get(conversation_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def _round(self, model: Any, messages: list) -> AIMessage:
        """收集一次模型流的内容和 ToolCall，供下一步决定是执行工具还是作答。"""
        merged: AIMessageChunk | None = None
        async for chunk in model.astream(messages):
            if not isinstance(chunk, AIMessageChunk):
                raise TypeError("模型流未返回 AIMessageChunk")
            merged = chunk if merged is None else merged + chunk
        return (
            AIMessage(content=merged.content, tool_calls=merged.tool_calls)
            if merged is not None
            else AIMessage(content="")
        )

    async def stream(
        self, query: str, conversation_id: str, file_id: str
    ) -> AsyncIterator[StreamEvent]:
        """输入问题/会话/文件 ID，按旧前端 SSE 格式输出文件问答。

        关键流程：先验证文件→读取同文件历史→创建会话记录→让模型决定
        loadContent 调用→把工具结果送回模型→输出最终答案并保存。工具闭包只
        绑定本次请求的 fileId，模型即使传入别的 ID 也读不到其他文件。
        """
        if not query.strip() or not file_id.strip():
            yield StreamEvent(type=StreamEventType.ERROR, message="查询参数或文件ID不能为空")
            return
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("FileChatRunner requires asyncio task")
        try:
            if self._task_manager is not None:
                registered = await self._task_manager.register_task(conversation_id, task, "file")
            else:
                registered = conversation_id not in self._active
                if registered:
                    self._active[conversation_id] = task
        except Exception:
            logger.exception("Failed to register file Agent task")
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message="任务登记失败，请检查 Redis 连接。",
                code="TASK_REGISTRATION_ERROR",
            )
            return
        if not registered:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message="该会话已有正在执行的任务",
                code="CONVERSATION_BUSY",
            )
            return

        started = monotonic()
        record_id: int | None = None
        answer = ""
        thinking = ""
        used_tools: list[str] = []
        first_response_time: int | None = None

        async def save() -> None:
            """把最终或被中断的部分答案保存到 Java 的 ai_session。"""
            if record_id is not None and answer:
                await self._sessions.finish(
                    record_id,
                    answer=answer,
                    thinking=thinking,
                    tools=",".join(used_tools),
                    reference=None,
                    recommend=None,
                    first_response_time=first_response_time,
                    total_response_time=int((monotonic() - started) * 1000),
                )

        try:
            info = await self._files.get(file_id)
            if info.status != "SUCCESS":
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    message=f"文件尚未处理完成，当前状态: {info.status}",
                )
                return
            if hasattr(self._sessions, "recent_file"):
                prior = await self._sessions.recent_file(conversation_id, file_id, 30)
            else:
                prior = await self._sessions.recent(conversation_id, 30)
            history: list = []
            for old_query, old_answer in prior:
                history.append(HumanMessage(old_query))
                if old_answer:
                    history.append(AIMessage(old_answer))
            messages: list = [SystemMessage(file_system_prompt()), *history[-30:]]
            messages += [
                HumanMessage(f"<question>{query}</question>"),
                HumanMessage(f"<fileid>{file_id}</fileid>"),
            ]
            record_id = await self._sessions.create(
                conversation_id, query, agent_type="file", file_id=file_id
            )

            @tool("loadContent", description=LOAD_CONTENT_DESCRIPTION)
            async def load_content(fileId: str, question: str = "") -> str:
                """根据文件ID加载文件内容或进行RAG语义检索。"""
                if fileId != file_id:
                    return "文件ID与当前会话不匹配"
                return await self._files.load_content(file_id, question)

            bound = self._model.bind_tools([load_content])
            for round_number in range(self._max_rounds):
                response = await self._round(bound, messages)
                if not response.tool_calls:
                    if not used_tools:
                        # 原提示词要求先读取文件；模型若忽略工具，服务端补读一次，
                        # 防止没有文件证据的回答直接发给用户。
                        content = await self._files.load_content(file_id, query)
                        messages.append(HumanMessage(content))
                        used_tools.append("loadContent")
                        continue
                    for event in _answer_events(_text(response.content)):
                        if first_response_time is None:
                            first_response_time = int((monotonic() - started) * 1000)
                        if event.type is StreamEventType.TEXT:
                            answer += event.content
                        elif event.type is StreamEventType.THINKING:
                            thinking += event.content
                        yield event
                    break
                messages.append(response)
                for call in response.tool_calls:
                    progress = "📂 正在检索文件内容，请稍等..."
                    thinking += progress
                    yield StreamEvent(type=StreamEventType.THINKING, content=progress)
                    name = str(call.get("name") or "")
                    call_id = str(call.get("id") or "")
                    if name != "loadContent":
                        content = f"工具未找到：{name}"
                    else:
                        args = call.get("args") or {}
                        content = await load_content.ainvoke(args)
                        if name not in used_tools:
                            used_tools.append(name)
                    messages.append(ToolMessage(content=content, name=name, tool_call_id=call_id))
                if round_number == self._max_rounds - 1:
                    messages.append(HumanMessage(FORCE_FINAL_PROMPT))
                    final = await self._round(self._model, messages)
                    for event in _answer_events(_text(final.content)):
                        if event.type is StreamEventType.TEXT:
                            answer += event.content
                        elif event.type is StreamEventType.THINKING:
                            thinking += event.content
                        yield event
        except asyncio.CancelledError:
            answer += "⏹ 用户已停止生成\n"
            yield StreamEvent(type=StreamEventType.TEXT, content="⏹ 用户已停止生成\n")
        except Exception:
            logger.exception("File Agent failed: conversation_id=%s", conversation_id)
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message="文件问答执行失败，请稍后重试。",
                code="FILE_AGENT_ERROR",
            )
        finally:
            try:
                await save()
            except Exception:
                logger.exception("Failed to save file Agent answer: %s", conversation_id)
            if self._task_manager is not None:
                try:
                    await self._task_manager.unregister_task(conversation_id, task)
                except Exception:
                    logger.exception("Failed to unregister file Agent task")
            elif self._active.get(conversation_id) is task:
                del self._active[conversation_id]
