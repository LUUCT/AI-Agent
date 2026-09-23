import asyncio
import json
import logging
from collections import defaultdict
from collections.abc import AsyncIterator, Sequence
from time import monotonic
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from app.models.events import StreamEvent, StreamEventType
from app.persistence.sessions import MySQLSessionRepository
from app.runtime.task_manager import AgentTaskManager
from app.web_search.model_io import SearchModel, _answer_events, _text
from app.web_search.prompts import FORCE_FINAL_PROMPT, WEB_SEARCH_PROMPT, local_time
from app.web_search.recommendations import RecommendationGenerator
from app.web_search.references import parse_tavily_references
from app.web_search.tools import ToolExecutor

logger = logging.getLogger(__name__)


class WebSearchRunner:
    """处理需要联网信息的问答：最多五轮模型/工具交互，并兼容 Java 旧前端事件。"""

    name = "web react"
    max_rounds = 5

    def __init__(
        self,
        model: Any,
        tools: Sequence[BaseTool],
        session_repository: MySQLSessionRepository | None = None,
        task_manager: AgentTaskManager | None = None,
    ) -> None:
        """输入模型、MCP 工具，以及可选的会话仓储和任务管理器；创建搜索运行器。"""
        # 模型流、工具执行与推荐分别封装；动态工具 schema 仍只绑定一次。
        self._model_rounds = SearchModel(model, tools)
        self._tool_executor = ToolExecutor(tools)
        self._recommender = RecommendationGenerator(model)
        self._session_repository = session_repository
        self._task_manager = task_manager
        # Used only by unit tests and callers explicitly omitting a repository.
        self._sessions: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._active: dict[str, asyncio.Task[Any]] = {}

    async def stop(self, conversation_id: str) -> bool:
        """输入会话 ID；取消对应异步任务，返回是否找到可停止的任务。"""
        if self._task_manager is not None:
            return await self._task_manager.stop_task(conversation_id)
        task = self._active.get(conversation_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def stream(self, query: str, conversation_id: str) -> AsyncIterator[StreamEvent]:
        """逐条输出联网问答事件，并在结束或取消时保存已有结果。

        先登记任务和恢复历史，再执行至多五轮模型与工具调用；回答完成后
        依次发送来源和推荐问题，最后释放任务登记。
        """
        # 这是异步生成器：每次 yield 一条事件，HTTP 流持续打开直到函数结束。
        if not query.strip():
            yield StreamEvent(type=StreamEventType.ERROR, message="查询参数不能为空")
            return
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("WebSearchRunner requires an asyncio task")
        # 一个会话同一时间只允许一个搜索任务，避免答案和数据库记录相互覆盖。
        try:
            if self._task_manager is None:
                registered = conversation_id not in self._active
                if registered:
                    self._active[conversation_id] = task
            else:
                registered = await self._task_manager.register_task(
                    conversation_id, task, "websearch"
                )
        except Exception:
            logger.exception("Failed to register Agent task: conversation_id=%s", conversation_id)
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
        # 以下变量累计本轮输出，既供最终 SSE 事件使用，也供 ai_session 回填。
        started = monotonic()
        record_id: int | None = None
        references: list[dict[str, str]] = []
        answer = ""
        thinking = ""
        recommendations_json: str | None = None
        first_response_time: int | None = None
        used_tools: list[str] = []
        last_saved: tuple[str, str, str | None] | None = None

        def record_event(event: StreamEvent) -> StreamEvent:
            """记录首字耗时，并把已发送的答案/思考文本累计到外层变量。"""
            # nonlocal 表示修改 stream 中的变量，供后续数据库回填使用。
            nonlocal answer, thinking, first_response_time
            if first_response_time is None:
                first_response_time = int((monotonic() - started) * 1000)
            if event.type is StreamEventType.TEXT:
                answer += event.content
            elif event.type is StreamEventType.THINKING:
                thinking += event.content
            return event

        async def save_result() -> None:
            """将目前已产生的内容写回本轮会话；快照相同则跳过重复写入。"""
            nonlocal last_saved
            if not answer:
                return
            snapshot = (answer, thinking, recommendations_json)
            if snapshot == last_saved:
                return
            if self._session_repository is None:
                self._sessions[conversation_id][-1] = (query, answer)
                last_saved = snapshot
                return
            if record_id is None:
                return
            reference = None
            if references:
                # Java 表的 reference 字段存整条事件 JSON，而非单独的来源数组。
                reference_event = StreamEvent(
                    type=StreamEventType.REFERENCE,
                    content=json.dumps(references, ensure_ascii=False),
                    count=len(references),
                )
                reference = json.dumps(
                    reference_event.model_dump(mode="json", by_alias=True, exclude_none=True),
                    ensure_ascii=False,
                )
            await self._session_repository.finish(
                record_id,
                answer=answer,
                thinking=thinking,
                tools=",".join(used_tools),
                reference=reference,
                recommend=recommendations_json,
                first_response_time=first_response_time,
                total_response_time=int((monotonic() - started) * 1000),
            )
            last_saved = snapshot

        try:
            if self._session_repository is None:
                prior = self._sessions[conversation_id][-30:]
            else:
                prior = await self._session_repository.recent(conversation_id, 30)
            # 数据库取最近 30 条问答记录，再转换为模型认识的 Human/AI 消息。
            history: list[BaseMessage] = []
            for old_query, old_answer in prior:
                history.append(HumanMessage(old_query))
                if old_answer:
                    history.append(AIMessage(old_answer))
            # 这里限制的是消息条数，不是问答轮数：一轮通常占两条消息。
            history = history[-30:]
            messages: list[BaseMessage] = [
                SystemMessage(WEB_SEARCH_PROMPT.format(current_time=local_time()))
            ]
            if history:
                messages.append(HumanMessage("对话历史："))
                messages.extend(history)
            # 与 Java 提示词格式保持一致，用 question 标签标出本轮问题。
            messages.append(HumanMessage(f"<question>{query}</question>"))
            # 先创建本轮空答案记录，后续文本、来源、耗时会回填同一行。
            if self._session_repository is None:
                self._sessions[conversation_id].append((query, ""))
            else:
                record_id = await self._session_repository.create(conversation_id, query)
            # ReAct：模型决定回答还是调用工具；工具结果会成为下一轮输入。
            for round_number in range(1, self.max_rounds + 1):
                response = await self._model_rounds.model_round(messages, allow_tools=True)
                calls = response.tool_calls
                if not calls:
                    for event in _answer_events(_text(response.content)):
                        yield record_event(event)
                    break
                if round_number == self.max_rounds:
                    # 第五轮若仍要求工具，按 Java 规则跳过本批调用，追加收尾提示。
                    # Python 同时取消工具绑定，避免收尾模型继续返回无法执行的调用。
                    messages.append(HumanMessage(FORCE_FINAL_PROMPT))
                    final = await self._model_rounds.model_round(messages, allow_tools=False)
                    for event in _answer_events(_text(final.content)):
                        yield record_event(event)
                    break
                messages.append(response)
                for call in calls:
                    if "search" in str(call.get("name") or "").lower():
                        args = call.get("args") or {}
                        search_query = args.get("query") if isinstance(args, dict) else None
                        progress = (
                            f"🔍 正在搜索信息: {search_query}\n"
                            if search_query
                            else "🔍 正在搜索相关信息\n"
                        )
                        yield record_event(
                            StreamEvent(type=StreamEventType.THINKING, content=progress)
                        )
                # 同轮工具并发执行；gather 的结果顺序与 calls 一致，便于正确配对 call_id。
                results = await asyncio.gather(
                    *(self._tool_executor.call_tool(call) for call in calls)
                )
                for call, result in zip(calls, results, strict=True):
                    messages.append(result)
                    tool_name = str(call.get("name") or "")
                    if tool_name and tool_name not in used_tools:
                        used_tools.append(tool_name)
                    if "tavily" in str(call.get("name") or "").lower():
                        # 来源卡片只从 Tavily 结果提取；工具原文仍留在 messages 供模型总结。
                        references.extend(parse_tavily_references(result.content))
            # 先持久化答案，再按旧前端协议发送来源和可选推荐问题。
            await save_result()
            if references:
                yield StreamEvent(
                    type=StreamEventType.REFERENCE,
                    content=json.dumps(references, ensure_ascii=False),
                    count=len(references),
                )
            recommendations = await self._recommender.recommend(history, query, answer)
            if recommendations:
                recommendations_json = json.dumps(recommendations, ensure_ascii=False)
                await save_result()
                yield StreamEvent(
                    type=StreamEventType.RECOMMEND,
                    content=recommendations_json,
                )
        except asyncio.CancelledError:
            # 停止时保留已生成的部分答案，并给用户一个明确的结束提示。
            yield record_event(StreamEvent(type=StreamEventType.TEXT, content="⏹ 用户已停止生成\n"))
            try:
                await save_result()
            except Exception:
                logger.exception(
                    "Failed to save stopped search: conversation_id=%s", conversation_id
                )
        except Exception:
            logger.exception("Web search failed: conversation_id=%s", conversation_id)
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message="联网搜索执行失败，请稍后重试。",
                code="WEB_SEARCH_ERROR",
            )
        finally:
            # 无论正常结束、停止还是报错，都尝试保存已有文本并释放会话任务锁。
            try:
                await save_result()
            except Exception:
                logger.exception("Failed to save final search result: %s", conversation_id)
            if self._task_manager is None:
                if self._active.get(conversation_id) is task:
                    del self._active[conversation_id]
            else:
                try:
                    await self._task_manager.unregister_task(conversation_id, task)
                except Exception:
                    logger.exception("Failed to remove Agent task: %s", conversation_id)
