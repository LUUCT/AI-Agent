import asyncio
import json
import logging
from collections import defaultdict
from collections.abc import AsyncIterator, Sequence
from time import monotonic
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool

from app.deep_research.context import ContextCompressor
from app.deep_research.model_io import ModelIO
from app.deep_research.models import PlanTask, ResearchState, TaskResult
from app.deep_research.planner import Planner
from app.deep_research.reporter import ReportGenerator
from app.deep_research.reviewer import Reviewer
from app.deep_research.search import SearchWorker
from app.models.events import StreamEvent, StreamEventType
from app.persistence.sessions import MySQLSessionRepository
from app.runtime.task_manager import AgentTaskManager

logger = logging.getLogger(__name__)


class DeepResearchRunner:
    """按 Java PlanExecuteAgent 的阶段执行研究，并输出旧前端可消费的事件。"""

    name = "plan-execute"

    def __init__(
        self,
        model: Any,
        tools: Sequence[BaseTool],
        session_repository: MySQLSessionRepository | None = None,
        task_manager: AgentTaskManager | None = None,
        *,
        max_rounds: int = 3,
        worker_max_rounds: int = 5,
        context_char_limit: int = 50_000,
        task_concurrency: int = 3,
    ) -> None:
        """注入模型、动态搜索工具和共享基础设施；数值默认值与 Java 构造配置一致。"""
        self._model_io = ModelIO(model, tools)
        self._worker = SearchWorker(self._model_io, tools, worker_max_rounds)
        self._planner = Planner(self._model_io, tools)
        self._reviewer = Reviewer(self._model_io)
        self._compressor = ContextCompressor(self._model_io, context_char_limit)
        self._reporter = ReportGenerator(self._model_io)
        self._session_repository = session_repository
        self._task_manager = task_manager
        self._max_rounds = max_rounds
        self._task_concurrency = task_concurrency
        self._active: dict[str, asyncio.Task[Any]] = {}
        self._sessions: dict[str, list[tuple[str, str]]] = defaultdict(list)

    async def stop(self, conversation_id: str) -> bool:
        """停止指定会话的研究任务；优先复用 Redis 任务管理器，否则取消本地 Task。"""
        if self._task_manager is not None:
            return await self._task_manager.stop_task(conversation_id)
        task = self._active.get(conversation_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def _execute_worker(
        self, task: PlanTask, dependency_context: str
    ) -> tuple[TaskResult, list[dict[str, str]]]:
        """保留旧测试与调用入口，实际搜索逻辑由 SearchWorker 执行。"""
        return await self._worker.execute_worker(task, dependency_context)

    async def stream(self, query: str, conversation_id: str) -> AsyncIterator[StreamEvent]:
        """按澄清、主题、规划搜索评审、报告的顺序输出研究事件。

        每轮任务按 order 分批执行，评审未通过时继续下一轮；流内保存已产生的
        thinking、text 和来源，取消或异常时仍尝试保存并释放会话任务登记。
        """
        if not query.strip():
            yield StreamEvent(type=StreamEventType.ERROR, message="查询参数不能为空")
            return
        running_task = asyncio.current_task()
        if running_task is None:
            raise RuntimeError("DeepResearchRunner requires an asyncio task")

        # 第一阶段：登记会话任务，确保同一 conversationId 不会并行写入两份研究结果。
        try:
            if self._task_manager is None:
                registered = conversation_id not in self._active
                if registered:
                    self._active[conversation_id] = running_task
            else:
                registered = await self._task_manager.register_task(
                    conversation_id, running_task, self.name
                )
        except Exception:
            logger.exception("Failed to register deep research: %s", conversation_id)
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message="任务登记失败，请检查 Redis 连接。",
                code="TASK_REGISTRATION_ERROR",
            )
            return
        if not registered:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message="该会话正在执行中，请稍后再试",
                code="CONVERSATION_BUSY",
            )
            return

        started = monotonic()
        record_id: int | None = None
        answer = ""
        thinking = ""
        references: list[dict[str, str]] = []
        task_semaphore = asyncio.Semaphore(self._task_concurrency)
        first_response_time: int | None = None
        last_saved: tuple[str, str, int] | None = None

        def record(event: StreamEvent) -> StreamEvent:
            """累计已向客户端发送的文本，确保数据库快照与可见流一致且不重复。"""
            nonlocal answer, thinking, first_response_time
            if first_response_time is None:
                first_response_time = int((monotonic() - started) * 1000)
            if event.type is StreamEventType.TEXT:
                answer += event.content
            elif event.type is StreamEventType.THINKING:
                thinking += event.content
            return event

        async def save_result() -> None:
            """保存当前答案、进度和来源；相同快照跳过重复数据库更新。"""
            nonlocal last_saved
            snapshot = (answer, thinking, len(references))
            if snapshot == last_saved or (not answer and not thinking):
                return
            if self._session_repository is None:
                turns = self._sessions[conversation_id]
                if turns:
                    turns[-1] = (query, answer)
                last_saved = snapshot
                return
            if record_id is None:
                return
            reference: str | None = None
            if references:
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
                tools="",
                reference=reference,
                recommend=None,
                first_response_time=first_response_time,
                total_response_time=int((monotonic() - started) * 1000),
            )
            last_saved = snapshot

        try:
            # 第二阶段：按 Java 的 30 消息窗口重建上下文，再创建本轮空会话记录。
            if self._session_repository is None:
                prior = self._sessions[conversation_id][-30:]
                self._sessions[conversation_id].append((query, ""))
            else:
                prior = await self._session_repository.recent(conversation_id, 30)
                record_id = await self._session_repository.create(
                    conversation_id, query, agent_type=None
                )
            history: list[BaseMessage] = []
            for old_question, old_answer in prior:
                history.append(HumanMessage(content=old_question))
                if old_answer:
                    history.append(AIMessage(content=old_answer))
            state = ResearchState(conversation_id, query, messages=history[-30:])
            state.messages.append(HumanMessage(content=query))

            # 第三阶段：澄清仅决定继续或暂停；模型的可见说明都作为 thinking 输出。
            yield record(
                StreamEvent(
                    type=StreamEventType.THINKING,
                    content="\n🔍 正在分析您的需求...\n",
                )
            )
            clarification_parts: list[str] = []
            clarification_messages = self._planner.clarification_messages(state)
            async for is_thinking, content in self._model_io.stream_segments(
                clarification_messages
            ):
                yield record(StreamEvent(type=StreamEventType.THINKING, content=content))
                if not is_thinking:
                    clarification_parts.append(content)
            clarification = "".join(clarification_parts)
            yield record(StreamEvent(type=StreamEventType.THINKING, content="\n✅ 需求分析完成\n"))
            if "【需要补充信息】" in clarification:
                pause = "⏸【暂停深入研究】" + clarification.replace("【需要补充信息】", "").strip()
                yield record(StreamEvent(type=StreamEventType.TEXT, content=pause))
                return

            # 第四阶段：生成后续规划使用的研究主题，不把主题误作最终答案。
            yield record(
                StreamEvent(
                    type=StreamEventType.THINKING,
                    content="✅ 信息充足，准备生成研究主题\n📝 正在生成研究主题...\n",
                )
            )
            topic_parts: list[str] = []
            topic_messages = self._planner.topic_messages(state)
            async for is_thinking, content in self._model_io.stream_segments(topic_messages):
                yield record(StreamEvent(type=StreamEventType.THINKING, content=content))
                if not is_thinking:
                    topic_parts.append(content)
            state.refined_topic = "".join(topic_parts)
            yield record(
                StreamEvent(type=StreamEventType.THINKING, content="\n✅ 研究主题已生成\n\n")
            )

            # 第五阶段：最多三轮规划、批次执行和评审；结果按计划顺序合并到状态。
            for round_number in range(1, self._max_rounds + 1):
                state.round = round_number
                yield record(
                    StreamEvent(
                        type=StreamEventType.THINKING,
                        content=f"\n🔄 第 {round_number} 轮研究开始\n📋 正在生成执行计划...\n",
                    )
                )
                plan = await self._planner.create_plan(state)
                yield record(
                    StreamEvent(
                        type=StreamEventType.THINKING,
                        content=f"\n✅ 执行计划已生成，共 {len(plan)} 个任务\n",
                    )
                )
                if plan:
                    table = "\n📋 执行计划表：\n" + "".join(
                        f"  🟠 {item.instruction} \n" for item in plan
                    )
                    yield record(StreamEvent(type=StreamEventType.THINKING, content=table))
                if not plan or all(item.id is None for item in plan):
                    break

                yield record(
                    StreamEvent(
                        type=StreamEventType.THINKING,
                        content="\n--- 开始执行任务 ---\n\n",
                    )
                )
                round_results: dict[str, TaskResult] = {}
                accumulated: dict[str, str] = {}
                for order in sorted({item.order for item in plan}):
                    tasks = [item for item in plan if item.order == order and item.id]
                    dependency = self._planner.dependency_context(accumulated, plan, order)
                    for item in tasks:
                        yield record(
                            StreamEvent(
                                type=StreamEventType.THINKING,
                                content=f"⚙️ 正在执行任务 {item.id} : {item.instruction}\n",
                            )
                        )
                    completed = await asyncio.gather(
                        *(self._worker.run_task(item, dependency, task_semaphore) for item in tasks)
                    )
                    for item, (result, found) in zip(tasks, completed, strict=True):
                        round_results[item.id or ""] = result
                        references.extend(found)
                        if result.success and result.output is not None:
                            accumulated[item.id or ""] = result.output
                            yield record(
                                StreamEvent(
                                    type=StreamEventType.THINKING,
                                    content=f"执行结果: {result.output}\n\n",
                                )
                            )
                        else:
                            yield record(
                                StreamEvent(
                                    type=StreamEventType.THINKING,
                                    content=f"\n❌ 任务 {item.id} 执行失败: {result.error}\n\n",
                                )
                            )
                        state.add_task_result(result)
                yield record(
                    StreamEvent(
                        type=StreamEventType.THINKING,
                        content="\n--- 任务执行完成 ---\n\n🔍 正在评估当前研究结果...\n",
                    )
                )
                critique = await self._reviewer.critique(state, plan, round_results)
                if critique.passed:
                    yield record(
                        StreamEvent(
                            type=StreamEventType.THINKING,
                            content="\n✅ 研究结果评估通过，准备生成最终报告\n",
                        )
                    )
                    break
                yield record(
                    StreamEvent(
                        type=StreamEventType.THINKING,
                        content=f"\n⚠️ 研究结果评估未通过，原因分析：{critique.feedback}\n",
                    )
                )
                state.messages.append(
                    AIMessage(content=f"【Critique Feedback】\n{critique.feedback}")
                )
                yield record(
                    StreamEvent(
                        type=StreamEventType.THINKING,
                        content="\n--- 准备进入下一轮迭代 ---\n",
                    )
                )
                if await self._compressor.compress(state):
                    yield record(
                        StreamEvent(
                            type=StreamEventType.THINKING,
                            content="📦 上下文过长，正在压缩...\n✅ 上下文压缩完成\n",
                        )
                    )

            # 第六阶段：只以原问题、主题和真实任务结果生成报告，随后发送来源列表。
            yield record(
                StreamEvent(
                    type=StreamEventType.THINKING,
                    content="\n✅ 研究阶段完成，准备生成最终报告\n\n📝 正在生成最终研究报告...\n\n",
                )
            )
            async for is_thinking, content in self._reporter.stream(state):
                event_type = StreamEventType.THINKING if is_thinking else StreamEventType.TEXT
                yield record(StreamEvent(type=event_type, content=content))
            await save_result()
            if references:
                yield StreamEvent(
                    type=StreamEventType.REFERENCE,
                    content=json.dumps(references, ensure_ascii=False),
                    count=len(references),
                )
        except asyncio.CancelledError:
            yield record(StreamEvent(type=StreamEventType.TEXT, content="⏹ 用户已停止生成\n"))
        except Exception:
            logger.exception("Deep research failed: conversation_id=%s", conversation_id)
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message="深度研究执行失败，请稍后重试。",
                code="DEEP_RESEARCH_ERROR",
            )
        finally:
            # 所有出口都保存已有内容，并释放本地与 Redis 中的会话任务登记。
            try:
                await save_result()
            except Exception:
                logger.exception("Failed to save deep research result: %s", conversation_id)
            if self._task_manager is None:
                if self._active.get(conversation_id) is running_task:
                    del self._active[conversation_id]
            else:
                try:
                    await self._task_manager.unregister_task(conversation_id, running_task)
                except Exception:
                    logger.exception("Failed to remove deep research task: %s", conversation_id)
