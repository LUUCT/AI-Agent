from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage

from app.deep_research.parsing import _content_text


@dataclass(frozen=True, slots=True)
class PlanTask:
    """表示 Java `PlanTask` 的三个字段；order 相同的任务属于同一并发批次。"""

    id: str | None
    instruction: str
    order: int


@dataclass(frozen=True, slots=True)
class TaskResult:
    """保存一个搜索子任务的最终文本或错误，供评审和最终报告读取。"""

    task_id: str
    success: bool
    output: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CritiqueResult:
    """表示当前轮研究是否足够，以及未通过时下一轮规划要处理的反馈。"""

    passed: bool
    feedback: str


@dataclass(slots=True)
class ResearchState:
    """保存一次请求内的研究状态；结构化任务结果额外保留以避免压缩后丢失证据。"""

    conversation_id: str
    question: str
    messages: list[BaseMessage] = field(default_factory=list)
    round: int = 0
    refined_topic: str | None = None
    completed_results: list[str] = field(default_factory=list)

    def render_full_context(self) -> str:
        """渲染规划上下文，并像 Java 一样只保留最近一次 Critique Feedback。"""
        last_critique = -1
        for index in range(len(self.messages) - 1, -1, -1):
            if "【Critique Feedback】" in _content_text(self.messages[index].content):
                last_critique = index
                break
        parts: list[str] = []
        for index, message in enumerate(self.messages):
            text = _content_text(message.content)
            if index < last_critique and "【Critique Feedback】" in text:
                continue
            parts.append(f"\n\n[{message.type.upper()}]\n\n{text}")
        return "".join(parts)

    def current_chars(self) -> int:
        """返回当前规划消息的字符数；该阈值对应 Java 字符计数而非 token 数。"""
        return sum(len(_content_text(message.content)) for message in self.messages)

    def add_task_result(self, result: TaskResult) -> None:
        """把任务结果同时写入规划消息和不可变报告输入，避免压缩清空原始结果。"""
        block = [
            "【Completed Task Result】",
            f"taskId: {result.task_id}",
            f"success: {str(result.success).lower()}",
        ]
        if result.output is not None:
            block.extend(("result:", result.output))
        if result.error is not None:
            block.extend(("error:", result.error))
        block.append("【End Task Result】")
        text = "\n".join(block)
        self.messages.append(AIMessage(content=text))
        self.completed_results.append(text)
