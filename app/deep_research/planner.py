from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from app.deep_research.model_io import ModelIO
from app.deep_research.models import PlanTask, ResearchState
from app.deep_research.parsing import _json_value
from app.deep_research.prompts import (
    PLAN,
    REQUIREMENT_CLARIFICATION,
    RESEARCH_TOPIC_GENERATION,
    current_time,
)


def _parse_plan(text: str) -> list[PlanTask]:
    """把规划模型的 JSON 数组转换成 PlanTask；缺字段或类型错误会终止本次研究。"""
    value = _json_value(text)
    if not isinstance(value, list):
        raise ValueError("执行计划必须是 JSON 数组")
    tasks: list[PlanTask] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("执行计划中的任务必须是 JSON 对象")
        task_id = item.get("id")
        instruction = item.get("instruction")
        order = item.get("order")
        if task_id is not None and not isinstance(task_id, str):
            raise ValueError("任务 id 必须是字符串或 null")
        if not isinstance(instruction, str) or not isinstance(order, int):
            raise ValueError("任务 instruction/order 类型错误")
        tasks.append(PlanTask(task_id, instruction, order))
    return tasks


class Planner:
    """构造澄清、主题与执行计划输入，并维护批次依赖语义。"""

    def __init__(self, model_io: ModelIO, tools: Sequence[BaseTool]) -> None:
        """保存规划模型和可见工具，以生成包含真实工具说明的计划。"""
        self._model_io = model_io
        self._tools = {tool.name: tool for tool in tools}

    def clarification_messages(self, state: ResearchState) -> list[BaseMessage]:
        """用现有会话构造澄清输入，结果仍由编排器决定是否暂停。"""
        return [
            SystemMessage(content=f"{current_time()}\n\n{REQUIREMENT_CLARIFICATION}"),
            *state.messages,
        ]

    def topic_messages(self, state: ResearchState) -> list[BaseMessage]:
        """用原始问题和会话历史构造研究主题生成输入。"""
        return [
            SystemMessage(content=f"{current_time()}\n\n{RESEARCH_TOPIC_GENERATION}"),
            *state.messages,
            HumanMessage(content=f"<original_question>{state.question}</original_question>"),
        ]

    @staticmethod
    def dependency_context(
        accumulated: dict[str, str], plan: list[PlanTask], current_order: int
    ) -> str:
        """只拼接前一个整数 order 的成功结果，保持 Java 的批次依赖语义。"""
        if current_order == 1:
            return "无\n"
        task_orders = {task.id: task.order for task in plan if task.id is not None}
        selected = [
            f"{task_id}: {output}\n"
            for task_id, output in accumulated.items()
            if task_orders.get(task_id) == current_order - 1
        ]
        return "任务 " + "\n".join(selected) if selected else "无\n"

    def _plan_messages(self, state: ResearchState) -> list[BaseMessage]:
        """构造规划调用的系统和用户消息，包括动态工具描述与最近研究上下文。"""
        tool_description = (
            "\n".join(
                f"- {tool.name}: {getattr(tool, 'description', '')}"
                for tool in self._tools.values()
            )
            or "（当前无可用工具）"
        )
        system = (
            f"{current_time()}\n\n{PLAN}"
            f"\n## 当前上下文\n当前轮次: {state.round}\n"
            f"\n## 可用工具说明（仅用于规划参考）\n{tool_description}\n"
            "\n## 输出格式\n返回字段为 id、instruction、order 的严格 JSON 数组。"
        )
        user = (
            "【研究主题】\n"
            f"{state.refined_topic or state.question}\n\n"
            "【对话历史】\n"
            f"{state.render_full_context()}\n\n"
            "## 重要约束\n"
            "如果会话历史中存在【Critique Feedback】，你必须：\n"
            "1. 仔细分析反馈中指出的不足\n"
            "2. 新的计划必须直接解决这些问题\n"
            "3. 不要重复之前失败的尝试"
        )
        return [SystemMessage(content=system), HumanMessage(content=user)]

    async def create_plan(self, state: ResearchState) -> list[PlanTask]:
        """调用规划模型并解析任务；格式错误沿用原流程向上抛出。"""
        return _parse_plan(await self._model_io.invoke_text(self._plan_messages(state)))
