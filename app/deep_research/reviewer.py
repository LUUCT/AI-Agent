from langchain_core.messages import HumanMessage, SystemMessage

from app.deep_research.model_io import ModelIO
from app.deep_research.models import CritiqueResult, PlanTask, ResearchState, TaskResult
from app.deep_research.parsing import _json_value
from app.deep_research.prompts import CRITIQUE, current_time


def _parse_critique(text: str) -> CritiqueResult:
    """把评审模型的 JSON 对象转换成 CritiqueResult，并拒绝缺少 passed 的结果。"""
    value = _json_value(text)
    if not isinstance(value, dict) or not isinstance(value.get("passed"), bool):
        raise ValueError("评审结果必须包含布尔字段 passed")
    feedback = value.get("feedback")
    return CritiqueResult(value["passed"], feedback if isinstance(feedback, str) else "")


class Reviewer:
    """审查当前轮计划和结果，给下一轮规划提供反馈。"""

    def __init__(self, model_io: ModelIO) -> None:
        """保存模型调用入口；评审不读取搜索工具状态。"""
        self._model_io = model_io

    async def critique(
        self,
        state: ResearchState,
        plan: list[PlanTask],
        results: dict[str, TaskResult],
    ) -> CritiqueResult:
        """仅使用当前轮计划和结果进行评审，返回是否结束外层研究循环。"""
        plan_text = "\n".join(f"- {task.instruction}" for task in plan) or "无"
        result_parts: list[str] = []
        for task_id, result in results.items():
            if result.success and result.output is not None:
                result_parts.append(f"任务 {task_id}: {result.output}")
            elif result.error is not None:
                result_parts.append(f"任务 {task_id}: 执行失败 - {result.error}")
        result_text = "\n\n".join(result_parts) or "无"
        user = (
            f"【用户原始问题】\n{state.question}\n\n"
            f"【研究主题】\n{state.refined_topic or '未生成研究主题'}\n\n"
            f"【当前轮次的执行计划】\n{plan_text}\n\n"
            f"【当前轮次的工具结果】\n{result_text}"
        )
        raw = await self._model_io.invoke_text(
            [
                SystemMessage(content=f"{current_time()}\n\n{CRITIQUE}"),
                HumanMessage(content=user),
            ]
        )
        return _parse_critique(raw)
