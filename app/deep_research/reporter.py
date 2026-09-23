from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage

from app.deep_research.model_io import ModelIO
from app.deep_research.models import ResearchState
from app.deep_research.prompts import SUMMARIZE, current_time


class ReportGenerator:
    """只根据原问题、研究主题和真实任务结果生成最终报告。"""

    def __init__(self, model_io: ModelIO) -> None:
        """保存模型流入口，报告内容由每次请求的研究状态提供。"""
        self._model_io = model_io

    async def stream(self, state: ResearchState) -> AsyncIterator[tuple[bool, str]]:
        """逐段输出最终报告；布尔值区分 think 内容与可见正文。"""
        result_text = "\n\n".join(state.completed_results) or "（未检索到相关结果）"
        report_prompt = (
            f"【用户原始问题】\n{state.question}\n\n"
            f"【研究主题】\n{state.refined_topic or '未生成研究主题'}\n\n"
            f"【工具检索结果】\n{result_text}"
        )
        report_messages = [
            SystemMessage(content=f"{current_time()}\n\n{SUMMARIZE}"),
            HumanMessage(content=report_prompt),
        ]
        async for segment in self._model_io.stream_segments(report_messages):
            yield segment
