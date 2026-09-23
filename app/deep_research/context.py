from langchain_core.messages import HumanMessage, SystemMessage

from app.deep_research.model_io import ModelIO
from app.deep_research.models import ResearchState
from app.deep_research.prompts import COMPRESS, current_time


class ContextCompressor:
    """达到字符阈值后压缩规划上下文，保留独立的报告证据。"""

    def __init__(self, model_io: ModelIO, char_limit: int) -> None:
        """保存模型调用入口与 Java 对应的字符阈值。"""
        self._model_io = model_io
        self._context_char_limit = char_limit

    async def compress(self, state: ResearchState) -> bool:
        """达到字符阈值时压缩规划消息；原始任务结果仍由 completed_results 保留。"""
        if state.current_chars() < self._context_char_limit:
            return False
        system = (
            f"{current_time()}\n\n"
            "##最大压缩限制（必须遵守）\n"
            "-你输出的最终内容【总字符数（包含所有标签、空格、换行）】\n"
            f"不得超过：{self._context_char_limit}\n"
            "- 这是硬性上限，不是建议\n- 如超过该限制，视为压缩失败\n\n"
            f"{COMPRESS}"
        )
        snapshot = await self._model_io.invoke_text(
            [SystemMessage(content=system), HumanMessage(content=state.render_full_context())]
        )
        state.messages = [SystemMessage(content=f"【Compressed Agent State】\n{snapshot}")]
        return True
