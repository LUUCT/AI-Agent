from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.agents.specs import ReactAgentSpec


class ReactAgentFactory:
    """把项目级 Agent 配置转换为可运行的 LangChain Agent。"""

    def __init__(
        self,
        model: BaseChatModel,
        checkpointer: BaseCheckpointSaver[Any],
    ) -> None:
        self._model = model
        self._checkpointer = checkpointer

    def create(self, spec: ReactAgentSpec) -> Any:
        """输入提示词、工具等配置；输出已绑定模型和 checkpoint 的 Agent。"""
        return create_agent(
            model=self._model,
            tools=list(spec.tools),
            system_prompt=spec.system_prompt,
            checkpointer=self._checkpointer,
            name=spec.name,
        )

