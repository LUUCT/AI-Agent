import asyncio
import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.web_search.model_io import _text
from app.web_search.prompts import RECOMMEND_PROMPT, local_time

logger = logging.getLogger(__name__)


class RecommendationGenerator:
    """在回答完成后生成三个与会话连续的问题。"""

    def __init__(self, model: Any) -> None:
        """保存未绑定工具的模型，推荐阶段不执行 MCP 调用。"""
        self._model = model

    async def recommend(
        self,
        history: list[BaseMessage],
        query: str,
        answer: str,
    ) -> list[str] | None:
        """根据历史、当前问题和答案再调用一次模型；成功时输出恰好三个推荐问题。"""
        messages: list[BaseMessage] = [
            SystemMessage(RECOMMEND_PROMPT.format(current_time=local_time()))
        ]
        messages.extend(history)
        messages.extend(
            [
                HumanMessage("当前会话："),
                HumanMessage(query),
                AIMessage(answer),
                HumanMessage("请根据上述对话生成3个推荐问题。输出格式为：\nJSON 字符串数组"),
            ]
        )
        try:
            result = await self._model.ainvoke(messages)
            questions = json.loads(_text(result.content))
            if (
                isinstance(questions, list)
                and len(questions) == 3
                and all(isinstance(item, str) for item in questions)
            ):
                return questions
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Recommendation generation failed", exc_info=True)
        return None
