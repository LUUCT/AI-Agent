"""对应 Java EmbeddingService.ragRetrieve 的问题改写和多查询检索。"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# Spring AI 的内部默认提示词不在 Java 项目源码中。这两句是 Python 的显式替代，
# 只复刻“压缩问题”和“生成三条检索问题”的职责，不宣称逐字兼容 Java。
COMPRESSION_PROMPT = "结合已提供的问题，把它改写成一条可独立检索的简短问题。只输出问题文本。"
EXPANSION_PROMPT = "围绕下列问题生成 3 条不同表述的检索问题。只输出 JSON 字符串数组。"


def _message_text(message: object) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


class RagRetriever:
    def __init__(self, model: object, index: object) -> None:
        self._model = model
        self._index = index

    async def retrieve(self, file_id: str, question: str) -> list[str]:
        """输入文件 ID 和用户问题，输出按命中顺序去重后的文本片段。

        Java 先压缩问题，再扩成 3 条查询且保留压缩后的原问题；每条仅在同一
        fileId 内取前 5 个向量结果。Python 也保持此顺序，按稳定块 ID 去重，
        使同一段被多个问法命中时不会重复送入回答模型。
        """
        if not file_id.strip() or not question.strip():
            return ["检索参数不能为空"]
        compressed = _message_text(
            await self._model.ainvoke(
                [SystemMessage(COMPRESSION_PROMPT), HumanMessage(question)]
            )
        ).strip()
        if not compressed:
            compressed = question
        expanded_text = _message_text(
            await self._model.ainvoke(
                [SystemMessage(EXPANSION_PROMPT), HumanMessage(compressed)]
            )
        )
        try:
            expanded = json.loads(expanded_text)
            if not isinstance(expanded, list) or len(expanded) != 3:
                raise ValueError("问题扩展结果应包含 3 条问题")
            queries = [compressed, *(q for q in expanded if isinstance(q, str) and q.strip())]
        except (ValueError, TypeError):
            logger.warning("Multi-query output invalid; retrieving with compressed query")
            queries = [compressed]
        seen: set[str] = set()
        results: list[str] = []
        for query in queries:
            for chunk_id, content in await self._index.search(file_id, query, top_k=5):
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    results.append(content)
        return results
