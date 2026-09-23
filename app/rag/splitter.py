"""逐字移植 Java OverlapParagraphTextSplitter 的 500/50 切块行为。"""

import re


def split_overlap_paragraphs(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """把全文切成向量化小块。

    需求：大文件不能一次送入向量模型。输入是解析后的全文及块大小/重叠数；
    输出是有顺序的文本块。先按换行分段、再填满 500 字符；复制末尾 50 字符
    到下一块，是为了让跨块句子保留上下文。段落之间不额外加换行，与 Java 一致。
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size 必须大于 overlap，且 overlap 不能为负数")
    if not text.strip():
        return []

    chunks: list[str] = []
    current = ""
    for paragraph in re.split(r"\n+", text):
        if not paragraph.strip():
            continue
        start = 0
        while start < len(paragraph):
            remaining = chunk_size - len(current)
            end = min(start + remaining, len(paragraph))
            current += paragraph[start:end]
            if len(current) >= chunk_size:
                chunks.append(current)
                current = current[-overlap:] if overlap else ""
            start = end
    if current:
        chunks.append(current)
    return chunks
