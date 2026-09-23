import json
import re
from typing import Any


def _content_text(content: Any) -> str:
    """把 LangChain 字符串或文本块内容合成普通文本，忽略非文本块。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else str(block.get("text") or "")
            for block in content
            if isinstance(block, (str, dict))
        )
    return ""


def _strip_think(text: str) -> str:
    """移除完整的 think 标签及内容，供 JSON 解析使用。"""
    return re.sub(r"<think[^>]*>.*?</think[^>]*>", "", text, flags=re.DOTALL).strip()


def _parse_think_chunk(chunk: str, in_think: bool) -> tuple[list[tuple[bool, str]], bool]:
    """按 Java ThinkTagParser 规则拆分一个模型片段，并返回延续到下一片的状态。"""
    segments: list[tuple[bool, str]] = []
    index = 0
    while index < len(chunk):
        start_index = chunk.find("<think", index)
        end_index = chunk.find("</think", index)
        if start_index < 0 and end_index < 0:
            segments.append((in_think, chunk[index:]))
            break
        is_start = start_index >= 0 and (end_index < 0 or start_index < end_index)
        tag_index = start_index if is_start else end_index
        if tag_index > index:
            segments.append((in_think, chunk[index:tag_index]))
        tag_end = chunk.find(">", tag_index)
        if tag_end < 0:
            in_think = is_start
            break
        in_think = is_start
        index = tag_end + 1
    return [(thinking, text) for thinking, text in segments if text], in_think


def _json_value(text: str) -> Any:
    """从模型文本解析首个 JSON 数组或对象；解析失败时抛出明确的 ValueError。"""
    cleaned = _strip_think(text)
    fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, flags=re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    starts = [index for index in (cleaned.find("["), cleaned.find("{")) if index >= 0]
    if not starts:
        raise ValueError("模型未返回 JSON")
    try:
        value, _ = json.JSONDecoder().raw_decode(cleaned[min(starts) :])
        return value
    except json.JSONDecodeError as exc:
        raise ValueError("模型返回的 JSON 无法解析") from exc
