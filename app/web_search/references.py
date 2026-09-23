import json
from typing import Any


def parse_tavily_references(raw: Any) -> list[dict[str, str]]:
    """从 Tavily 工具返回中提取来源；输入为嵌套结果，输出为 url/title/content 列表。"""
    # 与 Java 一致，只读取首个内容块的 text.results；解析失败不阻断主回答。
    if hasattr(raw, "content"):
        raw = raw.content
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return []
    if not isinstance(raw, list) or not raw or not isinstance(raw[0], dict):
        return []
    payload = raw[0].get("text")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            return []
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return []
    # 前端来源卡片只需要这三个字段；无 URL 的结果不能作为可点击来源。
    references: list[dict[str, str]] = []
    for item in payload["results"]:
        if not isinstance(item, dict) or not str(item.get("url") or "").strip():
            continue
        references.append(
            {
                "url": str(item["url"]),
                "title": str(item.get("title") or ""),
                "content": str(item.get("content") or ""),
            }
        )
    return references
