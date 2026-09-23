from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain.tools import tool


@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """按 IANA 时区返回当前 ISO 时间；时区无效时返回错误文本供模型解释。"""
    try:
        current = datetime.now(ZoneInfo(timezone))
    except ZoneInfoNotFoundError:
        return f"Unknown timezone: {timezone}"
    return current.isoformat(timespec="seconds")

