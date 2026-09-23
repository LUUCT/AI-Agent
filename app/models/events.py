import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class StreamEventType(StrEnum):
    TEXT = "text"
    THINKING = "thinking"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    REFERENCE = "reference"
    RECOMMEND = "recommend"
    ERROR = "error"
    COMPLETE = "complete"


class StreamEvent(BaseModel):
    """统一的前端事件载体；不同 runner 先构造事件，再由 API 编码为 SSE。"""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: StreamEventType
    content: str = ""
    tool_name: str | None = Field(default=None, serialization_alias="toolName")
    tool_call_id: str | None = Field(default=None, serialization_alias="toolCallId")
    count: int | None = None
    message: str | None = None
    detail: str | None = None
    code: str | None = None

    def to_sse(self) -> str:
        """输出一条 SSE 消息：data 行放 JSON，末尾空行表示本条消息结束。"""
        payload = json.dumps(
            self.model_dump(mode="json", by_alias=True, exclude_none=True),
            ensure_ascii=False,
        )
        return f"data: {payload}\n\n"
