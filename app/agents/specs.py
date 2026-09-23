from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ReactAgentSpec:
    """一个 ReAct Agent 的名称、系统提示词、可用工具和最大递归步数。"""

    name: str
    system_prompt: str
    tools: tuple[Any, ...] = ()
    recursion_limit: int = 24

    def __post_init__(self) -> None:
        """创建配置后立刻校验，避免运行时才发现名称或循环上限无效。"""
        if not self.name.strip():
            raise ValueError("Agent name cannot be empty")
        if self.recursion_limit < 2:
            raise ValueError("recursion_limit must be at least 2")

