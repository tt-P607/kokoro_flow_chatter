"""KFC 内部决策协议。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolCallSpec:
    """规范化后的工具调用描述。"""

    name: str
    call_id: str = ""
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProactiveSchedule:
    """模型预约的下一次主动发起计划。"""

    delay_minutes: float
    reason: str = ""


@dataclass(slots=True)
class Decision:
    """KFC 统一内部决策对象。"""

    thought: str = ""
    mood: str = ""
    expected_reaction: str = ""
    wait_seconds: float = 0.0
    actions: list[dict[str, Any]] = field(default_factory=list)
    visible_reply_segments: list[str] = field(default_factory=list)
    has_reply_action: bool = False
    chose_silence: bool = False
    has_meaningful_action: bool = False
    has_info_tool_calls: bool = False
    third_party_calls: list[ToolCallSpec] = field(default_factory=list)
    proactive_schedule: ProactiveSchedule | None = None

    @property
    def should_reply(self) -> bool:
        """是否产生了回复动作。"""
        return self.has_reply_action

    @property
    def should_wait(self) -> bool:
        """是否应继续等待对方回复。"""
        return self.wait_seconds > 0

    @property
    def should_end_turn(self) -> bool:
        """当前回合是否可以直接收口。"""
        return self.has_meaningful_action and not self.should_wait and not self.has_info_tool_calls

    @property
    def has_third_party_calls(self) -> bool:
        """是否包含第三方工具调用。"""
        return bool(self.third_party_calls)

    @property
    def reply_text(self) -> str:
        """按发送顺序拼接用户可见回复文本。"""
        return "\n".join(self.visible_reply_segments)