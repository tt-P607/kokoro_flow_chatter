"""KokoroFlowChatter 数据模型。

定义所有共享数据类型：事件类型枚举、等待配置、工具调用解析结果等。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# 控制流常量
KFC_REPLY: str = "kfc_reply"
DO_NOTHING: str = "do_nothing"


@dataclass
class ToolCallResult:
    """原生 Tool Calling 的结构化解析结果。

    将 LLM 返回的 call_list 解析后的散装变量集中封装，
    作为 ``_parse_tool_calls()`` 的返回值在主循环中传递。
    """

    thought: str = ""
    """LLM 内心想法（来自 kfc_reply / do_nothing 的 thought 参数）"""

    expected_reaction: str = ""
    """LLM 预期对方的反应"""

    max_wait_seconds: float = 0.0
    """LLM 愿意等待的最长时间（秒），0 表示不等待"""

    mood: str = ""
    """LLM 当前心情"""

    actions: list[dict[str, Any]] = field(default_factory=list)
    """动作列表，每项包含 type + 对应参数"""

    has_reply: bool = False
    """是否包含 kfc_reply 调用"""

    has_do_nothing: bool = False
    """是否包含 do_nothing 调用"""

    has_third_party: bool = False
    """是否包含第三方工具调用"""

    @property
    def has_meaningful_action(self) -> bool:
        """是否包含任何有效动作（回复、do_nothing 或第三方工具）。"""
        return self.has_reply or self.has_do_nothing or self.has_third_party


class KFCEventType(Enum):
    """活动流事件类型，用于标记 mental_log 中不同类型的事件。"""

    USER_MESSAGE = "user_message"
    BOT_PLANNING = "bot_planning"
    WAITING_START = "waiting_start"
    WAITING_UPDATE = "waiting_update"
    REPLY_IN_TIME = "reply_in_time"
    REPLY_LATE = "reply_late"
    WAIT_TIMEOUT = "wait_timeout"
    PROACTIVE_TRIGGER = "proactive_trigger"

    def __str__(self) -> str:
        return self.value


@dataclass
class WaitingConfig:
    """等待配置，当 Bot 发送消息后设置的等待参数。"""

    expected_reaction: str = ""
    max_wait_seconds: float = 0.0
    started_at: float = 0.0
    last_thinking_at: float = 0.0
    thinking_count: int = 0
    followup_count: int = 0

    def is_active(self) -> bool:
        """是否正在等待。"""
        return self.max_wait_seconds > 0 and self.started_at > 0

    def get_elapsed_seconds(self) -> float:
        """获取已等待时间（秒）。"""
        if not self.is_active():
            return 0.0
        return time.time() - self.started_at

    def is_timeout(self) -> bool:
        """是否已超时。"""
        if not self.is_active():
            return False
        return self.get_elapsed_seconds() >= self.max_wait_seconds

    def get_progress(self) -> float:
        """获取等待进度 (0.0~1.0)。"""
        if not self.is_active() or self.max_wait_seconds <= 0:
            return 0.0
        return min(self.get_elapsed_seconds() / self.max_wait_seconds, 1.0)

    def reset(self) -> None:
        """重置等待配置。"""
        self.expected_reaction = ""
        self.max_wait_seconds = 0.0
        self.started_at = 0.0
        self.last_thinking_at = 0.0
        self.thinking_count = 0
        self.followup_count = 0

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "expected_reaction": self.expected_reaction,
            "max_wait_seconds": self.max_wait_seconds,
            "started_at": self.started_at,
            "last_thinking_at": self.last_thinking_at,
            "thinking_count": self.thinking_count,
            "followup_count": self.followup_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WaitingConfig:
        """从字典反序列化。"""
        return cls(
            expected_reaction=data.get("expected_reaction", ""),
            max_wait_seconds=float(data.get("max_wait_seconds", 0)),
            started_at=float(data.get("started_at", 0)),
            last_thinking_at=float(data.get("last_thinking_at", 0)),
            thinking_count=int(data.get("thinking_count", 0)),
            followup_count=int(data.get("followup_count", 0)),
        )

