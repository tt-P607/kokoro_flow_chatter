"""KokoroFlowChatter 数据模型。

定义所有共享数据类型：事件类型枚举、等待配置、策略结果等。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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


@dataclass
class StrategyResult:
    """策略解析 LLM 响应后的结构化结果。"""

    thought: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    expected_reaction: str = ""
    max_wait_seconds: float = 0.0
    mood: str = ""

    @classmethod
    def create_error(cls, error_message: str) -> StrategyResult:
        """创建错误结果。"""
        return cls(
            thought=f"出现了问题：{error_message}",
            actions=[{"type": "do_nothing"}],
        )

    def has_reply(self) -> bool:
        """是否包含回复动作。"""
        return any(
            a.get("type") in ("kfc_reply", "respond") for a in self.actions
        )

    def get_reply_content(self) -> str:
        """获取回复内容。"""
        for action in self.actions:
            if action.get("type") in ("kfc_reply", "respond"):
                return action.get("content", "")
        return ""

    def get_actions_summary(self) -> str:
        """获取动作摘要。"""
        descriptions: list[str] = []
        for action in self.actions:
            action_type = action.get("type", "unknown")
            if action_type in ("kfc_reply", "respond"):
                content = action.get("content", "")
                descriptions.append(
                    f'发送消息："{content[:50]}{"..." if len(content) > 50 else ""}"'
                )
            elif action_type == "do_nothing":
                descriptions.append("什么都没做")
            else:
                descriptions.append(f"执行动作：{action_type}")
        return " + ".join(descriptions)
