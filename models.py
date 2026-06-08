"""KokoroFlowChatter 数据模型。

定义所有共享数据类型：事件类型枚举、等待配置、工具调用解析结果、备忘录等。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# 控制流常量
KFC_REPLY: str = "kfc_reply"
DO_NOTHING: str = "do_nothing"
PASS_AND_WAIT: str = "pass_and_wait"


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

    has_info_tool: bool = False
    """是否包含 agent-* / tool-* 类工具调用（有实际返回值，需立即续轮让 LLM 看到结果）"""

    has_pass_and_wait: bool = False
    """是否包含 pass_and_wait 调用（完成当前动作后等待）"""

    has_failed_tool: bool = False
    """是否有工具执行失败（用于重试计数）"""

    @property
    def has_meaningful_action(self) -> bool:
        """是否包含任何有效动作（回复、do_nothing、pass_and_wait 或第三方工具）。"""
        return self.has_reply or self.has_do_nothing or self.has_third_party or self.has_pass_and_wait


def extract_visible_reply_text(result: ToolCallResult) -> str:
    """从工具调用结果中提取用户实际看到的回复文本。

    KFC 在 tool-calling 模式下会把 thought、第三方动作、预约等信息
    记录到 MentalLog，但持久化对话链只应保留用户真正看到的回复内容。

    Args:
        result: 工具调用解析结果

    Returns:
        str: 按发送顺序拼接的可见回复文本；无可见回复时返回空串
    """
    segments: list[str] = []

    for action in result.actions:
        if action.get("type") != KFC_REPLY:
            continue

        raw_content = action.get("content")
        if isinstance(raw_content, list):
            segments.extend(
                str(item).strip() for item in raw_content if str(item).strip()
            )
        elif isinstance(raw_content, str):
            stripped = raw_content.strip()
            if stripped:
                segments.append(stripped)

    return "\n".join(segments)


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
    USER_INTERRUPTED = "user_interrupted"
    # 备忘录事件
    MEMO_WRITTEN = "memo_written"
    MEMO_DELETED = "memo_deleted"
    MEMO_EXPIRED = "memo_expired"

    def __str__(self) -> str:
        return self.value


# ── 备忘录 ────────────────────────────────────────────────


# 备忘录配置常量（不暴露为配置项，全部写死）
MEMO_MAX_ENTRIES: int = 10
"""单流最大有效备忘条目数；超出按 created_at 升序淘汰。"""

MEMO_DEFAULT_EXPIRE_HOURS: float = 24.0
"""LLM 不指定 expire_hours 时的默认过期时长。"""

MEMO_MIN_EXPIRE_HOURS: float = 1.0
"""允许的最小过期时长（小于此值会夹到此值）。"""

MEMO_MAX_EXPIRE_HOURS: float = 14 * 24.0
"""允许的最大过期时长（14 天）。超出会夹到此值。"""

MEMO_ID_LENGTH: int = 6
"""备忘 id 的短 hex 长度，便于 LLM 输出。"""


@dataclass
class Memo:
    """单条备忘录。

    设计定位：LLM 显式标记的、带过期时间的关键事项。
    与 mental_log（自动事件流）和 history_summary（叙事压缩）互补，
    覆盖"接下来一段时间需要明确意识到的事"这一语义层。
    """

    memo_id: str = ""
    content: str = ""
    intent: str = ""
    created_at: float = 0.0
    expires_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.memo_id:
            self.memo_id = uuid.uuid4().hex[:MEMO_ID_LENGTH]
        if self.created_at <= 0:
            self.created_at = time.time()

    def is_expired(self, now: float | None = None) -> bool:
        """判断该备忘是否已过期。"""
        current = now if now is not None else time.time()
        return self.expires_at > 0 and current >= self.expires_at

    def remaining_seconds(self, now: float | None = None) -> float:
        """剩余秒数（已过期返回 0）。"""
        current = now if now is not None else time.time()
        return max(0.0, self.expires_at - current)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "memo_id": self.memo_id,
            "content": self.content,
            "intent": self.intent,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Memo:
        """从字典反序列化。"""
        return cls(
            memo_id=str(data.get("memo_id", "") or ""),
            content=str(data.get("content", "") or ""),
            intent=str(data.get("intent", "") or ""),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            expires_at=float(data.get("expires_at", 0.0) or 0.0),
        )


def clamp_expire_hours(raw: float) -> float:
    """把 LLM 给的 expire_hours 夹到 [MEMO_MIN, MEMO_MAX] 范围。"""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return MEMO_DEFAULT_EXPIRE_HOURS
    if value <= 0:
        return MEMO_DEFAULT_EXPIRE_HOURS
    return max(MEMO_MIN_EXPIRE_HOURS, min(value, MEMO_MAX_EXPIRE_HOURS))


@dataclass
class WaitingConfig:
    """等待配置，当 Bot 发送消息后设置的等待参数。"""

    expected_reaction: str = ""
    max_wait_seconds: float = 0.0
    started_at: float = 0.0
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
        self.followup_count = 0

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "expected_reaction": self.expected_reaction,
            "max_wait_seconds": self.max_wait_seconds,
            "started_at": self.started_at,
            "followup_count": self.followup_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WaitingConfig:
        """从字典反序列化。"""
        return cls(
            expected_reaction=data.get("expected_reaction", ""),
            max_wait_seconds=float(data.get("max_wait_seconds", 0)),
            started_at=float(data.get("started_at", 0)),
            followup_count=int(data.get("followup_count", 0)),
        )

