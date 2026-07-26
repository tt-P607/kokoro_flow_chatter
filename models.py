"""KokoroFlowChatter 共享数据模型。

本模块定义跨层复用的基础数据类型，不依赖 KFC 内部任何其他模块：

- 控制动作名常量（``KFC_REPLY`` / ``DO_NOTHING`` / ``PASS_AND_WAIT``）
- ``KFCEventType``：心理活动流事件类型枚举
- ``ToolCallResult``：执行层产出的原始工具执行汇总
- ``Memo`` 及其边界常量：LLM 自主管理的私人备忘录
- ``WaitingConfig``：一次"发完消息后等待回复"的状态快照
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── 控制动作名常量 ────────────────────────────────────────
# 这三个动作由 KFC 执行层特殊解释（不走框架 run_tool_call），
# 其余 action-/tool-/agent- 一律交给框架调度。

KFC_REPLY: str = "kfc_reply"
"""发送文本消息给对方。"""

DO_NOTHING: str = "do_nothing"
"""主动选择不回复。"""

PASS_AND_WAIT: str = "pass_and_wait"
"""完成本轮动作后登记等待点。"""


# ── 事件类型 ──────────────────────────────────────────────


class KFCEventType(Enum):
    """心理活动流事件类型，标记 ``MentalLog`` 中每条记录的语义。"""

    USER_MESSAGE = "user_message"
    BOT_PLANNING = "bot_planning"
    WAITING_START = "waiting_start"
    REPLY_IN_TIME = "reply_in_time"
    REPLY_LATE = "reply_late"
    WAIT_TIMEOUT = "wait_timeout"
    PROACTIVE_TRIGGER = "proactive_trigger"
    USER_INTERRUPTED = "user_interrupted"
    MEMO_WRITTEN = "memo_written"
    MEMO_DELETED = "memo_deleted"
    MEMO_EXPIRED = "memo_expired"

    def __str__(self) -> str:
        """序列化时直接使用字符串值。"""
        return self.value


# ── 工具执行结果 ──────────────────────────────────────────


@dataclass
class ToolCallResult:
    """执行层对一轮 tool call 的执行结果汇总。

    由 ``execute_decision_draft()`` 产出，随后被 ``build_decision()``
    收敛为面向主循环的 :class:`~..domain.decision.Decision`。
    """

    thought: str = ""
    """LLM 内心想法（来自控制动作的 ``thought`` 参数）。"""

    expected_reaction: str = ""
    """LLM 预期对方会有的反应。"""

    max_wait_seconds: float = 0.0
    """LLM 愿意等待的最长时间（秒），0 表示不等待。"""

    mood: str = ""
    """LLM 当前心情。"""

    actions: list[dict[str, Any]] = field(default_factory=list)
    """已执行动作列表，每项含 ``type`` 及其余参数。"""

    has_reply: bool = False
    """是否执行了 ``kfc_reply``。"""

    has_do_nothing: bool = False
    """是否执行了 ``do_nothing``。"""

    has_pass_and_wait: bool = False
    """是否执行了 ``pass_and_wait``。"""

    has_third_party: bool = False
    """是否存在交由框架执行的第三方调用。"""

    has_info_tool: bool = False
    """是否存在 ``tool-`` / ``agent-`` 类调用（有返回值，需续轮让模型消化）。"""

    has_failed_tool: bool = False
    """是否有调用执行失败（用于主循环的续轮重试计数）。"""

    @property
    def has_meaningful_action(self) -> bool:
        """是否产生了任何可以据以收口本轮的有效动作。"""
        return (
            self.has_reply
            or self.has_do_nothing
            or self.has_pass_and_wait
            or self.has_third_party
        )


# ── 备忘录 ────────────────────────────────────────────────

MEMO_MAX_ENTRIES: int = 10
"""单流最大有效备忘条目数；超出按 ``created_at`` 升序淘汰。"""

MEMO_DEFAULT_EXPIRE_HOURS: float = 24.0
"""LLM 未指定 ``expire_hours`` 时的默认过期时长。"""

MEMO_MIN_EXPIRE_HOURS: float = 1.0
"""允许的最小过期时长，低于此值会被夹到此值。"""

MEMO_MAX_EXPIRE_HOURS: float = 14 * 24.0
"""允许的最大过期时长（14 天），超出会被夹到此值。"""

MEMO_ID_LENGTH: int = 6
"""备忘 id 的短 hex 长度，便于 LLM 在对话中引用。"""


@dataclass
class Memo:
    """一条带过期时间的私人备忘录。

    定位为 LLM 显式标记的中短期关键事项，与 ``mental_log``（自动事件流）
    和 ``history_summary``（叙事压缩）互补，覆盖"接下来一段时间需要
    明确意识到的事"这一语义层。
    """

    memo_id: str = ""
    content: str = ""
    intent: str = ""
    created_at: float = 0.0
    expires_at: float = 0.0

    def __post_init__(self) -> None:
        """补齐缺省的 id 与创建时间。"""
        if not self.memo_id:
            self.memo_id = uuid.uuid4().hex[:MEMO_ID_LENGTH]
        if self.created_at <= 0:
            self.created_at = time.time()

    def is_expired(self, now: float | None = None) -> bool:
        """判断该备忘是否已过期。"""
        current = now if now is not None else time.time()
        return self.expires_at > 0 and current >= self.expires_at

    def remaining_seconds(self, now: float | None = None) -> float:
        """返回剩余秒数；已过期返回 0。"""
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
    """把 LLM 给出的 ``expire_hours`` 夹到合法区间。

    Args:
        raw: LLM 传入的原始小时数。

    Returns:
        float: 位于 ``[MEMO_MIN_EXPIRE_HOURS, MEMO_MAX_EXPIRE_HOURS]``
        的时长；非数值或非正数时回退到默认值。
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return MEMO_DEFAULT_EXPIRE_HOURS
    if value <= 0:
        return MEMO_DEFAULT_EXPIRE_HOURS
    return max(MEMO_MIN_EXPIRE_HOURS, min(value, MEMO_MAX_EXPIRE_HOURS))


# ── 等待状态 ──────────────────────────────────────────────


@dataclass
class WaitingConfig:
    """Bot 发送消息后进入等待状态的参数快照。"""

    expected_reaction: str = ""
    max_wait_seconds: float = 0.0
    started_at: float = 0.0

    def is_active(self) -> bool:
        """是否处于有效等待中。"""
        return self.max_wait_seconds > 0 and self.started_at > 0

    def get_elapsed_seconds(self) -> float:
        """返回已等待秒数；未在等待时返回 0。"""
        if not self.is_active():
            return 0.0
        return time.time() - self.started_at

    def is_timeout(self) -> bool:
        """是否已超过 ``max_wait_seconds``。"""
        if not self.is_active():
            return False
        return self.get_elapsed_seconds() >= self.max_wait_seconds

    def reset(self) -> None:
        """清空等待状态。"""
        self.expected_reaction = ""
        self.max_wait_seconds = 0.0
        self.started_at = 0.0

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "expected_reaction": self.expected_reaction,
            "max_wait_seconds": self.max_wait_seconds,
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WaitingConfig:
        """从字典反序列化。"""
        return cls(
            expected_reaction=data.get("expected_reaction", ""),
            max_wait_seconds=float(data.get("max_wait_seconds", 0)),
            started_at=float(data.get("started_at", 0)),
        )
