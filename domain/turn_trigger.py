"""KFC 回合触发类型。

把 ``prepare_turn_input`` 中原本散落在 if/elif 链里的 4 种隐式分支
显式化为枚举，便于阅读、调试和后续扩展。
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..session import KFCSession


class TurnTrigger(str, Enum):
    """单轮 LLM 调用的触发原因。

    Members:
        NEW_MESSAGES: 收到新的未读消息（最常见路径）。
        FOLLOWUP_TOOL_RESULT: 上一轮 LLM 产生 tool_call，本轮基于工具结果继续推理。
        TIMEOUT_EXPIRED: ``session`` 处于 waiting 且 ``timeout_service`` 判定已超时，
            需要由 bot 主动续话。
        PROACTIVE_WAKE: 主动发起调度唤醒。
        IDLE_WAIT: 既无新消息、也未到超时阈值，应该让出本 tick 等待下一次唤醒。
    """

    NEW_MESSAGES = "new_messages"
    FOLLOWUP_TOOL_RESULT = "followup_tool_result"
    TIMEOUT_EXPIRED = "timeout_expired"
    PROACTIVE_WAKE = "proactive_wake"
    IDLE_WAIT = "idle_wait"

    TOOL_CONTINUE = FOLLOWUP_TOOL_RESULT
    TIMEOUT = TIMEOUT_EXPIRED


def classify_turn_trigger(
    *,
    has_unread: bool,
    has_pending_tool_results: bool,
    session: "KFCSession",
    is_timeout: bool,
) -> TurnTrigger:
    """根据当前状态确定本轮触发类型。

    优先级：``NEW_MESSAGES`` > ``FOLLOWUP_TOOL_RESULT`` > ``TIMEOUT_EXPIRED`` > ``IDLE_WAIT``。
    """
    if has_unread:
        return TurnTrigger.NEW_MESSAGES
    if has_pending_tool_results:
        return TurnTrigger.FOLLOWUP_TOOL_RESULT
    if session.is_waiting() and is_timeout:
        return TurnTrigger.TIMEOUT_EXPIRED
    return TurnTrigger.IDLE_WAIT
