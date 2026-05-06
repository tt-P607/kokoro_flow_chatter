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
        TOOL_CONTINUE: 上一轮 LLM 产生 tool_call，本轮基于工具结果继续推理。
        TIMEOUT: ``session`` 处于 waiting 且 ``timeout_service`` 判定已超时，
            需要由 bot 主动续话。
        IDLE_WAIT: 既无新消息、也未到超时阈值，应该让出本 tick 等待下一次唤醒。
    """

    NEW_MESSAGES = "new_messages"
    TOOL_CONTINUE = "tool_continue"
    TIMEOUT = "timeout"
    IDLE_WAIT = "idle_wait"


def classify_turn_trigger(
    *,
    has_unread: bool,
    has_pending_tool_results: bool,
    session: "KFCSession",
    is_timeout: bool,
) -> TurnTrigger:
    """根据当前状态确定本轮触发类型。

    优先级：``NEW_MESSAGES`` > ``TOOL_CONTINUE`` > ``TIMEOUT`` > ``IDLE_WAIT``。
    """
    if has_unread:
        return TurnTrigger.NEW_MESSAGES
    if has_pending_tool_results:
        return TurnTrigger.TOOL_CONTINUE
    if session.is_waiting() and is_timeout:
        return TurnTrigger.TIMEOUT
    return TurnTrigger.IDLE_WAIT
