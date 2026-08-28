"""KFC 回合触发类型。

把主循环"本 tick 为什么要运行"这一判断显式化为枚举，与
``runtime.phase_machine`` 的角色相位区分开：触发类型回答**为什么运行**，
相位回答**当前上下文链允许做什么**。
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
        FOLLOWUP_TOOL_RESULT: 上轮产生了工具结果，本轮让模型继续消化。
        EXTERNAL_RESUME: 收到框架外部恢复事件，需要形成 transient turn。
        TIMEOUT_EXPIRED: 等待已超时，需要 bot 主动续话。
        IDLE_WAIT: 既无新消息也未到超时，让出本 tick。
    """

    NEW_MESSAGES = "new_messages"
    FOLLOWUP_TOOL_RESULT = "followup_tool_result"
    EXTERNAL_RESUME = "external_resume"
    TIMEOUT_EXPIRED = "timeout_expired"
    IDLE_WAIT = "idle_wait"


def classify_turn_trigger(
    *,
    has_unread: bool,
    has_pending_tool_results: bool,
    session: KFCSession,
    is_timeout: bool,
    has_external_resume: bool = False,
) -> TurnTrigger:
    """确定本轮触发类型。

    优先级固定为
    ``FOLLOWUP_TOOL_RESULT > EXTERNAL_RESUME > NEW_MESSAGES > TIMEOUT_EXPIRED > IDLE_WAIT``，
    确保未闭合工具链先消化完毕；跨流行动等恢复事件优先于普通未读，
    形成不受真实聊天稀释的独立行动轮，未读保留在队列中待下一轮处理。

    Args:
        has_unread: 是否存在待处理的未读消息。
        has_pending_tool_results: 上轮是否留下了待消化的工具结果。
        session: 当前会话，用于判断是否处于等待状态。
        is_timeout: 等待是否已超时。
        has_external_resume: 是否有待处理的框架恢复事件。

    Returns:
        TurnTrigger: 本轮触发类型。
    """
    if has_pending_tool_results:
        return TurnTrigger.FOLLOWUP_TOOL_RESULT
    if has_external_resume:
        return TurnTrigger.EXTERNAL_RESUME
    if has_unread:
        return TurnTrigger.NEW_MESSAGES
    if session.is_waiting() and is_timeout:
        return TurnTrigger.TIMEOUT_EXPIRED
    return TurnTrigger.IDLE_WAIT
