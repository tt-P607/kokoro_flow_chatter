"""KFC 上下文链角色相位。

描述 LLM 上下文链当前所处的角色阶段，回答"这条链现在允许做什么"。
与 ``domain.turn_trigger`` 区分：后者回答"本 tick 为什么要运行"。

链尾是 TOOL_RESULT 时不能直接追加新的 USER——必须先让模型消化工具
结果，否则 provider 会因角色顺序非法而拒绝请求。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from src.app.plugin_system.types import ROLE


class ConversationPhase(str, Enum):
    """上下文链的角色相位。

    Members:
        WAIT_INPUT: 可以接收新的 USER 输入。
        MODEL_TURN: 正在请求模型生成 assistant / tool_calls。
        TOOL_EXEC: 正在执行工具并写入 TOOL_RESULT。
        FOLLOW_UP: 已有 TOOL_RESULT，需让模型继续消化。
        COMMIT: 本轮决策已执行，准备提交会话状态。
    """

    WAIT_INPUT = "wait_input"
    MODEL_TURN = "model_turn"
    TOOL_EXEC = "tool_exec"
    FOLLOW_UP = "follow_up"
    COMMIT = "commit"


def has_tool_result_tail(response: Any) -> bool:
    """判断上下文链尾部是否为 TOOL_RESULT。"""
    payloads = response.payloads
    return bool(payloads and payloads[-1].role == ROLE.TOOL_RESULT)


def phase_for_turn_start(
    response: Any,
    *,
    has_pending_tool_results: bool,
) -> ConversationPhase:
    """按链尾状态与待续轮信号确定回合起始相位。

    Args:
        response: 当前上下文链。
        has_pending_tool_results: 上轮是否留下待消化的工具结果。

    Returns:
        ConversationPhase: 回合起始相位。
    """
    if has_pending_tool_results or has_tool_result_tail(response):
        return ConversationPhase.FOLLOW_UP
    return ConversationPhase.WAIT_INPUT


def phase_for_model_result(response: Any) -> ConversationPhase:
    """按模型响应是否含工具调用确定后续相位。"""
    if response.call_list:
        return ConversationPhase.TOOL_EXEC
    return ConversationPhase.COMMIT


def can_accept_user_payload(phase: ConversationPhase) -> bool:
    """判断当前相位是否允许追加新的 USER payload。"""
    return phase in {ConversationPhase.WAIT_INPUT, ConversationPhase.COMMIT}
