"""KFC 对话 role-phase 状态机。

本模块只描述 LLM 上下文链所处的角色相位，不替代
:mod:`plugins.kokoro_flow_chatter.domain.turn_trigger` 中的触发原因。
触发原因回答“为什么本 tick 要运行”，相位回答“当前 response 链允许做什么”。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from src.app.plugin_system.types import ROLE


class ConversationPhase(str, Enum):
    """KFC 主链角色相位。

    Members:
        WAIT_INPUT: 可以接收新的 USER 输入。
        MODEL_TURN: 正在请求模型生成 assistant/tool_calls。
        TOOL_EXEC: 正在执行模型请求的工具并写入 TOOL_RESULT。
        FOLLOW_UP: 已有 TOOL_RESULT，需要让模型继续吸收工具结果。
        COMMIT: 本轮模型决策已执行，准备提交 session 状态。
    """

    WAIT_INPUT = "wait_input"
    MODEL_TURN = "model_turn"
    TOOL_EXEC = "tool_exec"
    FOLLOW_UP = "follow_up"
    COMMIT = "commit"


def has_tool_result_tail(response: Any) -> bool:
    """判断 response 尾部是否为 TOOL_RESULT。"""
    payloads = getattr(response, "payloads", None)
    return bool(payloads and payloads[-1].role == ROLE.TOOL_RESULT)


def phase_for_turn_start(response: Any, *, has_pending_tool_results: bool) -> ConversationPhase:
    """根据 response 链尾和待续轮信号选择回合起始相位。"""
    if has_pending_tool_results or has_tool_result_tail(response):
        return ConversationPhase.FOLLOW_UP
    return ConversationPhase.WAIT_INPUT


def phase_for_model_result(response: Any) -> ConversationPhase:
    """根据模型响应是否包含 tool call 选择后续相位。"""
    if getattr(response, "call_list", None):
        return ConversationPhase.TOOL_EXEC
    return ConversationPhase.COMMIT


def can_accept_user_payload(phase: ConversationPhase) -> bool:
    """判断当前相位是否允许追加新的 USER payload。"""
    return phase in {ConversationPhase.WAIT_INPUT, ConversationPhase.COMMIT}
