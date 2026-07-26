"""KFC 决策收敛。

把执行层产出的 ``ToolCallResult`` 与原始响应一起，收敛为主循环使用的
``Decision``。本模块为纯函数，不产生任何副作用。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..domain.decision import Decision, ProactiveSchedule, ToolCallSpec
from ..models import DO_NOTHING, KFC_REPLY, PASS_AND_WAIT, ToolCallResult
from .tool_call_adapter import extract_call_args, normalize_call_name

if TYPE_CHECKING:
    from src.app.plugin_system.types import ToolCall

_SCHEDULE_PROACTIVE_NAME = "schedule_proactive"
"""主动发起预约动作名，需要在收敛时提取为结构化计划。"""

_DEFAULT_SCHEDULE_DELAY_MINUTES = 30.0
"""模型未给出或给出非法 ``delay_minutes`` 时的回退值。"""

_CONTROL_NAMES: frozenset[str] = frozenset({KFC_REPLY, DO_NOTHING, PASS_AND_WAIT})
"""KFC 控制动作，不计入第三方调用列表。"""


def _extract_visible_reply_segments(result: ToolCallResult) -> list[str]:
    """提取用户实际可见的回复分段。

    KFC 会把内心活动、第三方动作等一并记入 ``actions``，但持久化对话链
    只应保留用户真正看到的文本。

    Args:
        result: 执行层产出的工具执行结果。

    Returns:
        list[str]: 按发送顺序排列的非空回复分段。
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
        elif isinstance(raw_content, str) and raw_content.strip():
            segments.append(raw_content.strip())
    return segments


def _build_proactive_schedule(args: dict[str, Any]) -> ProactiveSchedule:
    """从 ``schedule_proactive`` 参数构造预约计划。"""
    try:
        delay_minutes = float(args.get("delay_minutes", _DEFAULT_SCHEDULE_DELAY_MINUTES))
    except (TypeError, ValueError):
        delay_minutes = _DEFAULT_SCHEDULE_DELAY_MINUTES
    return ProactiveSchedule(
        delay_minutes=delay_minutes,
        reason=str(args.get("reason", "") or "").strip(),
    )


def build_decision(
    result: ToolCallResult,
    call_list: list[ToolCall] | None,
) -> Decision:
    """把执行结果与原始调用列表收敛为统一 ``Decision``。

    Args:
        result: 执行层产出的工具执行结果。
        call_list: 模型本轮返回的原始工具调用列表。

    Returns:
        Decision: 供主循环消费的统一决策对象。
    """
    third_party_calls: list[ToolCallSpec] = []
    proactive_schedule: ProactiveSchedule | None = None

    for call in call_list or []:
        normalized_name = normalize_call_name(call.name)
        if normalized_name in _CONTROL_NAMES:
            continue

        args = extract_call_args(call.args)
        third_party_calls.append(
            ToolCallSpec(
                name=normalized_name,
                call_id=str(call.id or ""),
                args=args,
            )
        )
        if normalized_name == _SCHEDULE_PROACTIVE_NAME:
            proactive_schedule = _build_proactive_schedule(args)

    return Decision(
        thought=result.thought,
        mood=result.mood,
        expected_reaction=result.expected_reaction,
        wait_seconds=result.max_wait_seconds,
        actions=list(result.actions),
        visible_reply_segments=_extract_visible_reply_segments(result),
        has_reply_action=result.has_reply,
        chose_silence=result.has_do_nothing and not result.has_reply,
        has_meaningful_action=result.has_meaningful_action,
        has_info_tool_calls=result.has_info_tool,
        has_failed_tool=result.has_failed_tool,
        third_party_calls=third_party_calls,
        proactive_schedule=proactive_schedule,
    )
