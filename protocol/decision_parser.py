"""KFC 决策对象构建。"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from ..domain.decision import Decision, ProactiveSchedule, ToolCallSpec
from ..models import DO_NOTHING, KFC_REPLY, ToolCallResult
from .tool_call_adapter import build_decision_draft


def _normalize_call_name(name: str) -> str:
    """归一化工具调用名称。"""
    if not name:
        return ""
    if ":" in name:
        return name.rsplit(":", 1)[-1]
    for prefix in ("action-", "tool-", "agent-"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _extract_args(raw_args: Any) -> dict[str, Any]:
    """提取工具参数字典，兼容字符串 JSON。"""
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _extract_visible_reply_segments(result: ToolCallResult) -> list[str]:
    """提取用户实际可见的回复段落。"""
    segments: list[str] = []
    for action in result.actions:
        if action.get("type") != KFC_REPLY:
            continue

        raw_content = action.get("content")
        if isinstance(raw_content, list):
            segments.extend(str(item).strip() for item in raw_content if str(item).strip())
            continue

        if isinstance(raw_content, str):
            stripped = raw_content.strip()
            if stripped:
                segments.append(stripped)

    return segments


def build_decision(result: ToolCallResult, response: Any) -> Decision:
    """根据已执行的工具结果构建统一 Decision。"""
    visible_reply_segments = _extract_visible_reply_segments(result)
    third_party_calls: list[ToolCallSpec] = []
    proactive_schedule: ProactiveSchedule | None = None

    for call in getattr(response, "call_list", None) or []:
        normalized_name = _normalize_call_name(getattr(call, "name", ""))
        if normalized_name in (KFC_REPLY, DO_NOTHING):
            continue

        args = _extract_args(getattr(call, "args", {}))
        third_party_calls.append(
            ToolCallSpec(
                name=normalized_name,
                call_id=str(getattr(call, "id", "") or ""),
                args=args,
            )
        )

        if normalized_name == "schedule_proactive":
            delay_raw = args.get("delay_minutes", 30)
            try:
                delay_minutes = float(delay_raw)
            except (TypeError, ValueError):
                delay_minutes = 30.0
            proactive_schedule = ProactiveSchedule(
                delay_minutes=delay_minutes,
                reason=str(args.get("reason", "") or "").strip(),
            )

    return Decision(
        thought=result.thought,
        mood=result.mood,
        expected_reaction=result.expected_reaction,
        wait_seconds=result.max_wait_seconds,
        actions=list(result.actions),
        visible_reply_segments=visible_reply_segments,
        has_reply_action=result.has_reply,
        chose_silence=result.has_do_nothing and not result.has_reply,
        has_meaningful_action=result.has_meaningful_action,
        has_info_tool_calls=result.has_info_tool,
        third_party_calls=third_party_calls,
        proactive_schedule=proactive_schedule,
    )


async def parse_response_decision(
    response: Any,
    usable_map: Any,
    trigger_msg: Any | None,
    config: Any,
    *,
    execute_reply_fn: Callable[[str, Any, Any | None, str], Awaitable[bool]],
    run_tool_call_fn: Callable[[list[Any], Any, Any, Any | None], Awaitable[list[tuple[bool, bool]]]],
    pre_execute_hook: Callable[[ToolCallResult], None] | None = None,
) -> Decision:
    """将模型响应拆为草稿、执行结果，并统一收敛为 Decision。"""
    from ..execution.decision_executor import execute_decision_draft

    draft = build_decision_draft(getattr(response, "call_list", None))
    tool_result = await execute_decision_draft(
        draft,
        response,
        usable_map,
        trigger_msg,
        config,
        execute_reply_fn=execute_reply_fn,
        run_tool_call_fn=run_tool_call_fn,
        pre_execute_hook=pre_execute_hook,
    )
    return build_decision(tool_result, response)
