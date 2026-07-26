"""KFC 上下文链清理。

KFC 的主循环会在同一条 ``response`` 链上反复追加 payload，若干框架行为
在多轮累积下会产生两类脏数据，本模块负责在发送前就地修复：

1. **孤立 TOOL_RESULT**：前面缺少配对的 ASSISTANT(tool_calls)，会被
   provider 判为非法请求；
2. **残留 reminder 前缀**：框架只维护首尾 USER 上的 system_reminder，
   中间 USER 的旧前缀不会被清理，多轮后会大量堆积。
"""

from __future__ import annotations

from typing import Any

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.types import LLMPayload, ROLE, Text

logger = get_logger("kfc_payload_hygiene")

_REMINDER_OPEN_TAG = "<system_reminder>"

_SNAPSHOT_RADIUS = 5
"""发现孤立 TOOL_RESULT 时，日志中前后各打印的 payload 条数。"""

_PREVIEW_LIMIT = 80
"""单条 payload 预览的最大字符数。"""

_TOOL_CALL_TYPE_NAME = "ToolCall"


def heal_orphan_tool_results(response: Any, *, where: str) -> int:
    """丢弃 ``response.payloads`` 中的孤立 TOOL_RESULT。

    合法的 TOOL_RESULT 之前必须是携带 tool_calls 的 ASSISTANT，或另一条
    连续的 TOOL_RESULT；其余情况视为非法链路状态，就地移除并打日志。
    SYSTEM / TOOL 这类固定 payload 在回溯时被跳过，不影响配对判定。

    Args:
        response: 持有 ``payloads`` 列表的响应对象。
        where: 调用位置标识，用于日志定位。

    Returns:
        int: 被丢弃的孤立 TOOL_RESULT 数量。
    """
    payloads = response.payloads
    if not isinstance(payloads, list) or not payloads:
        return 0

    pinned_roles = {ROLE.SYSTEM, ROLE.TOOL}
    healed = 0
    index = 0
    while index < len(payloads):
        if payloads[index].role != ROLE.TOOL_RESULT:
            index += 1
            continue

        prev_index = index - 1
        while prev_index >= 0 and payloads[prev_index].role in pinned_roles:
            prev_index -= 1

        prev_payload = payloads[prev_index] if prev_index >= 0 else None
        prev_role = prev_payload.role if prev_payload is not None else None
        is_valid = prev_role == ROLE.TOOL_RESULT or (
            prev_role == ROLE.ASSISTANT
            and prev_payload is not None
            and _has_tool_calls(prev_payload)
        )
        if is_valid:
            index += 1
            continue

        logger.error(
            f"孤立 TOOL_RESULT 自愈（{where}）：丢弃 idx={index}，"
            f"prev_role={prev_role.value if prev_role else None}\n"
            + _format_snapshot(payloads, index)
        )
        payloads.pop(index)
        healed += 1

    return healed


def strip_stale_reminder_prefixes(response: Any) -> None:
    """剥离中间 USER payload 上残留的旧 reminder 前缀。

    框架的 reminder 管线只处理首个和最后一个 USER payload。主循环每轮
    追加新 USER 后，上一轮的"末尾 USER"会变成中间 USER，其 reminder
    前缀便永久残留——多轮之后上下文里会堆满重复的系统提醒。

    Args:
        response: 持有 ``payloads`` 列表的响应对象。
    """
    payloads = response.payloads
    if not isinstance(payloads, list) or not payloads:
        return

    user_indices = [
        index for index, payload in enumerate(payloads) if payload.role == ROLE.USER
    ]
    if len(user_indices) <= 2:
        return

    for index in user_indices[1:-1]:
        content = payloads[index].content
        if not isinstance(content, list):
            continue

        kept: list[Any] = []
        in_leading_reminder = True
        for item in content:
            is_reminder = (
                in_leading_reminder
                and isinstance(item, Text)
                and item.text.startswith(_REMINDER_OPEN_TAG)
            )
            if is_reminder:
                continue
            in_leading_reminder = False
            kept.append(item)

        if len(kept) != len(content):
            payloads[index] = LLMPayload(ROLE.USER, kept)


def _has_tool_calls(payload: LLMPayload) -> bool:
    """判断 ASSISTANT payload 是否携带 tool_calls。"""
    content = payload.content
    if not isinstance(content, list):
        return False
    return any(type(item).__name__ == _TOOL_CALL_TYPE_NAME for item in content)


def _format_snapshot(payloads: list[LLMPayload], center: int) -> str:
    """渲染问题位置附近的 payload 快照，便于排查链路异常。"""
    start = max(0, center - _SNAPSHOT_RADIUS)
    end = min(len(payloads), center + _SNAPSHOT_RADIUS + 1)
    return "\n".join(
        f"[{index}] {payloads[index].role.value}: {_preview_payload(payloads[index])}"
        for index in range(start, end)
    )


def _preview_payload(payload: LLMPayload) -> str:
    """把 payload 内容压成短预览字符串。"""
    content = payload.content
    if not isinstance(content, list):
        content = [content]

    parts: list[str] = []
    for item in content:
        type_name = type(item).__name__
        if isinstance(item, Text):
            parts.append(f"{type_name}({item.text[:30]!r})")
        else:
            parts.append(type_name)
    return " | ".join(parts)[:_PREVIEW_LIMIT]
