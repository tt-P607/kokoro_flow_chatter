"""KFC LLM 上下文链桥接工具。

用于在向 ``response.payloads`` 追加新 USER 之前，先确保已闭合的
``assistant(tool_calls) → tool_result`` 链有 ASSISTANT 承接，
否则 LLMContextManager 会以 ``tool_result 后不能直接跟 user`` 拒绝请求。
"""

from __future__ import annotations

from typing import Any

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.types import LLMPayload, ROLE, Text


logger = get_logger("kfc_context_bridge")

_BRIDGE_TEXT = "好的。"


def ensure_tool_chain_closed(response: Any, *, reason: str = "") -> bool:
    """若 response 末尾为 TOOL_RESULT，则插入占位 ASSISTANT 桥接。

    Args:
        response: LLM 响应/请求对象，其 ``payloads`` 列表会被就地修改。
        reason: 触发桥接的上下文（仅用于日志）。

    Returns:
        bool: 是否实际插入了桥接 payload。
    """
    payloads = getattr(response, "payloads", None)
    if not payloads:
        return False
    if payloads[-1].role != ROLE.TOOL_RESULT:
        return False
    if reason:
        logger.debug(f"插入 ASSISTANT 桥接闭合 tool_result 链：{reason}")
    response.add_payload(LLMPayload(ROLE.ASSISTANT, Text(_BRIDGE_TEXT)))
    return True


def safe_add_payload(response: Any, payload: LLMPayload, *, reason: str = "") -> None:
    """带桥接保护的 ``add_payload``。

    若待追加 payload 是 USER 而 response 末尾仍是 TOOL_RESULT，
    会自动先插入 ASSISTANT 桥接，然后再委派到 ``response.add_payload``。

    Args:
        response: 目标响应/请求对象。
        payload: 待追加的 payload。
        reason: 桥接日志中的上下文标识。
    """
    if payload.role == ROLE.USER:
        ensure_tool_chain_closed(response, reason=reason)
    response.add_payload(payload)


def heal_orphan_tool_results(response: Any, *, where: str) -> int:
    """扫描 ``response.payloads``，丢弃孤立的 TOOL_RESULT。

    "孤立" 的判定：一个 TOOL_RESULT 之前必须紧跟 ASSISTANT(含 tool_calls)
    或另一个连续的 TOOL_RESULT；否则视为非法链路状态。

    检测到孤立 TOOL_RESULT 时：
    - 打 ERROR 日志，附带被腐蚀点 ±5 个 payload 的角色快照与文本片段
    - 就地从 ``response.payloads`` 移除该 TOOL_RESULT

    Args:
        response: 拥有 ``payloads`` 列表的响应对象。
        where: 调用位置标识（用于日志），例如 ``"loop-top"``。

    Returns:
        int: 被丢弃的孤立 TOOL_RESULT 数量。
    """
    payloads = getattr(response, "payloads", None)
    if not isinstance(payloads, list) or not payloads:
        return 0

    pinned_roles = {ROLE.SYSTEM, ROLE.TOOL}
    healed = 0
    idx = 0
    while idx < len(payloads):
        payload = payloads[idx]
        if payload.role != ROLE.TOOL_RESULT or payload.role in pinned_roles:
            idx += 1
            continue

        # 反向找最近的非 pinned payload
        prev_idx = idx - 1
        while prev_idx >= 0 and payloads[prev_idx].role in pinned_roles:
            prev_idx -= 1

        prev_payload = payloads[prev_idx] if prev_idx >= 0 else None
        prev_role = prev_payload.role if prev_payload is not None else None

        valid_prev = prev_role == ROLE.TOOL_RESULT or (
            prev_role == ROLE.ASSISTANT
            and _assistant_has_tool_calls(prev_payload)
        )
        if valid_prev:
            idx += 1
            continue

        # 命中孤立 TOOL_RESULT，记录现场后丢弃
        snapshot_start = max(0, idx - 5)
        snapshot_end = min(len(payloads), idx + 6)
        snapshot = []
        for s_idx in range(snapshot_start, snapshot_end):
            snap_payload = payloads[s_idx]
            preview = _preview_payload(snap_payload)
            snapshot.append(
                f"[{s_idx}] {snap_payload.role.value}: {preview}"
            )
        marker = "孤立 TOOL_RESULT 自愈"
        logger.error(
            f"{marker}（{where}）：丢弃 idx={idx}，"
            f"prev_role={prev_role.value if prev_role else None}\n"
            + "\n".join(snapshot)
        )
        payloads.pop(idx)
        healed += 1
        # idx 不前进，重新检查当前位置（被后续 payload 顶上来）

    return healed


def _assistant_has_tool_calls(payload: LLMPayload) -> bool:
    """判断 ASSISTANT payload 是否包含 tool_calls。"""
    content = payload.content
    if not isinstance(content, list):
        return False
    for item in content:
        # ToolCall 类型由框架定义，类名末尾即 ToolCall
        if type(item).__name__ == "ToolCall":
            return True
    return False


def _preview_payload(payload: LLMPayload) -> str:
    """将 payload 内容压成短预览字符串（最多 80 字符）。"""
    content = payload.content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            type_name = type(item).__name__
            text_attr = getattr(item, "text", None)
            if isinstance(text_attr, str):
                parts.append(f"{type_name}({text_attr[:30]!r})")
            else:
                name_attr = getattr(item, "name", None)
                parts.append(f"{type_name}(name={name_attr!r})")
        preview = " | ".join(parts)
    else:
        text_attr = getattr(content, "text", None)
        preview = repr(text_attr)[:80] if isinstance(text_attr, str) else repr(content)[:80]
    return preview[:80]
