"""KFC 上下文快照。

快照源是主链持久 payload——即 ``RequestView`` 回写的 LLM 原始返回（推理
``ReasoningText`` + 正文 ``Text`` + 工具调用
``ToolCall`` + 工具回执 ``ToolResult`` + 媒体），逐 part 一对一转存，
不做二次格式化，恢复后即模型当时的真实输出。

捕获只在**回合闭合点**（主循环走 ``Wait``/``Stop`` 收口、无待消化工具
结果）进行，此刻主链必然闭合——所有 ``tool_calls`` 都有配对
``TOOL_RESULT``，链尾不可能悬挂，因此恢复后的链在结构上不会触发
框架的 ``LLMContextError``。反序列化后仍显式自检，失败返回 ``None``，
由调用方按“无历史、新会话”处理，绝不注入非法链。
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.types import (
    LLMPayload,
    ROLE,
    Text,
    ToolCall,
    ToolResult,
)
from src.kernel.llm.context_structure import validate_payload_sequence
from src.kernel.llm.payload import ReasoningText

logger = get_logger("kfc_snapshot")

#: 反序列化时跳过、由框架重新构建的固定角色。
_SKIP_ROLES = {ROLE.SYSTEM, ROLE.TOOL}

#: 快照单条 content 文本的最大长度，防止异常巨型内容撑爆会话文件。
_MAX_ENTRY_CHARS = 4000

#: 匹配 user 文本中框架动态注入的 system_reminder 块（含标签本身）。
_SYSTEM_REMINDER_RE = re.compile(
    r"<system_reminder>.*?</system_reminder>",
    flags=re.DOTALL,
)

#: 媒体类型的类名集合；这些类型的 value 为 base64 字符串，应完整入快照。
_MEDIA_TYPES = {"Image", "Audio", "Video", "File"}

DYNAMIC_BACKGROUND_MARKER = "【动态背景】"
"""当前请求动态背景的可见标记；序列化时用它排除伪 USER 历史。"""

_TRIM_TRIGGER_RATIO = 0.8
"""快照达到上限的 80% 时触发一次批量裁剪。"""

_TRIM_KEEP_RATIO = 0.2
"""批量裁剪后保留末尾约 20% 的近期条目。"""


def serialize_payloads(payloads: list[Any]) -> list[dict[str, Any]]:
    """把主链 payload 列表序列化为快照条目。

    跳过 SYSTEM / TOOL（恢复时框架重新构建）；user 文本剥离
    system_reminder 块（框架在每次请求时重新注入）。其余内容逐 part
    无损转存，保持角色与顺序。

    Args:
        payloads: 主链 ``response.payloads``。

    Returns:
        list[dict]: 序列化后的快照条目，形如
        ``{"role": "user|assistant|tool_result", "content": [...]}``。
    """
    entries: list[dict[str, Any]] = []
    for payload in payloads or []:
        role = getattr(payload, "role", None)
        if role in _SKIP_ROLES:
            continue
        if role == ROLE.USER and _is_dynamic_background(payload):
            continue
        content = getattr(payload, "content", None)
        if not isinstance(content, list):
            content = [content] if content is not None else []

        parts: list[dict[str, Any]] = []
        for part in content:
            serialized = _serialize_part(part)
            if serialized is not None:
                parts.append(serialized)
        if not parts:
            continue
        entries.append(
            {
                "role": getattr(role, "value", str(role or "")),
                "content": parts,
            }
        )
    return entries


def trim_snapshot(
    entries: list[dict[str, Any]],
    max_payloads: int,
) -> list[dict[str, Any]]:
    """在阈值点批量裁剪快照，保证链头为 USER、尾部工具段闭合。

    平时不逐轮滑动删除；当条目数接近上限时一次性截掉旧前缀，只保留
    近期尾巴，减少多次请求间持久上下文前缀的变化。

    Args:
        entries: 待裁剪的快照条目。
        max_payloads: 快照容量；触发与保留数量均按该配置计算。

    Returns:
        list[dict]: 裁剪后的快照条目。
    """
    if max_payloads <= 0:
        max_payloads = 30

    trigger_count = math.ceil(max_payloads * _TRIM_TRIGGER_RATIO)
    keep_count = max(1, math.ceil(max_payloads * _TRIM_KEEP_RATIO))
    if len(entries) >= trigger_count:
        entries = _batch_trim_to_recent_boundary(entries, keep_count)
    elif len(entries) > max_payloads:
        entries = entries[-max_payloads:]

    # 丢弃头部非 USER 条目，保证恢复后对话以 USER 起始
    while entries and entries[0].get("role") != ROLE.USER.value:
        entries.pop(0)

    # 从尾部丢弃未配对的 assistant(tool_calls) 悬挂段
    entries = _drop_unpaired_tail(entries)
    return entries


def _batch_trim_to_recent_boundary(
    entries: list[dict[str, Any]],
    keep_count: int,
) -> list[dict[str, Any]]:
    """按目标数量截尾，并向前调整到最近的真实对话边界。

    从候选起点向后弹出会拆断正在进行的工具段或把 ASSISTANT 变成链头；
    因此这里向左扩展到最近的 USER 条目，允许实际保留数略多于目标值。
    """
    start = len(entries) - keep_count
    while start > 0 and entries[start].get("role") != ROLE.USER.value:
        start -= 1
    return entries[start:]


def deserialize_snapshot(
    entries: list[dict[str, Any]] | None,
) -> list[LLMPayload] | None:
    """把快照条目反序列化为 LLM payload 列表。

    先丢弃头部非 USER 与尾部悬挂工具段，再按框架同规则自检；校验
    失败或结果为空时返回 ``None``，由调用方按无历史处理。

    Args:
        entries: 快照条目。

    Returns:
        list[LLMPayload] | None: 合法 payload 列表；不可用返回 ``None``。
    """
    if not entries:
        return None

    entries = list(entries)
    while entries and entries[0].get("role") != ROLE.USER.value:
        entries.pop(0)
    entries = _drop_unpaired_tail(entries)
    if not entries:
        return None

    payloads: list[LLMPayload] = []
    for entry in entries:
        payload = _deserialize_entry(entry)
        if payload is not None:
            payloads.append(payload)
    if not payloads:
        return None

    try:
        validate_payload_sequence(payloads, allow_incomplete_tail=True)
    except Exception as error:  # noqa: BLE001 - 统一按不可用处理并回退
        logger.warning(f"快照反序列化后校验失败，回退历史链: {error}")
        return None
    return payloads


def capture_snapshot(payloads: list[Any], max_payloads: int) -> list[dict[str, Any]] | None:
    """从主链捕获并裁剪快照；无有效对话内容时返回 ``None``。

    Args:
        payloads: 主链 ``response.payloads``。
        max_payloads: 快照条目上限（来自 ``config.prompt.max_context_payloads``）。

    Returns:
        list[dict] | None: 快照条目；无可捕获内容返回 ``None``。
    """
    entries = serialize_payloads(payloads)
    if not entries:
        return None
    return trim_snapshot(entries, max_payloads)


# ── 内部实现 ──────────────────────────────────────────────


def _serialize_part(part: Any) -> dict[str, Any] | None:
    """把单个 content part 序列化为字典；不可识别时返回 ``None``。"""
    type_name = type(part).__name__
    if isinstance(part, Text):
        return {"type": "text", "text": _truncate(_strip_system_reminders(part.text))}
    if isinstance(part, ReasoningText):
        item: dict[str, Any] = {"type": "reasoning", "text": _truncate(part.text)}
        if part.signature:
            item["signature"] = part.signature
        if part.redacted_data:
            item["redacted_data"] = part.redacted_data
        return item
    if isinstance(part, ToolCall):
        return {
            "type": "tool_call",
            "id": str(part.id or ""),
            "name": str(part.name or ""),
            "args": part.args if isinstance(part.args, dict) else str(part.args or ""),
        }
    if isinstance(part, ToolResult):
        return {
            "type": "tool_result",
            "value": _serialize_tool_result_value(part.value),
            "call_id": str(part.call_id or ""),
            "name": str(part.name or ""),
        }
    if type_name in _MEDIA_TYPES:
        item: dict[str, Any] = {
            "type": type_name.lower(),
            "data": _safe_text(getattr(part, "value", "") or ""),
        }
        mime_type = getattr(part, "mime_type", None)
        if mime_type:
            item["mime_type"] = str(mime_type)
        return item
    rendered = _render_unknown_part(part)
    if rendered:
        return {"type": "text", "text": _truncate(rendered)}
    return None


def _is_dynamic_background(payload: Any) -> bool:
    """判断是否为每轮重建的动态背景 USER payload。"""
    content = payload.content
    if not isinstance(content, list):
        content = [content]
    text = "".join(part.text for part in content if isinstance(part, Text))
    return text.startswith(DYNAMIC_BACKGROUND_MARKER)


def _deserialize_entry(entry: dict[str, Any]) -> LLMPayload | None:
    """把单条快照条目还原为 LLMPayload。"""
    role_name = str(entry.get("role", "") or "")
    try:
        role = ROLE(role_name)
    except ValueError:
        return None
    if role in _SKIP_ROLES:
        return None

    content: list[Any] = []
    for part in entry.get("content", []) or []:
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type", "") or "")
        if part_type == "text":
            text = str(part.get("text", "") or "")
            if text:
                content.append(Text(text))
        elif part_type == "reasoning":
            text = str(part.get("text", "") or "")
            if text or part.get("redacted_data"):
                content.append(
                    ReasoningText(
                        text,
                        signature=part.get("signature") or None,
                        redacted_data=part.get("redacted_data") or None,
                    )
                )
        elif part_type == "tool_call":
            content.append(
                ToolCall(
                    id=str(part.get("id", "") or ""),
                    name=str(part.get("name", "") or ""),
                    args=part.get("args", {}) if isinstance(part.get("args"), dict) else str(part.get("args", "") or ""),
                )
            )
        elif part_type == "tool_result":
            content.append(
                ToolResult(
                    value=part.get("value", ""),
                    call_id=str(part.get("call_id", "") or "") or None,
                    name=str(part.get("name", "") or "") or None,
                )
            )
        elif part_type in ("image", "audio", "video", "file"):
            media = _deserialize_media(part_type, part)
            if media is not None:
                content.append(media)

    if not content:
        return None
    return LLMPayload(role, content)


def _drop_unpaired_tail(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从尾部丢弃未闭合的 assistant(tool_calls) 段及其后内容。

    维护一个"安全点"：每当所有已声明的 tool_call 都被配对 tool_result
    闭合时，记录当前已处理长度；若结尾仍存在未配对 tool_call，回退到
    最近安全点，丢弃整个未闭合段。
    """
    pending_ids: set[str] = set()
    safe_len = 0
    cleaned: list[dict[str, Any]] = []
    for entry in entries:
        role = str(entry.get("role", "") or "")
        content = entry.get("content", []) or []

        if role == ROLE.USER.value:
            # 新 user 到来时若仍有未配对 tool_call，回退到最近安全点
            if pending_ids:
                del cleaned[safe_len:]
                pending_ids.clear()
            cleaned.append(entry)
            if not pending_ids:
                safe_len = len(cleaned)
        elif role == ROLE.ASSISTANT.value:
            for part in content:
                if isinstance(part, dict) and part.get("type") == "tool_call":
                    call_id = str(part.get("id", "") or "")
                    if call_id:
                        pending_ids.add(call_id)
            cleaned.append(entry)
        elif role == ROLE.TOOL_RESULT.value:
            for part in content:
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    call_id = str(part.get("call_id", "") or "")
                    if call_id in pending_ids:
                        pending_ids.discard(call_id)
            cleaned.append(entry)
            if not pending_ids:
                safe_len = len(cleaned)
        else:
            cleaned.append(entry)
            if not pending_ids:
                safe_len = len(cleaned)

    if pending_ids:
        del cleaned[safe_len:]
    return cleaned


def _serialize_tool_result_value(value: Any) -> str:
    """序列化 ToolResult 的值。"""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return _safe_text(value)


def _deserialize_media(part_type: str, part: dict[str, Any]) -> Any | None:
    """把快照媒体条目还原为框架媒体对象。"""
    from src.kernel.llm import Audio, Image, Video
    from src.kernel.llm.payload.content import File

    data = str(part.get("data", "") or "")
    if not data:
        return None
    mime_type = part.get("mime_type")
    try:
        if part_type == "image":
            return Image(data)
        if part_type == "audio":
            return Audio(data)
        if part_type == "video":
            return Video(data, mime_type=str(mime_type or "video/mp4"))
        return File(data)
    except Exception:  # noqa: BLE001
        return None


def _strip_system_reminders(text: str) -> str:
    """剥离 user 文本中的 system_reminder 块。"""
    return _SYSTEM_REMINDER_RE.sub("", text)


def _render_unknown_part(part: Any) -> str:
    """渲染无法识别的 content 类型。"""
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        try:
            return json.dumps(part, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            return _safe_text(part)
    return _safe_text(part)


def _safe_text(value: Any) -> str:
    """把任意值安全转换为字符串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:  # noqa: BLE001
        return ""


def _truncate(text: str) -> str:
    """裁剪过长文本。"""
    if len(text) <= _MAX_ENTRY_CHARS:
        return text
    return text[:_MAX_ENTRY_CHARS].rstrip() + "..."
