"""KFC 完整上下文快照序列化与恢复。

KFC 的主循环维持同一条 ``response`` 链跨轮累积 payload（与 NDFC 的
``request.payloads`` 行为一致），因此每次成功发送后保存的完整 payload 链
天然就是"真正发给模型的上下文"。本模块把它序列化为可持久化的
:class:`ContextSnapshot`，作为 ``KFCSession.context_snapshot`` 的一个字段
写进会话 JSON（``data/kokoro_flow_chatter/sessions/<stream_id>.json``），
并在重启后首次 ``execute()`` 启动时反序列化回多角色 payload 链合并进
请求，使重启前后的上下文保持一致，消除"重启后连续性变差"。

与 ``session.chain_payloads``（精简的 user/assistant 文本链）的关系：
快照是更高保真的来源，保存工具调用与工具结果；启动时快照优先，缺失时
回退到 ``chain_payloads`` 的既有还原路径。快照只存
user / assistant / tool_result 历史链，system 与 tool 声明不进入恢复链
（KFC 每次自行构建 system prompt）。媒体不进入快照——多模态图片在
native_multimodal 模式下由主模型即时理解，不需要跨重启还原。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from src.app.plugin_system.api.log_api import COLOR, get_logger
from src.app.plugin_system.types import LLMPayload, ROLE, Text, ToolCall, ToolResult

SNAPSHOT_ROLE_SYSTEM = "system"
SNAPSHOT_ROLE_USER = "user"
SNAPSHOT_ROLE_ASSISTANT = "assistant"
SNAPSHOT_ROLE_TOOL_RESULT = "tool_result"

#: 工具 schema 声明角色，不属于对话历史，序列化与恢复时都跳过。
_SNAPSHOT_ROLE_TOOL = "tool"

#: 单条内容最大文本长度，防止异常巨型内容撑爆快照。
_MAX_ENTRY_CHARS = 4000

logger = get_logger("kfc_snapshot", display="KFC快照", color=COLOR.CYAN)

#: 快照条目角色到框架 ROLE 枚举的映射。
_ROLE_MAP = {
    SNAPSHOT_ROLE_SYSTEM: ROLE.SYSTEM,
    SNAPSHOT_ROLE_USER: ROLE.USER,
    SNAPSHOT_ROLE_ASSISTANT: ROLE.ASSISTANT,
    SNAPSHOT_ROLE_TOOL_RESULT: ROLE.TOOL_RESULT,
}


@dataclass
class SnapshotEntry:
    """单条上下文快照条目，对应一个 LLM payload。"""

    role: str
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    call_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": self.tool_calls,
            "call_id": self.call_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SnapshotEntry":
        """从字典反序列化。"""
        return cls(
            role=str(data.get("role", "") or ""),
            content=str(data.get("content", "") or ""),
            tool_calls=data.get("tool_calls", []) or [],
            call_id=str(data.get("call_id", "") or ""),
        )


@dataclass
class ContextSnapshot:
    """单个聊天流的完整上下文快照。"""

    stream_id: str
    updated_at: float = field(default_factory=time.time)
    version: int = 1
    entries: list[SnapshotEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "stream_id": self.stream_id,
            "updated_at": self.updated_at,
            "version": self.version,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextSnapshot":
        """从字典反序列化。"""
        raw_entries = data.get("entries", []) or []
        entries = [
            SnapshotEntry.from_dict(item)
            for item in raw_entries
            if isinstance(item, dict)
        ]
        return cls(
            stream_id=str(data.get("stream_id", "") or ""),
            updated_at=float(data.get("updated_at", time.time())),
            version=int(data.get("version", 1)),
            entries=entries,
        )


def serialize_payloads(payloads: list[Any]) -> list[SnapshotEntry]:
    """把 LLM payload 列表序列化为快照条目。

    Args:
        payloads: LLM payload 列表（来自 ``response.payloads``）。

    Returns:
        list[SnapshotEntry]: 序列化后的快照条目。
    """
    entries: list[SnapshotEntry] = []
    for payload in payloads or []:
        role = _extract_role(payload)
        if not role or role == _SNAPSHOT_ROLE_TOOL:
            continue

        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        def flush_buffered() -> None:
            """把当前累积的文本与 tool_call 落为一条快照条目。"""
            if not content_parts and not tool_calls:
                return
            entry = SnapshotEntry(role=role)
            if content_parts:
                entry.content = _truncate("\n".join(content_parts))
            entry.tool_calls = tool_calls
            entries.append(entry)

        for part in getattr(payload, "content", None) or []:
            part_type = type(part).__name__
            if part_type == "Text":
                text = _safe_text(getattr(part, "text", ""))
                if text:
                    if role == SNAPSHOT_ROLE_USER:
                        text = _strip_system_reminders(text)
                    if text:
                        content_parts.append(text)
            elif part_type == "ToolCall":
                tool_calls.append(_serialize_tool_call(part))
            elif part_type == "ToolResult":
                # 每个 ToolResult 独立成条，保证恢复时 assistant 的
                # tool_call 与 tool_result 能一一配对。
                flush_buffered()
                call_id = getattr(part, "call_id", None) or ""
                value = _serialize_tool_result_value(getattr(part, "value", ""))
                entries.append(
                    SnapshotEntry(
                        role=role,
                        content=_truncate(value),
                        call_id=str(call_id),
                    )
                )
                content_parts = []
                tool_calls = []
            else:
                rendered = _render_unknown_part(part)
                if rendered:
                    content_parts.append(rendered)

        flush_buffered()

    return _merge_consecutive(entries)


def capture_snapshot_from_payloads(
    stream_id: str,
    payloads: list[Any],
) -> ContextSnapshot | None:
    """从 LLM 请求 payloads 构造快照。

    Args:
        stream_id: 聊天流 ID。
        payloads: LLM 请求的 payload 列表。

    Returns:
        ContextSnapshot: 序列化成功且非空时返回，否则返回 None。
    """
    if not stream_id or not payloads:
        return None

    entries = serialize_payloads(payloads)
    if not entries:
        return None

    return ContextSnapshot(
        stream_id=stream_id,
        updated_at=time.time(),
        entries=entries,
    )


def deserialize_snapshot(snapshot: ContextSnapshot) -> list[Any]:
    """把快照反序列化为可注入的多角色 LLMPayload 列表。

    恢复只保留 user / assistant / tool_result 的历史链，跳过 system
    （KFC 每次自行构建 system prompt）。反序列化前先清洗快照条目，
    保证恢复出的 payload 序列对 LLM 上下文校验合法。

    Args:
        snapshot: 待反序列化的快照。

    Returns:
        list[Any]: LLMPayload 列表。
    """
    entries = _sanitize_entries(list(snapshot.entries or []))
    payloads: list[Any] = []
    for entry in entries:
        role = _ROLE_MAP.get(entry.role)
        if role is None:
            continue
        if entry.tool_calls:
            calls: list[Any] = [
                ToolCall(
                    id=str(call.get("id", "") or ""),
                    name=str(call.get("name", "") or ""),
                    args=call.get("args", {}) or {},
                )
                for call in entry.tool_calls
            ]
            payloads.append(LLMPayload(role, calls))
        elif entry.role == SNAPSHOT_ROLE_TOOL_RESULT:
            payloads.append(
                LLMPayload(
                    role,
                    ToolResult(entry.content, call_id=entry.call_id or None),
                )
            )
        else:
            content: list[Any] = []
            if entry.content:
                content.append(Text(entry.content))
            if not content:
                continue
            payloads.append(LLMPayload(role, content))
    return payloads


def log_snapshot_restored(snapshot: ContextSnapshot, payload_count: int) -> None:
    """输出快照恢复面板日志。"""
    entry_count = len(snapshot.entries)
    role_summary = _summarize_roles(snapshot.entries)
    updated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snapshot.updated_at))
    logger.print_panel(
        "上下文快照已恢复\n"
        f"- 聊天流：{snapshot.stream_id}\n"
        f"- 恢复条目：{entry_count}（payload {payload_count}）\n"
        f"- 角色构成：{role_summary}\n"
        f"- 快照时间：{updated_at}\n"
        "（快照源自上次运行，多角色上下文链已还原并合并回初始请求。）",
        title="KFC 上下文快照恢复",
        border_style="bright_cyan",
    )
    logger.info(
        f"快照已恢复 stream={snapshot.stream_id[:8]} entries={entry_count} payloads={payload_count}"
    )


def _sanitize_entries(entries: list[SnapshotEntry]) -> list[SnapshotEntry]:
    """清洗快照条目，保证反序列化后角色序列对 LLM 上下文校验合法。

    步骤：
    1. 移除 system 条目。
    2. 丢弃起始处孤立的 assistant / tool_result（无 user 起始）。
    3. 丢弃未闭合的 assistant tool_calls 段：维护"安全点"，所有 tool_call
       都被对应 tool_result 闭合时记录当前位置；后续出现未配对 tool_call
       且被新 user 打断或到达结尾时，回退到安全点丢弃整个未闭合段。
    4. 合并连续同角色文本条目。

    Args:
        entries: 原始快照条目。

    Returns:
        清洗后的条目列表。
    """
    entries = [e for e in entries if e.role != SNAPSHOT_ROLE_SYSTEM]
    if not entries:
        return []

    while entries and entries[0].role != SNAPSHOT_ROLE_USER:
        entries.pop(0)

    cleaned: list[SnapshotEntry] = []
    pending_ids: set[str] = set()
    safe_len = 0

    def record_safe_point() -> None:
        """当没有未配对 tool_call 时，更新安全点。"""
        nonlocal safe_len
        if not pending_ids:
            safe_len = len(cleaned)

    for entry in entries:
        if entry.role == SNAPSHOT_ROLE_USER:
            if pending_ids:
                del cleaned[safe_len:]
                pending_ids.clear()
            cleaned.append(entry)
            record_safe_point()
        elif entry.role == SNAPSHOT_ROLE_ASSISTANT and entry.tool_calls:
            for call in entry.tool_calls:
                call_id = str(call.get("id", "") or "")
                if call_id:
                    pending_ids.add(call_id)
            cleaned.append(entry)
        elif entry.role == SNAPSHOT_ROLE_TOOL_RESULT:
            if entry.call_id and entry.call_id in pending_ids:
                pending_ids.discard(entry.call_id)
            cleaned.append(entry)
            record_safe_point()
        else:
            cleaned.append(entry)
            record_safe_point()

    if pending_ids:
        del cleaned[safe_len:]
        pending_ids.clear()

    if not cleaned:
        return []
    merged: list[SnapshotEntry] = [cleaned[0]]
    for entry in cleaned[1:]:
        prev = merged[-1]
        if (
            prev.role == entry.role
            and not prev.tool_calls
            and not entry.tool_calls
            and prev.call_id == entry.call_id
        ):
            prev.content = _truncate(
                "\n".join(part for part in [prev.content, entry.content] if part)
            )
            continue
        merged.append(entry)
    return merged


def _extract_role(payload: Any) -> str:
    """提取 payload 角色字符串。"""
    role = getattr(payload, "role", None)
    if role is None:
        return ""
    value = getattr(role, "value", role)
    return str(value or "")


def _safe_text(value: Any) -> str:
    """把任意值安全转换为字符串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return ""


def _serialize_tool_call(part: Any) -> dict[str, Any]:
    """序列化单个 ToolCall。"""
    call_id = getattr(part, "id", None) or ""
    name = getattr(part, "name", "") or ""
    args = getattr(part, "args", {}) or {}
    return {
        "id": str(call_id),
        "name": str(name),
        "args": args,
    }


def _serialize_tool_result_value(value: Any) -> str:
    """序列化 ToolResult 的值。"""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return _safe_text(value)


def _render_unknown_part(part: Any) -> str:
    """渲染无法识别的 content 类型。"""
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        return json.dumps(part, ensure_ascii=False)
    return _safe_text(part)


def _truncate(text: str) -> str:
    """裁剪过长文本。"""
    if len(text) <= _MAX_ENTRY_CHARS:
        return text
    return text[:_MAX_ENTRY_CHARS].rstrip() + "..."


#: 匹配 user 文本中框架动态注入的 system_reminder 块（含标签本身）。
#: 这些 reminder 由框架的 reminder 管线在每次请求时重新注入，不属于
#: 对话历史，快照保存时应剥离，避免超长 reminder 挤占条目预算。
_SYSTEM_REMINDER_RE = re.compile(
    r"<system_reminder>.*?</system_reminder>",
    flags=re.DOTALL,
)


def _strip_system_reminders(text: str) -> str:
    """从 user 文本中剥离框架动态注入的 system_reminder 块。

    Args:
        text: 原始 user 文本。

    Returns:
        剥离 system_reminder 块后的文本。
    """
    stripped = _SYSTEM_REMINDER_RE.sub("", text)
    return stripped.strip()


def _merge_consecutive(entries: list[SnapshotEntry]) -> list[SnapshotEntry]:
    """合并相邻同角色条目，减少快照冗余。"""
    if not entries:
        return entries
    merged: list[SnapshotEntry] = [entries[0]]
    for entry in entries[1:]:
        prev = merged[-1]
        if (
            prev.role == entry.role
            and not prev.tool_calls
            and not entry.tool_calls
            and prev.call_id == entry.call_id
        ):
            prev.content = _truncate(
                "\n".join(part for part in [prev.content, entry.content] if part)
            )
            continue
        merged.append(entry)
    return merged


def _summarize_roles(entries: list[SnapshotEntry]) -> str:
    """统计快照条目的角色构成。"""
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.role] = counts.get(entry.role, 0) + 1
    return ", ".join(f"{role}×{count}" for role, count in counts.items())


__all__ = [
    "ContextSnapshot",
    "SnapshotEntry",
    "capture_snapshot_from_payloads",
    "deserialize_snapshot",
    "log_snapshot_restored",
    "serialize_payloads",
]
