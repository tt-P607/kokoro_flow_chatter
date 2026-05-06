"""KFC 历史上下文 source 辅助。"""

from __future__ import annotations

import datetime
from typing import Any

from src.app.plugin_system.types import LLMPayload, ROLE, Text, ToolCall, ToolResult

from ...domain.chain_entry import ChainEntry
from ...models import KFCEventType

# mental_log 回溯的对话消息条数：取最近 N 条消息的时间戳作为剪切点，
# 使 mental_log 中的思考记录仅覆盖近期对话窗口。
_MENTAL_LOG_LOOKBACK = 7


def build_history_summary_payload(
    chat_stream: Any,
    history_summary: str,
) -> LLMPayload | None:
    """将 session.history_summary 渲染为 SYSTEM payload。"""
    summary = history_summary.strip()
    if not summary:
        return None

    user_name = (
        getattr(chat_stream, "partner_name", None)
        or getattr(chat_stream, "group_name", None)
        or "对方"
    )
    return LLMPayload(
        ROLE.SYSTEM,
        Text(f"【你对{user_name}的近期记忆】\n{summary}"),
    )


def build_current_time_payload(
    now: datetime.datetime | None = None,
) -> LLMPayload:
    """在无可用历史时渲染当前时间 payload。"""
    current = now or datetime.datetime.now()
    return LLMPayload(
        ROLE.SYSTEM,
        Text(f"当前时间：{current.strftime('%Y-%m-%d %H:%M')}")
    )


def restore_chain_payloads(
    serialized_chain_payloads: list[dict[str, Any]],
) -> list[LLMPayload]:
    """从序列化的 USER/ASSISTANT pair 恢复 payload。

    输入 ``list[dict]`` 经 :class:`ChainEntry` 校验后再渲染，
    保证 ASSISTANT(tool_calls) → TOOL_RESULT 链路一定有 ASSISTANT 桥接。
    """
    entries: list[ChainEntry] = []
    for raw in serialized_chain_payloads:
        entry = ChainEntry.from_dict(raw)
        if entry is not None:
            entries.append(entry)

    payloads: list[LLMPayload] = []
    for entry in entries:
        if entry.is_user:
            payloads.append(LLMPayload(ROLE.USER, Text(entry.text)))
            continue
        # assistant
        if entry.has_tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.get("id"),
                    name=tc["name"],
                    args=tc.get("args", {}),
                )
                for tc in entry.tool_calls
            ]
            # 格式：ASSISTANT(ToolCall) → TOOL_RESULT → ASSISTANT(text) → ...
            payloads.append(LLMPayload(ROLE.ASSISTANT, list(tool_calls)))  # type: ignore[arg-type]
            payloads.append(
                LLMPayload(
                    ROLE.TOOL_RESULT,
                    [
                        ToolResult(value="ok", call_id=tc.id, name=tc.name)
                        for tc in tool_calls
                    ],  # type: ignore[arg-type]
                )
            )
            payloads.append(LLMPayload(ROLE.ASSISTANT, Text(entry.text)))
        else:
            payloads.append(LLMPayload(ROLE.ASSISTANT, Text(entry.text)))

    while payloads and payloads[0].role == ROLE.ASSISTANT:
        payloads.pop(0)
    return payloads


def build_fused_narrative(
    chat_stream: Any,
    mental_log: Any,
    before_ts: float | None = None,
) -> str:
    """构建聊天历史与内心独白的融合叙事。"""
    msgs: list[Any] = list(chat_stream.context.history_messages)
    bot_id = str(chat_stream.bot_id or "")
    timeline: list[tuple[float, str]] = []

    for msg in msgs:
        raw_time = msg.time
        if not isinstance(raw_time, (int, float)):
            continue
        ts = float(raw_time)
        try:
            time_str = datetime.datetime.fromtimestamp(ts).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except (OSError, ValueError, OverflowError):
            continue

        sender = msg.sender_name or "未知"
        sender_id = msg.sender_id or ""
        message_id = msg.message_id or ""
        text = msg.processed_plain_text or ""
        if not text or not text.strip():
            continue

        if before_ts is not None and ts >= before_ts:
            continue

        is_bot = bool(
            (bot_id and sender_id == bot_id)
            or message_id.startswith("action_kfc_reply")
        )
        if is_bot:
            timeline.append((ts, f"[{time_str}] 你回复：{text}"))
        else:
            msg_id_part = f" [消息id:{message_id}]" if message_id else ""
            timeline.append((ts, f"[{time_str}] {sender}{msg_id_part}说：{text}"))

    chat_timestamps = [ts for ts, _ in timeline]
    mental_cutoff = chat_timestamps[-_MENTAL_LOG_LOOKBACK] if len(chat_timestamps) >= _MENTAL_LOG_LOOKBACK else 0.0

    for entry in (mental_log.entries if mental_log else []):
        if entry.timestamp < mental_cutoff:
            continue
        ts = entry.timestamp
        try:
            time_str = datetime.datetime.fromtimestamp(ts).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except (OSError, ValueError, OverflowError):
            continue

        if before_ts is not None and entry.timestamp >= before_ts:
            continue

        if entry.event_type == KFCEventType.BOT_PLANNING and entry.thought:
            timeline.append((ts, f"[{time_str}] （你的内心：{entry.thought}）"))

    if not timeline:
        return ""

    timeline.sort(key=lambda item: item[0])
    lines = [item[1] for item in timeline]
    return "以下为融合了聊天记录与你内心活动的时间线：\n" + "\n".join(lines)