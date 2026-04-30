"""KFC 历史上下文 source 辅助。"""

from __future__ import annotations

import datetime
from typing import Any

from src.kernel.llm import LLMPayload, ROLE, Text


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
    """从序列化的 USER/ASSISTANT pair 恢复 payload。"""
    payloads: list[LLMPayload] = []
    for entry in serialized_chain_payloads:
        role_str = str(entry.get("role", "") or "")
        text = str(entry.get("text", "") or "")
        if not text:
            continue
        if role_str == "user":
            payloads.append(LLMPayload(ROLE.USER, Text(text)))
        elif role_str == "assistant":
            payloads.append(LLMPayload(ROLE.ASSISTANT, Text(text)))

    while payloads and payloads[0].role == ROLE.ASSISTANT:
        payloads.pop(0)
    return payloads


def build_fused_narrative(
    chat_stream: Any,
    mental_log: Any,
    before_ts: float | None = None,
) -> str:
    """构建聊天历史与内心独白的融合叙事。"""
    from ...models import KFCEventType

    msgs: list[Any] = list(
        getattr(
            getattr(chat_stream, "context", None),
            "history_messages",
            [],
        )
    )
    bot_id = str(chat_stream.bot_id or "")
    timeline: list[tuple[float, str]] = []

    for msg in msgs:
        raw_time = getattr(msg, "time", None)
        if not isinstance(raw_time, (int, float)):
            continue
        ts = float(raw_time)
        try:
            time_str = datetime.datetime.fromtimestamp(ts).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except (OSError, ValueError, OverflowError):
            continue

        sender = getattr(msg, "sender_name", "未知")
        sender_id = str(getattr(msg, "sender_id", ""))
        message_id = str(getattr(msg, "message_id", "") or "")
        text = getattr(msg, "processed_plain_text", "")
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
    mental_cutoff = chat_timestamps[-7] if len(chat_timestamps) >= 7 else 0.0

    for entry in getattr(mental_log, "entries", []) or []:
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