"""KFC 历史上下文 source 辅助。"""

from __future__ import annotations

import datetime
from typing import Any

from src.app.plugin_system.types import LLMPayload, ROLE, Text

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
    """在动态 USER 上下文中渲染当前时间 payload。"""
    current = now or datetime.datetime.now()
    return LLMPayload(
        ROLE.USER,
        Text(f"当前时间：{current.strftime('%Y-%m-%d %H:%M')}")
    )


def build_channel_payload(chat_stream: Any) -> LLMPayload:
    """在动态 USER 上下文中渲染平台与通道参数。"""
    platform = str(getattr(chat_stream, "platform", "") or "unknown")
    chat_type = str(getattr(chat_stream, "chat_type", "") or "unknown")
    bot_id = str(getattr(chat_stream, "bot_id", "") or "")
    nickname = str(getattr(chat_stream, "bot_nickname", "") or "")
    lines = [
        "[当前通道参数]",
        f"聊天平台：{platform}",
        f"聊天类型：{chat_type}",
    ]
    if nickname or bot_id:
        lines.append(f"你的通道身份：昵称 {nickname or '未知'}，ID {bot_id or '未知'}")
    lines.extend(
        [
            "- 上述平台/聊天类型/ID 只是通道参数。除非有明确证据，否则不要自行脑补手机、屏幕、房间等物理场景细节。",
            "- 进行角色扮演时，应优先依据双方关系、语境和时间来组织描写。",
        ]
    )
    return LLMPayload(ROLE.USER, Text("\n".join(lines)))


def restore_chain_payloads(
    serialized_chain_payloads: list[dict[str, Any]],
) -> list[LLMPayload]:
    """从序列化的 USER/ASSISTANT pair 恢复可读历史 payload。

    ``chain_payloads`` 中的 ``tool_calls`` 只作为审计/调试数据持久化，
    不再还原为 ``ASSISTANT(tool_calls) -> TOOL_RESULT -> ASSISTANT(text)``。
    原还原方式会把 ``kfc_reply.content`` 暴露在 tool call 参数中，同时又把
    同一回复作为 assistant 文本放入上下文，造成模型输入层面的重复。

    运行期尚未完成的 tool-call 链仍由 ``response.payloads`` 保持；跨 execute
    重载的历史链只需要保留用户可读对话文本，动作细节由 ``mental_log`` 提供。
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
        if not isinstance(entry.timestamp, (int, float)):
            continue
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
