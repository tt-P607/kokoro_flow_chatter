"""KFC 历史上下文 source。

负责把聊天记录和心理活动流渲染成当前请求动态背景。
核心产物是 ``build_fused_narrative()``——把"说了什么"和"当时在想什么"
按时间线交织成统一叙事，这是 KFC 与普通聊天器的关键差异。
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.types import LLMPayload, ROLE, Text

from ...models import KFCEventType

if TYPE_CHECKING:
    from src.app.plugin_system.types import ChatStream, Message

    from ...mental_log import MentalLog

_MENTAL_LOG_LOOKBACK = 7
"""心理活动流的回溯窗口：取最近 N 条消息的时间戳为剪切点，
使内心独白只覆盖近期对话，避免久远的想法混入叙事。"""

_BOT_ACTION_MESSAGE_PREFIX = "action_kfc_reply"
"""KFC 自身发出的消息 ID 前缀，用于在 ``sender_id`` 缺失时识别 bot 消息。"""

_NARRATIVE_HEADER = "以下为融合了聊天记录与你内心活动的时间线："

_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _resolve_partner_name(chat_stream: ChatStream) -> str:
    """解析当前私聊对象的展示名。

    ``ChatStream.stream_name`` 在私聊场景下即为对方昵称；未设置时回退
    为中性称呼，避免提示词里出现空字符串。
    """
    return chat_stream.stream_name or "对方"


def _format_timestamp(timestamp: float) -> str | None:
    """格式化时间戳；越界或非法时返回 ``None``。"""
    try:
        return datetime.datetime.fromtimestamp(timestamp).strftime(_TIME_FORMAT)
    except (OSError, ValueError, OverflowError):
        return None


def build_history_summary_payload(
    chat_stream: ChatStream,
    history_summary: str,
) -> LLMPayload | None:
    """把近期记忆摘要渲染为 payload；摘要为空时返回 ``None``。"""
    summary = history_summary.strip()
    if not summary:
        return None
    return LLMPayload(
        ROLE.SYSTEM,
        Text(f"【你对{_resolve_partner_name(chat_stream)}的近期记忆】\n{summary}"),
    )


def build_current_time_payload(
    now: datetime.datetime | None = None,
) -> LLMPayload:
    """渲染当前时间 payload，作为无历史时的兜底上下文。"""
    current = now or datetime.datetime.now()
    return LLMPayload(
        ROLE.USER,
        Text(f"当前时间：{current.strftime('%Y-%m-%d %H:%M')}"),
    )


def build_channel_payload(chat_stream: ChatStream) -> LLMPayload:
    """渲染当前通道参数 payload。

    明确告知模型平台、聊天类型与对话对象，同时声明这些只是通道参数，
    抑制其据此脑补物理场景。
    """
    platform = chat_stream.platform or "unknown"
    chat_type = str(chat_stream.chat_type or "unknown")
    bot_id = chat_stream.bot_id or ""
    bot_nickname = chat_stream.bot_nickname or ""
    target_name = _resolve_partner_name(chat_stream)

    lines = [
        "[当前通道参数]",
        f"聊天平台：{platform}",
        f"聊天类型：{chat_type}",
    ]
    if bot_nickname or bot_id:
        lines.append(
            f"你的通道身份：昵称 {bot_nickname or '未知'}，ID {bot_id or '未知'}"
        )
    lines.append(f"当前私聊对象：{target_name}")
    lines.append(f"- 当前是一对一私聊，正在与你对话的是{target_name}。")
    return LLMPayload(ROLE.USER, Text("\n".join(lines)))


def _is_bot_message(message: Message, bot_id: str) -> bool:
    """判断消息是否由 bot 自己发出。"""
    if bot_id and message.sender_id == bot_id:
        return True
    return bool(
        message.message_id
        and message.message_id.startswith(_BOT_ACTION_MESSAGE_PREFIX)
    )


def _collect_message_timeline(
    chat_stream: ChatStream,
    before_ts: float | None,
) -> list[tuple[float, str]]:
    """把聊天记录收集为 ``(时间戳, 渲染行)`` 序列。"""
    bot_id = chat_stream.bot_id or ""
    timeline: list[tuple[float, str]] = []

    for message in chat_stream.context.history_messages:
        raw_time = message.time
        if not isinstance(raw_time, (int, float)):
            continue
        timestamp = float(raw_time)
        if before_ts is not None and timestamp >= before_ts:
            continue

        text = (message.processed_plain_text or "").strip()
        if not text:
            continue

        time_str = _format_timestamp(timestamp)
        if time_str is None:
            continue

        if _is_bot_message(message, bot_id):
            timeline.append((timestamp, f"[{time_str}] 你回复：{text}"))
            continue

        sender = message.sender_name or "未知"
        message_id = message.message_id or ""
        id_part = f" [消息id:{message_id}]" if message_id else ""
        timeline.append((timestamp, f"[{time_str}] {sender}{id_part}说：{text}"))

    return timeline


def _collect_thought_timeline(
    mental_log: MentalLog | None,
    cutoff_ts: float,
    before_ts: float | None,
) -> list[tuple[float, str]]:
    """把心理活动流中的内心独白收集为 ``(时间戳, 渲染行)`` 序列。"""
    if mental_log is None:
        return []

    timeline: list[tuple[float, str]] = []
    for entry in mental_log.entries:
        if entry.event_type != KFCEventType.BOT_PLANNING or not entry.thought:
            continue
        if not isinstance(entry.timestamp, (int, float)):
            continue
        if entry.timestamp < cutoff_ts:
            continue
        if before_ts is not None and entry.timestamp >= before_ts:
            continue

        time_str = _format_timestamp(entry.timestamp)
        if time_str is None:
            continue
        timeline.append((entry.timestamp, f"[{time_str}] （你的内心：{entry.thought}）"))

    return timeline


def build_fused_narrative(
    chat_stream: ChatStream,
    mental_log: MentalLog | None,
    before_ts: float | None = None,
) -> str:
    """构建聊天记录与内心独白的融合叙事。

    消息来源为 ``context.history_messages``（受核心配置的上下文长度管控）；
    它与 ``context_snapshot`` 都用于背景构建，输出只作为当前请求动态背景。

    Args:
        chat_stream: 当前聊天流。
        mental_log: 心理活动流；``None`` 时只渲染聊天记录。
        before_ts: 只包含时间戳严格小于该值的内容；缺省时包含全部记录。

    Returns:
        str: 融合叙事文本；无内容时返回空串。
    """
    message_timeline = _collect_message_timeline(chat_stream, before_ts)

    message_timestamps = [timestamp for timestamp, _ in message_timeline]
    cutoff_ts = (
        message_timestamps[-_MENTAL_LOG_LOOKBACK]
        if len(message_timestamps) >= _MENTAL_LOG_LOOKBACK
        else 0.0
    )
    timeline = message_timeline + _collect_thought_timeline(
        mental_log, cutoff_ts, before_ts
    )
    if not timeline:
        return ""

    timeline.sort(key=lambda item: item[0])
    return _NARRATIVE_HEADER + "\n" + "\n".join(line for _, line in timeline)
