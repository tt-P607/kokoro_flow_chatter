"""语音通话历史回填。

场景：用户在 KFC 私聊中触发语音通话后，该流由 anima_chatter 接管；
通话期间的对话只写入聊天流历史，不会进入 KFC 的持久化对话链。通话
结束时 anima_chatter 广播 ``voice_call.ended``，本处理器据此把整段
通话补回对话链。

**为什么打包成一对而非逐条补入**：对话链默认上限仅 20 条，一通 5 分钟
的通话轻易产生 10+ 条消息，逐条补入会吞掉一半额度。因此把整段通话压成
一对（1 user + 1 assistant）摘要——无论通话多长，链占用恒为 1 对。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.event_api import EventDecision
from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.base import BaseEventHandler

from ..config import KFCConfig
from ..domain.chain_entry import ChainEntry

if TYPE_CHECKING:
    from src.app.plugin_system.api.event_api import EventType

logger = get_logger("kfc_voice_call_handler")

VOICE_CALL_ENDED_EVENT = "voice_call.ended"

_KFC_SIGNATURE = "kokoro_flow_chatter:chatter:kokoro_flow_chatter"
"""KFC Chatter 的组件签名，用于判断通话前是否由 KFC 接管该流。"""

_BOT_PRE_CALL_LABEL = "（接通时）"
"""首条 user 消息之前的 bot 发言标签。"""

_EMPTY_TIMELINE_TEXT = "（通话期间没有任何对话发生。）"

_ASSISTANT_ACK_TEXT = "（已收到上面整段通话稿。）"


class VoiceCallHistoryHandler(BaseEventHandler):
    """把通话历史打包成一对摘要补回 KFC 对话链。"""

    name: str = "kfc_voice_call_history_handler"
    description: str = (
        "通话结束后把 anima_chatter 在 KFC 流上记录的整段对话打包成一对 "
        "user/assistant 摘要补回对话链，保证挂断后上下文连贯，"
        "且不会用一通通话挤占多个链槽位。"
    )
    weight: int = 0
    intercept_message: bool = False
    init_subscribe: list[EventType | str] = [VOICE_CALL_ENDED_EVENT]

    async def execute(
        self,
        event_name: str,
        params: dict[str, Any],
    ) -> tuple[EventDecision, dict[str, Any]]:
        """处理 ``voice_call.ended`` 事件。

        Args:
            event_name: 触发本处理器的事件名。
            params: 事件参数，需含通话流 ID 与通话消息列表。

        Returns:
            tuple: 事件决策与参数。
        """
        _ = event_name

        # 只处理通话前由 KFC 接管的流，避免改写其他 chatter 的会话
        if str(params.get("previous_chatter_signature") or "") != _KFC_SIGNATURE:
            return EventDecision.PASS, params

        stream_id = str(params.get("caller_stream_id") or "")
        if not stream_id:
            return EventDecision.PASS, params

        messages_in_call = params.get("messages_in_call") or []
        if not isinstance(messages_in_call, list) or not messages_in_call:
            logger.debug(f"通话无消息，跳过 stream={stream_id[:8]}")
            return EventDecision.PASS, params

        try:
            await self._patch_chain(stream_id, messages_in_call, params)
        except Exception as error:
            logger.error(
                f"补写对话链异常 stream={stream_id[:8]}: {error}",
                exc_info=True,
            )
            return EventDecision.PASS, params

        return EventDecision.SUCCESS, params

    async def _patch_chain(
        self,
        stream_id: str,
        messages_in_call: list[Any],
        event_params: dict[str, Any],
    ) -> None:
        """把整段通话写入 KFC 对话链。"""
        from ..plugin import KFCPlugin

        if not isinstance(self.plugin, KFCPlugin):
            logger.warning("处理器不在 KFCPlugin 上下文中，跳过")
            return

        config = self.plugin.config
        if not isinstance(config, KFCConfig):
            logger.warning("KFC 配置未加载，跳过对话链补丁")
            return

        user_summary, assistant_summary, first_user_ts = _summarize_call(
            messages_in_call
        )
        if not user_summary.strip() and not assistant_summary.strip():
            logger.debug(f"通话摘要为空 stream={stream_id[:8]}，跳过")
            return

        entries = [
            ChainEntry.user(text=user_summary, ts=first_user_ts).to_dict(),
            ChainEntry.assistant(text=assistant_summary).to_dict(),
        ]

        store = self.plugin.session_store
        async with store.lock(stream_id):
            session = await store.get_or_create(stream_id)
            session.update_chain(entries, config.prompt.max_context_payloads)
            await store.save(session)

        raw_count = sum(
            1
            for message in messages_in_call
            if isinstance(message, dict) and str(message.get("text") or "").strip()
        )
        duration = float(event_params.get("duration_seconds") or 0.0)
        logger.info(
            f"已把通话打包成 1 对链条目 stream={stream_id[:8]} "
            f"(原始消息 {raw_count} 条 / 持续 {duration:.0f}s)"
        )


def _summarize_call(messages_in_call: list[Any]) -> tuple[str, str, float]:
    """把整段通话压缩成按时间顺序编号的对话稿。

    与简单罗列不同，这里为每轮标注**轮次编号 + 角色**，让模型能清楚
    分辨每一轮谁说了什么、相对位置如何。轮次以 user 发声为界：bot 在
    首条 user 之前的发言（如接通寒暄）计入第 0 轮。

    user 摘要承载完整对话稿；assistant 摘要只写一句确认，既保持 chain
    的 user/assistant 交替合法，又避免把发言全文重复一遍。

    Args:
        messages_in_call: 形如 ``{"role", "text", "ts"}`` 的消息列表。

    Returns:
        tuple: ``(user 摘要, assistant 摘要, 首条 user 消息时间戳)``。
    """
    system_open = ""
    system_close = ""
    seen_system_count = 0
    timeline: list[tuple[str, str]] = []
    first_user_ts = 0.0
    fallback_ts = 0.0

    for message in messages_in_call:
        if not isinstance(message, dict):
            continue
        text = str(message.get("text") or "").strip()
        if not text:
            continue

        role = str(message.get("role") or "")
        raw_ts = message.get("ts")
        timestamp = (
            float(raw_ts) if isinstance(raw_ts, (int, float)) and raw_ts > 0 else 0.0
        )

        if role == "system":
            # 首条系统标注视为通话开始边界，其余覆盖为结束边界
            if seen_system_count == 0:
                system_open = text
            else:
                system_close = text
            seen_system_count += 1
            if fallback_ts == 0.0 and timestamp > 0:
                fallback_ts = timestamp
        elif role in ("user", "assistant"):
            timeline.append((role, text))
            if role == "user" and first_user_ts == 0.0 and timestamp > 0:
                first_user_ts = timestamp

    lines: list[str] = []
    round_no = 0
    for role, text in timeline:
        if role == "user":
            round_no += 1
            lines.append(f"【第 {round_no} 轮 / 用户】{text}")
        else:
            label = (
                _BOT_PRE_CALL_LABEL if round_no == 0 else f"（第 {round_no} 轮回应）"
            )
            lines.append(f"【你的回应{label}】{text}")

    timeline_block = "\n".join(lines) if lines else _EMPTY_TIMELINE_TEXT

    user_parts = [part for part in (system_open,) if part]
    user_parts.append("【通话对话稿（按时间顺序）】")
    user_parts.append(timeline_block)

    assistant_parts = [_ASSISTANT_ACK_TEXT]
    if system_close:
        assistant_parts.append(system_close)

    return (
        "\n\n".join(user_parts),
        "\n\n".join(assistant_parts),
        first_user_ts or fallback_ts or time.time(),
    )
