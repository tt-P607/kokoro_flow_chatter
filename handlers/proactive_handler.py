"""主动发起事件处理器。

订阅 ``kfc.proactive_trigger``，把系统触发消息注入目标流的未读队列并
唤醒流循环，从而端到端打通主动发起。

流不在内存中时执行冷启动：先从会话存档取回平台与账号信息重建流，
再显式拉起流循环——否则注入的消息不会被任何 tick 消费。
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.event_api import EventDecision
from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.stream_api import get_or_create_stream, get_stream
from src.app.plugin_system.base import BaseEventHandler
from src.app.plugin_system.types import Message

from ..framework_compat import start_stream_loop
from ..prompts.modules import build_proactive_context
from ..runtime.unread_policy import PROACTIVE_MESSAGE_PREFIX

if TYPE_CHECKING:
    from src.app.plugin_system.api.event_api import EventType
    from src.app.plugin_system.types import ChatStream

    from ..session import KFCSession

logger = get_logger("kfc_proactive_handler")

PROACTIVE_TRIGGER_EVENT = "kfc.proactive_trigger"
"""主动发起事件名，由插件的周期任务发布。"""

_RECENT_ACTIVITY_LIMIT = 5
"""构建近期活动摘要时回溯的历史消息条数。"""

_FALLBACK_CONTENT = "[主动发起] 你已经沉默很久了，主动找对方聊聊吧。"
"""富上下文构建失败时的兜底触发内容。"""

_SECONDS_PER_MINUTE = 60.0


class ProactiveHandler(BaseEventHandler):
    """响应主动发起事件，唤醒目标聊天流。"""

    name: str = "kfc_proactive_handler"
    description: str = "响应主动发起事件，唤醒目标聊天流"
    weight: int = 0
    intercept_message: bool = False
    init_subscribe: list[EventType | str] = [PROACTIVE_TRIGGER_EVENT]

    async def execute(
        self,
        event_name: str,
        params: dict[str, Any],
    ) -> tuple[EventDecision, dict[str, Any]]:
        """处理主动发起事件。

        Args:
            event_name: 触发本处理器的事件名。
            params: 事件参数，需含 ``stream_id``。

        Returns:
            tuple: 事件决策与参数。
        """
        _ = event_name
        stream_id = str(params.get("stream_id") or "")
        if not stream_id:
            return EventDecision.PASS, params

        try:
            if await self._wake_stream(
                stream_id,
                str(params.get("scheduled_reason") or ""),
            ):
                logger.info(f"主动发起: 流 {stream_id[:8]} 已唤醒")
            return EventDecision.SUCCESS, params
        except Exception as error:
            logger.error(f"主动发起处理异常: {error}", exc_info=True)
            return EventDecision.PASS, params

    async def _wake_stream(self, stream_id: str, scheduled_reason: str) -> bool:
        """向目标流注入触发消息并确保其循环在运行。

        Args:
            stream_id: 目标流 ID。
            scheduled_reason: 预约理由，由调用方在清除预约前读出。

        Returns:
            bool: 是否成功唤醒。
        """
        chat_stream = await get_stream(stream_id)
        is_cold_start = chat_stream is None

        if chat_stream is None:
            chat_stream = await self._cold_start_stream(stream_id)
            if chat_stream is None:
                return False

        session = await self._load_session(stream_id)
        target_user_id = session.user_id if session else ""
        silence_minutes = (
            (time.time() - session.last_activity_at) / _SECONDS_PER_MINUTE
            if session
            else 0.0
        )

        content = await self._build_trigger_content(
            chat_stream, silence_minutes, scheduled_reason
        )
        chat_stream.context.add_unread_message(
            _build_proactive_message(stream_id, chat_stream, target_user_id, content)
        )

        if is_cold_start:
            # 冷启动流没有正在运行的 tick，必须显式拉起循环
            try:
                await start_stream_loop(stream_id)
                logger.info(f"冷启动流 {stream_id[:8]} 循环已启动")
            except Exception as error:
                logger.warning(f"冷启动流 {stream_id[:8]} 启动循环失败: {error}")
        else:
            logger.debug(f"触发消息已注入热流 {stream_id[:8]}，等待下一次 tick 处理")
        return True

    async def _cold_start_stream(self, stream_id: str) -> ChatStream | None:
        """从会话存档重建不在内存中的聊天流。"""
        session = await self._load_session(stream_id, peek_only=True)
        if session is None:
            logger.warning(f"流 {stream_id[:8]} 不在内存中且无会话记录，跳过")
            return None

        try:
            chat_stream = await get_or_create_stream(
                stream_id=stream_id,
                platform=session.platform,
                user_id=session.user_id,
            )
        except Exception as error:
            logger.warning(f"冷启动流 {stream_id[:8]} 失败: {error}")
            return None

        logger.info(f"主动发起：冷启动流 {stream_id[:8]}")
        return chat_stream

    async def _load_session(
        self,
        stream_id: str,
        peek_only: bool = False,
    ) -> KFCSession | None:
        """读取目标流的会话。

        Args:
            stream_id: 目标流 ID。
            peek_only: 为真时不写入内存缓存。

        Returns:
            KFCSession | None: 会话；插件上下文不可用或无记录时为 ``None``。
        """
        from ..plugin import KFCPlugin

        if not isinstance(self.plugin, KFCPlugin):
            return None
        try:
            store = self.plugin.session_store
            return await (
                store.peek(stream_id) if peek_only else store.get(stream_id)
            )
        except Exception as error:
            logger.debug(f"读取会话失败: {error}")
            return None

    @staticmethod
    async def _build_trigger_content(
        chat_stream: ChatStream,
        silence_minutes: float,
        scheduled_reason: str,
    ) -> str:
        """构建注入的触发内容；失败时退回兜底文本。"""
        history = chat_stream.context.history_messages
        recent_messages = history[-_RECENT_ACTIVITY_LIMIT:] if history else []
        recent_activity = "\n".join(
            f"{message.sender_name or '未知'}: "
            f"{message.processed_plain_text or str(message.content or '')}"
            for message in recent_messages
        )

        try:
            return await build_proactive_context(
                silence_minutes=silence_minutes,
                recent_activity=recent_activity,
                scheduled_reason=scheduled_reason,
            )
        except Exception as error:
            logger.debug(f"构建主动发起上下文失败，使用兜底消息: {error}")
            return _FALLBACK_CONTENT


def _build_proactive_message(
    stream_id: str,
    chat_stream: ChatStream,
    target_user_id: str,
    content: str,
) -> Message:
    """构造一条主动发起用的系统触发消息。

    消息 ID 带 ``proactive_`` 前缀，使未读策略能识别其内部来源——
    与真实消息撞车时让位，也永不打断正在进行的生成。

    Args:
        stream_id: 目标流 ID。
        chat_stream: 目标聊天流。
        target_user_id: 目标用户的平台账号，用于消息路由。
        content: 注入的触发内容。

    Returns:
        Message: 系统触发消息。
    """
    extra_kwargs: dict[str, Any] = {}
    if target_user_id:
        extra_kwargs["target_user_id"] = target_user_id

    return Message(
        message_id=f"{PROACTIVE_MESSAGE_PREFIX}{uuid.uuid4().hex[:12]}",
        platform=chat_stream.platform or "unknown",
        stream_id=stream_id,
        sender_id=target_user_id or "system",
        sender_name="系统",
        content=content,
        processed_plain_text=content,
        time=time.time(),
        **extra_kwargs,
    )
