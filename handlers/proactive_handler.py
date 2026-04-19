"""主动发起事件处理器。

订阅 ``kfc.proactive_trigger`` 事件，
将系统触发消息注入目标流的 unread_messages 并唤醒流循环，
从而端到端打通主动发起功能。
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.base import BaseEventHandler
from src.app.plugin_system.api.event_api import EventDecision

if TYPE_CHECKING:
    from src.app.plugin_system.api.event_api import EventType

logger = get_logger("kfc_proactive_handler")

# 主动发起事件名
_PROACTIVE_EVENT = "kfc.proactive_trigger"


class ProactiveHandler(BaseEventHandler):
    """主动发起事件处理器。

    当定时器检测到满足主动发起条件后，通过 EventBus 发布
    ``kfc.proactive_trigger`` 事件。本处理器接收该事件，
    向目标流注入一条系统触发消息并唤醒流循环。
    """

    handler_name: str = "kfc_proactive_handler"
    handler_description: str = "响应主动发起事件，唤醒目标聊天流"
    weight: int = 0
    intercept_message: bool = False
    init_subscribe: list[EventType | str] = [_PROACTIVE_EVENT]

    async def execute(
        self, event_name: str, params: dict[str, Any]
    ) -> tuple[EventDecision, dict[str, Any]]:
        """处理主动发起事件。

        流程:
        1. 从事件参数中提取 stream_id
        2. 获取目标流的 StreamContext
        3. 构造系统触发 Message 并塞入 unread_messages
        4. 清除 StreamLoopManager 的 _wait_states 等待锁，唤醒流循环

        Args:
            event_name: 触发本处理器的事件名称
            params: 事件参数，需包含 ``stream_id`` 字段

        Returns:
            tuple[EventDecision, dict[str, Any]]: 决策与参数
        """
        stream_id = params.get("stream_id")
        if not stream_id:
            return EventDecision.PASS, params

        try:
            success = await self._wake_stream(stream_id)
            if success:
                logger.info(f"主动发起: 流 {stream_id[:8]} 已唤醒")
            return EventDecision.SUCCESS, params
        except Exception as e:
            logger.error(f"主动发起处理异常: {e}", exc_info=True)
            return EventDecision.PASS, params

    async def _wake_stream(self, stream_id: str) -> bool:
        """向目标流注入触发消息并唤醒流循环。

        Args:
            stream_id: 目标聊天流 ID

        Returns:
            bool: 是否成功唤醒
        """
        from src.app.plugin_system.api.stream_api import get_stream

        chat_stream = await get_stream(stream_id)
        if not chat_stream:
            logger.warning(f"目标流 {stream_id[:8]} 不在内存中，跳过")
            return False

        context = chat_stream.context

        # 尝试从 KFCSession 获取真实 user_id 和沉默时长
        target_user_id: str = ""
        silence_minutes: float = 0.0
        try:
            from ..plugin import KFCPlugin
            if isinstance(self.plugin, KFCPlugin):
                session = await self.plugin._session_store.get(stream_id)  # type: ignore[attr-defined]
                if session:
                    if session.user_id:
                        target_user_id = session.user_id
                    if session.last_activity_at:
                        silence_minutes = (time.time() - session.last_activity_at) / 60
        except Exception as e:
            logger.debug(f"获取 session 信息失败，将使用默认值: {e}")

        # 从近期历史消息构建近期活动摘要
        recent_activity = ""
        try:
            recent_msgs = context.history_messages[-5:] if context.history_messages else []
            if recent_msgs:
                lines = []
                for msg in recent_msgs:
                    sender = getattr(msg, "sender_name", "") or "未知"
                    content = getattr(msg, "processed_plain_text", "") or str(getattr(msg, "content", ""))
                    lines.append(f"{sender}: {content}")
                recent_activity = "\n".join(lines)
        except Exception as e:
            logger.debug(f"构建近期活动摘要失败: {e}")

        # 调用 build_proactive_context 生成富上下文提示词
        proactive_content = "[主动发起] 你已经沉默很久了，主动找对方聊聊吧。"
        try:
            from ..prompts.modules import build_proactive_context
            proactive_content = await build_proactive_context(
                silence_minutes=silence_minutes,
                recent_activity=recent_activity,
            )
        except Exception as e:
            logger.debug(f"构建主动发起上下文失败，使用默认消息: {e}")

        # 构造系统触发消息
        trigger_message = self._build_proactive_message(stream_id, chat_stream, target_user_id, proactive_content)
        context.add_unread_message(trigger_message)
        logger.debug(f"已注入主动发起触发消息到流 {stream_id[:8]}")

        # 清除 StreamLoopManager 的等待状态，让下一次 tick 立即唤醒
        try:
            from src.core.transport.distribution.stream_loop_manager import (
                get_stream_loop_manager,
            )

            loop_mgr = get_stream_loop_manager()
            removed = loop_mgr._wait_states.pop(stream_id, None)  # HACK: 需要框架公开 API (loop_mgr.wake_stream)
            if removed:
                logger.debug(f"已清除流 {stream_id[:8]} 的等待状态")
        except ImportError:
            logger.warning("StreamLoopManager 不可用，无法清除等待状态")

        return True

    @staticmethod
    def _build_proactive_message(
        stream_id: str,
        chat_stream: Any,
        target_user_id: str = "",
        content: str = "[主动发起] 你已经沉默很久了，主动找对方聊聊吧。",
    ) -> Any:
        """构造一条用于主动发起的系统触发消息。

        Args:
            stream_id: 目标流 ID
            chat_stream: 聊天流对象
            target_user_id: 目标用户的真实 ID（QQ 号等），用于消息路由
            content: 注入的消息内容（默认为简单提示，推荐传入 build_proactive_context 生成的富上下文）

        Returns:
            Message: 系统触发消息对象
        """
        from src.core.models.message import Message

        extra_kwargs: dict[str, Any] = {}
        if target_user_id:
            extra_kwargs["target_user_id"] = target_user_id

        return Message(
            message_id=f"proactive_{uuid.uuid4().hex[:12]}",
            platform=chat_stream.platform or "unknown",
            stream_id=stream_id,
            sender_id=target_user_id or "system",
            sender_name="系统",
            content=content,
            processed_plain_text=content,
            time=time.time(),
            **extra_kwargs,
        )
