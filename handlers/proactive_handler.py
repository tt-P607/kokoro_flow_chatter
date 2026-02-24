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
from src.core.components.base.event_handler import BaseEventHandler

if TYPE_CHECKING:
    from src.core.components.base.plugin import BasePlugin

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
    init_subscribe: list[str] = [_PROACTIVE_EVENT]  # type: ignore[assignment]

    async def execute(
        self, kwargs: dict[str, Any] | None
    ) -> tuple[bool, bool, str | None]:
        """处理主动发起事件。

        流程:
        1. 从事件参数中提取 stream_id
        2. 获取目标流的 StreamContext
        3. 构造系统触发 Message 并塞入 unread_messages
        4. 清除 StreamLoopManager 的 _wait_states 等待锁，唤醒流循环

        Args:
            kwargs: 事件参数，需包含 ``stream_id`` 字段

        Returns:
            tuple[bool, bool, str | None]: (成功, 不拦截, 描述)
        """
        if not kwargs:
            return False, False, "事件参数为空"

        stream_id = kwargs.get("stream_id")
        if not stream_id:
            return False, False, "缺少 stream_id"

        try:
            success = await self._wake_stream(stream_id)
            if success:
                logger.info(f"主动发起: 流 {stream_id[:8]} 已唤醒")
                return True, False, f"已唤醒流 {stream_id[:8]}"
            return False, False, f"唤醒流 {stream_id[:8]} 失败"
        except Exception as e:
            logger.error(f"主动发起处理异常: {e}", exc_info=True)
            return False, False, f"处理异常: {e}"

    async def _wake_stream(self, stream_id: str) -> bool:
        """向目标流注入触发消息并唤醒流循环。

        Args:
            stream_id: 目标聊天流 ID

        Returns:
            bool: 是否成功唤醒
        """
        from src.core.managers.stream_manager import get_stream_manager

        sm = get_stream_manager()
        chat_stream = sm._streams.get(stream_id)  # HACK: 需要框架公开 API (stream_manager.get_stream)
        if not chat_stream:
            logger.warning(f"目标流 {stream_id[:8]} 不在内存中，跳过")
            return False

        context = chat_stream.context

        # 构造系统触发消息
        trigger_message = self._build_proactive_message(stream_id, chat_stream)
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
    def _build_proactive_message(stream_id: str, chat_stream: Any) -> Any:
        """构造一条用于主动发起的系统触发消息。

        Args:
            stream_id: 目标流 ID
            chat_stream: 聊天流对象

        Returns:
            Message: 系统触发消息对象
        """
        from src.core.models.message import Message

        return Message(
            message_id=f"proactive_{uuid.uuid4().hex[:12]}",
            platform=chat_stream.platform or "unknown",
            stream_id=stream_id,
            sender_id="system",
            sender_name="系统",
            content="[主动发起] 你已经沉默很久了，主动找对方聊聊吧。",
            processed_plain_text="[主动发起] 你已经沉默很久了，主动找对方聊聊吧。",
            time=time.time(),
        )
