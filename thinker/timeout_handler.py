"""超时处理器。

TimeoutHandler 在等待超时后生成决策提示，
供 Chatter 的 execute() 在下一轮循环中使用。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.app.plugin_system.api.log_api import get_logger

from ..mental_log import MentalLogEntry
from ..models import KFCEventType

if TYPE_CHECKING:
    from ..config import KFCConfig
    from ..session import KFCSession

logger = get_logger("kfc_timeout")


class TimeoutHandler:
    """等待超时处理器。

    当 Session 的 WaitingConfig 超时后，更新 Session 状态
    并准备超时上下文。
    """

    def __init__(self, config: KFCConfig) -> None:
        self._config = config

    def check_timeout(self, session: KFCSession) -> bool:
        """检查 Session 是否超时。

        Args:
            session: 当前会话状态

        Returns:
            bool: 是否已超时
        """
        return session.waiting_config.is_timeout()

    def handle_timeout(self, session: KFCSession) -> dict[str, object]:
        """处理超时，更新 Session 状态并返回超时上下文。

        Args:
            session: 当前会话状态

        Returns:
            dict: 超时上下文信息
        """
        elapsed = session.waiting_config.get_elapsed_seconds()
        expected = session.waiting_config.expected_reaction
        session.consecutive_timeout_count += 1

        # 记录超时到活动流
        timeout_entry = MentalLogEntry(
            event_type=KFCEventType.WAIT_TIMEOUT,
            timestamp=time.time(),
            elapsed_seconds=elapsed,
            content=f"等待超时，已等待 {elapsed:.0f} 秒",
        )
        session.mental_log.add(timeout_entry)

        # 收集等待期间的想法
        pending = list(session.pending_thoughts)

        # 提取最后一条 Bot 发送的消息
        last_bot_message = session.mental_log.get_last_bot_reply_content()

        # 清除等待状态
        session.clear_waiting()

        context = {
            "elapsed_seconds": elapsed,
            "expected_reaction": expected,
            "consecutive_timeouts": session.consecutive_timeout_count,
            "pending_thoughts": pending,
            "last_bot_message": last_bot_message,
        }

        logger.info(
            f"等待超时: stream={session.stream_id[:8]}, "
            f"elapsed={elapsed:.0f}s, "
            f"consecutive={session.consecutive_timeout_count}"
        )

        return context

    def should_give_up(self, session: KFCSession) -> bool:
        """判断是否应该放弃等待（连续超时次数过多）。

        注意：此方法应在 ``handle_timeout()`` 之后调用。
        ``handle_timeout()`` 会先递增 ``consecutive_timeout_count``，
        因此第 N 次超时触发时 count 已为 N。
        当 ``count >= max_consecutive_timeouts`` 时返回 True，
        即允许最多 ``max_consecutive_timeouts - 1`` 次超时后的重试。

        例如 ``max_consecutive_timeouts=3`` 时：
        - 第 1 次超时: count=1, 继续
        - 第 2 次超时: count=2, 继续
        - 第 3 次超时: count=3, 放弃

        Args:
            session: 当前会话状态

        Returns:
            bool: 是否应放弃等待
        """
        max_timeouts = self._config.wait.max_consecutive_timeouts
        return session.consecutive_timeout_count >= max_timeouts
