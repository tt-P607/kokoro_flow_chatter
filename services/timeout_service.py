"""KFC 等待超时服务。

管理"发完消息后等待回复"的超时判定与状态推进：检测超时、递增连续
超时计数、写入活动流事件，并构建供模型决策的超时 payload。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.types import LLMPayload

from ..mental_log import MentalLogEntry
from ..models import KFCEventType
from ..prompts.modules import build_timeout_payload

if TYPE_CHECKING:
    from ..config import KFCConfig
    from ..session import KFCSession

logger = get_logger("kfc_timeout")


@dataclass(slots=True)
class TimeoutResult:
    """一次超时处理的产出。"""

    payload: LLMPayload
    """供模型决策的超时提示 payload，由调用方追加到上下文。"""

    is_final_timeout: bool
    """是否为最后一次超时——为真时主循环必须强制结束等待。"""


class TimeoutService:
    """等待超时的判定与状态推进。

    ``build_timeout_result()`` 会修改 session（递增计数、清除等待、写
    活动流），但**不会**触碰 ``response.payloads``——payload 的追加时机
    由主循环掌握。
    """

    def __init__(self, config: KFCConfig) -> None:
        """初始化服务。

        Args:
            config: KFC 配置，提供连续超时上限。
        """
        self._config = config

    def check_timeout(self, session: KFCSession) -> bool:
        """检查会话的等待是否已超时。"""
        return session.waiting_config.is_timeout()

    def build_timeout_result(self, session: KFCSession) -> TimeoutResult:
        """处理一次超时并构建决策 payload。

        副作用：递增 ``consecutive_timeout_count``、写入 ``WAIT_TIMEOUT``
        活动流事件、清除等待状态。

        Args:
            session: 当前会话。

        Returns:
            TimeoutResult: 超时 payload 与是否为最后一次超时。
        """
        elapsed = session.waiting_config.get_elapsed_seconds()
        expected_reaction = session.waiting_config.expected_reaction
        session.consecutive_timeout_count += 1

        session.mental_log.add(
            MentalLogEntry(
                event_type=KFCEventType.WAIT_TIMEOUT,
                timestamp=time.time(),
                elapsed_seconds=elapsed,
                content=f"等待超时，已等待 {elapsed:.0f} 秒",
            )
        )
        last_bot_message = session.mental_log.get_last_bot_reply_content()
        session.clear_waiting()

        max_timeouts = self._config.wait.max_consecutive_timeouts
        is_final_timeout = session.consecutive_timeout_count >= max_timeouts

        logger.info(
            f"等待超时: stream={session.stream_id[:8]}, "
            f"elapsed={elapsed:.0f}s, "
            f"consecutive={session.consecutive_timeout_count}/{max_timeouts}"
        )

        payload = build_timeout_payload(
            elapsed_seconds=elapsed,
            expected_reaction=expected_reaction,
            consecutive_timeouts=session.consecutive_timeout_count,
            last_bot_message=last_bot_message,
            max_consecutive_timeouts=max_timeouts,
        )
        return TimeoutResult(payload=payload, is_final_timeout=is_final_timeout)
