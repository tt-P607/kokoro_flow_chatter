"""主动发起模块。

ProactiveThinker 负责在长时间沉默后评估是否主动发起对话。
通过 Scheduler 定期调度。
"""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING

from src.app.plugin_system.api.log_api import get_logger

if TYPE_CHECKING:
    from ..config import KFCConfig
    from ..session import KFCSession, KFCSessionStore

logger = get_logger("kfc_proactive")


class ProactiveThinker:
    """主动发起思考器。

    检查所有活跃 Session 的沉默时长，
    在满足条件时通过事件总线触发主动对话。
    """

    def __init__(
        self,
        config: KFCConfig,
        session_store: KFCSessionStore,
    ) -> None:
        self._config = config
        self._session_store = session_store

    async def check_all_sessions(self) -> list[str]:
        """检查所有缓存中的 Session，返回需要主动发起的 stream_id 列表。"""
        proactive_config = self._config.proactive
        if not proactive_config.enabled:
            return []

        # 检查是否在勿扰时段
        if self._is_quiet_hours():
            return []

        triggered: list[str] = []
        sessions = self._session_store.get_all_cached()

        for stream_id, session in sessions.items():
            if self._should_trigger(session):
                triggered.append(stream_id)

        return triggered

    def _should_trigger(self, session: KFCSession) -> bool:
        """判断指定 Session 是否应主动发起。

        Args:
            session: KFC 会话对象
        """
        proactive_config = self._config.proactive
        now = time.time()

        # 检查最后活动时间
        silence_duration = now - session.last_activity_at
        if silence_duration < proactive_config.silence_threshold:
            return False

        # 检查最小间隔
        if session.last_proactive_at:
            interval = now - session.last_proactive_at
            if interval < proactive_config.min_interval:
                return False

        # 概率触发
        if random.random() > proactive_config.trigger_probability:
            return False

        logger.info(
            f"主动发起条件满足: stream={session.stream_id[:8]}, "
            f"沉默 {silence_duration:.0f}s"
        )
        return True

    def _is_quiet_hours(self) -> bool:
        """检查当前是否在勿扰时段。"""
        proactive_config = self._config.proactive

        try:
            now = time.localtime()
            current_minutes = now.tm_hour * 60 + now.tm_min

            start_parts = proactive_config.quiet_hours_start.split(":")
            start_minutes = int(start_parts[0]) * 60 + int(start_parts[1])

            end_parts = proactive_config.quiet_hours_end.split(":")
            end_minutes = int(end_parts[0]) * 60 + int(end_parts[1])

            if start_minutes <= end_minutes:
                return start_minutes <= current_minutes < end_minutes
            # 跨午夜
            return current_minutes >= start_minutes or current_minutes < end_minutes

        except (ValueError, IndexError):
            return False

    async def mark_triggered(self, stream_id: str) -> None:
        """标记 Session 已触发主动发起。"""
        session = await self._session_store.get(stream_id)
        if session:
            session.last_proactive_at = time.time()
            await self._session_store.save(session)
