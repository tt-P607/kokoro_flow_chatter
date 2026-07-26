"""KFC 主动发起服务。

两条触发路径：

1. **模型预约**——模型通过 ``action-schedule_proactive`` 指定时间点，
   到期必定触发，不受勿扰时段限制；
2. **沉默兜底**——无预约时，沉默超过阈值后按概率触发，受勿扰时段限制。

预约存在时沉默兜底不生效，避免两条路径互相干扰。
"""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING

from src.app.plugin_system.api.log_api import get_logger

if TYPE_CHECKING:
    from ..config import KFCConfig
    from ..domain.decision import ProactiveSchedule
    from ..session import KFCSession, KFCSessionStore

logger = get_logger("kfc_proactive")

MIN_SCHEDULE_DELAY_MINUTES = 30.0
"""预约延迟下限（分钟）。低于此值的预约会被夹到该值。"""

MAX_SCHEDULE_DELAY_MINUTES = 1440.0
"""预约延迟上限（分钟，即 24 小时）。超出会被夹到该值。"""

_TIME_SEPARATOR = ":"


def clamp_schedule_delay(delay_minutes: float) -> float:
    """把预约延迟夹到合法区间。

    Args:
        delay_minutes: 模型给出的原始延迟分钟数。

    Returns:
        float: 位于 ``[MIN_SCHEDULE_DELAY_MINUTES, MAX_SCHEDULE_DELAY_MINUTES]``
        的延迟值。
    """
    return max(
        MIN_SCHEDULE_DELAY_MINUTES,
        min(MAX_SCHEDULE_DELAY_MINUTES, delay_minutes),
    )


class ProactiveService:
    """主动发起的预约管理与触发判定。"""

    def __init__(
        self,
        config: KFCConfig,
        session_store: KFCSessionStore,
    ) -> None:
        """初始化服务。

        Args:
            config: KFC 配置。
            session_store: 会话存储，用于扫描与更新会话。
        """
        self._config = config
        self._session_store = session_store

    # ── 预约管理 ──────────────────────────────────────────

    @staticmethod
    def apply_schedule(
        session: KFCSession,
        proactive_schedule: ProactiveSchedule,
    ) -> None:
        """把模型提交的预约计划写入会话。

        ``delay_minutes`` 为 0 表示取消当前预约；其余值会被夹到合法区间。

        Args:
            session: 目标会话。
            proactive_schedule: 模型提交的预约计划。
        """
        if proactive_schedule.delay_minutes == 0:
            session.set_scheduled_proactive(None)
            logger.info("已取消主动思考预约")
            return

        delay_minutes = clamp_schedule_delay(proactive_schedule.delay_minutes)
        reason = proactive_schedule.reason
        session.set_scheduled_proactive(
            time.time() + delay_minutes * 60,
            reason=reason,
        )
        logger.info(
            f"已预约主动思考: {delay_minutes:.0f} 分钟后"
            + (f"，理由：{reason}" if reason else "")
        )

    async def mark_triggered(self, stream_id: str) -> str:
        """标记指定流已触发主动发起，并清除已消费的预约。

        Args:
            stream_id: 目标流 ID。

        Returns:
            str: 清除前的预约理由；无预约时为空串。
        """
        async with self._session_store.lock(stream_id):
            session = await self._session_store.get(stream_id)
            if session is None:
                return ""
            reason = session.scheduled_proactive_reason
            session.last_proactive_at = time.time()
            session.set_scheduled_proactive(None)
            await self._session_store.save(session)
            return reason

    # ── 触发判定 ──────────────────────────────────────────

    async def collect_triggered_streams(self) -> list[str]:
        """扫描所有会话，返回本轮应主动发起的流 ID 列表。

        内存中的会话走完整判定（预约 + 沉默兜底）；仅存于磁盘的会话
        只检查预约，避免大批量冷会话同时被沉默兜底唤醒。

        Returns:
            list[str]: 应触发主动发起的流 ID。
        """
        if not self._config.proactive.enabled:
            return []

        triggered: list[str] = []
        cached_sessions = self._session_store.get_all_cached()
        for stream_id, session in cached_sessions.items():
            if self._should_trigger_cached(stream_id, session):
                triggered.append(stream_id)

        for stream_id in await self._session_store.list_all_stream_ids():
            if stream_id in cached_sessions:
                continue
            session = await self._session_store.peek(stream_id)
            if session is None or session.scheduled_proactive_at is None:
                continue
            if time.time() >= session.scheduled_proactive_at:
                logger.info(f"主动思考（磁盘会话）：触发预约 stream={stream_id[:8]}")
                triggered.append(stream_id)

        return triggered

    def _should_trigger_cached(self, stream_id: str, session: KFCSession) -> bool:
        """判定内存中的会话本轮是否应触发。"""
        if session.scheduled_proactive_at is not None:
            if time.time() >= session.scheduled_proactive_at:
                logger.info(f"主动思考：触发模型预约 stream={stream_id[:8]}")
                return True
            # 有预约在等待时，沉默兜底不介入
            return False
        return self._should_trigger_by_silence(session)

    def _should_trigger_by_silence(self, session: KFCSession) -> bool:
        """无预约时按沉默时长与概率判定是否触发。"""
        if self._is_quiet_hours():
            return False

        proactive_config = self._config.proactive
        now = time.time()

        silence_duration = now - session.last_activity_at
        if silence_duration < proactive_config.silence_threshold:
            return False

        if session.last_proactive_at is not None:
            if now - session.last_proactive_at < proactive_config.min_interval:
                return False

        if random.random() > proactive_config.trigger_probability:
            return False

        logger.info(
            f"主动发起条件满足: stream={session.stream_id[:8]}, "
            f"沉默 {silence_duration:.0f}s"
        )
        return True

    def _is_quiet_hours(self) -> bool:
        """判断当前是否处于勿扰时段（仅约束沉默兜底路径）。"""
        proactive_config = self._config.proactive
        try:
            start_minutes = _parse_clock_minutes(proactive_config.quiet_hours_start)
            end_minutes = _parse_clock_minutes(proactive_config.quiet_hours_end)
        except (ValueError, IndexError):
            return False

        now = time.localtime()
        current_minutes = now.tm_hour * 60 + now.tm_min
        if start_minutes <= end_minutes:
            return start_minutes <= current_minutes < end_minutes
        # 跨午夜区间
        return current_minutes >= start_minutes or current_minutes < end_minutes


def _parse_clock_minutes(clock_text: str) -> int:
    """把 ``HH:MM`` 解析为当日分钟数。"""
    hour_text, minute_text = clock_text.split(_TIME_SEPARATOR)
    return int(hour_text) * 60 + int(minute_text)
