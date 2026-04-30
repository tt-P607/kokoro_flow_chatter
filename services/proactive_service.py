"""KFC 主动思考服务。"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.app.plugin_system.api.log_api import get_logger

if TYPE_CHECKING:
    from ..domain.decision import ProactiveSchedule
    from ..session import KFCSession


logger = get_logger("kfc_proactive_service")


class ProactiveService:
    """处理主动思考预约的 session 副作用。"""

    @staticmethod
    def apply_schedule(
        session: KFCSession,
        proactive_schedule: ProactiveSchedule,
    ) -> None:
        """根据决策结果更新主动思考预约。"""
        delay_minutes = float(proactive_schedule.delay_minutes)
        if delay_minutes == 0:
            session.scheduled_proactive_at = None
            session.scheduled_proactive_reason = ""
            logger.info("[KFC] 已取消主动思考预约")
            return

        delay_minutes = max(30.0, min(1440.0, delay_minutes))
        delay_seconds = delay_minutes * 60
        reason = proactive_schedule.reason
        session.set_scheduled_proactive(
            time.time() + delay_seconds,
            reason=reason,
        )
        logger.info(
            f"[KFC] 已预约主动思考: {delay_minutes:.0f} 分钟后"
            + (f"，理由：{reason}" if reason else "")
        )