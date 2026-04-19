"""ScheduleProactive 动作。

允许 LLM 预约下一次主动思考的时间。
预约存在时，条件主动发起逻辑暂停，直到预约时间到达。
"""

from __future__ import annotations

import time
from typing import Annotated

from src.app.plugin_system.api.log_api import get_logger
from src.core.components.base.action import BaseAction

logger = get_logger("kfc_schedule_proactive")


class ScheduleProactiveAction(BaseAction):
    """预约下一次主动思考时间。"""

    action_name: str = "schedule_proactive"
    action_description: str = (
        "在当前对话轮次结束后，预约一个时间点主动发起下一次对话。"
        "当你判断当前话题暂告一段落、短期内无需等待对方回复、"
        "但希望在某个时间点主动发起时，应调用此工具。"
        "预约后，沉默自动触发逻辑将暂停，直到你指定的时间到达。"
        "delay_seconds 应根据对话语境合理估计，范围限制为 1800~43200 秒。"
    )
    chatter_allow: list[str] = ["kokoro_flow_chatter"]

    _MIN_DELAY = 1800    # 最小 30 分钟
    _MAX_DELAY = 43200   # 最大 12 小时

    async def execute(
        self,
        delay_seconds: Annotated[
            int,
            "多少秒后发起主动思考，范围 1800~43200（30 分钟~12 小时）",
        ] = 1800,
        reason: Annotated[
            str,
            "预约原因，简要说明为什么想在这个时间主动发起对话",
        ] = "",
    ) -> tuple[bool, str]:
        """设置主动思考预约时间。

        Returns:
            (True, 状态描述)
        """
        delay_seconds = max(self._MIN_DELAY, min(self._MAX_DELAY, delay_seconds))
        at = time.time() + delay_seconds
        from datetime import datetime

        dt_str = datetime.fromtimestamp(at).strftime("%H:%M:%S")
        if reason:
            logger.debug(f"预约主动思考: {dt_str}（{reason}）")
        else:
            logger.debug(f"预约主动思考: {dt_str}")
        return True, f"已预约在 {delay_seconds} 秒后主动思考"
