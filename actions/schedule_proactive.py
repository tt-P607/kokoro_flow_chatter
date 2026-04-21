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
        "预约一个时间点，届时系统会主动唤醒你去发起新一轮对话。"
        "**新的预约会覆盖旧的预约。**\n"
        "**不要只在话题结束时才想到这个工具**——下面这些日常场景都值得预约：\n"
        "- 对方说去忙了、睡了、去做某件事了，你想等他/她回来后主动问候\n"
        "- 聊了一段有趣的话题，你想晚些继续，或者想到了什么要告诉对方\n"
        "- 约好了某件事（一起玩、明天见等），你想到时候提醒\n"
        "- 对方今天心情不好，你想过段时间去关心一下\n"
        "- 没什么特别的事，就是过一段时间想找对方说说话\n"
        "只要你有「之后某个时间想主动联系」的念头，就调用它。\n"
        "预约后，系统的沉默触发逻辑会暂停，直到你指定的时间到达。\n"
        "delay_minutes 范围限制为 30~720 分钟（30 分钟~12 小时）。"
    )
    chatter_allow: list[str] = ["kokoro_flow_chatter"]

    _MIN_DELAY_MIN = 30    # 最小 30 分钟
    _MAX_DELAY_MIN = 720   # 最大 12 小时

    async def execute(
        self,
        delay_minutes: Annotated[
            int,
            "多少分钟后发起主动思考，范围 30~720（30 分钟~12 小时）",
        ] = 30,
        reason: Annotated[
            str,
            "预约原因（必填）：用一两句话说明届时你想主动聊什么、或者为什么想在这个时候联系对方。"
            "这段理由会在预约时间到达时注入到你的上下文中，帮助你记住当时的意图。",
        ] = "",
    ) -> tuple[bool, str]:
        """设置主动思考预约时间。

        Returns:
            (True, 状态描述)
        """
        delay_minutes = max(self._MIN_DELAY_MIN, min(self._MAX_DELAY_MIN, delay_minutes))
        delay_seconds = delay_minutes * 60
        at = time.time() + delay_seconds
        from datetime import datetime

        dt_str = datetime.fromtimestamp(at).strftime("%H:%M:%S")
        if reason:
            logger.debug(f"预约主动思考: {dt_str}（{reason}）")
        else:
            logger.debug(f"预约主动思考: {dt_str}")
        return True, f"已预约在 {delay_minutes} 分钟后主动思考"
