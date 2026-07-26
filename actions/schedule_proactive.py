"""预约下一次主动思考。

模型可通过本动作指定未来某个时间点主动发起对话。预约存在时，基于
沉默时长的兜底触发暂停，直到预约到期或被取消。

实际的会话状态写入由 ``ProactiveService.apply_schedule()`` 完成——
本动作只负责暴露 schema 与回执，避免同一份 clamp 逻辑写两遍。
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Annotated, ClassVar

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.base import BaseAction

from ..services import (
    MAX_SCHEDULE_DELAY_MINUTES,
    MIN_SCHEDULE_DELAY_MINUTES,
    clamp_schedule_delay,
)

logger = get_logger("kfc_schedule_proactive")

_BASE_DESCRIPTION = (
    "预约一个时间点，届时系统会主动唤醒你去发起新一轮对话。"
    "**新的预约会覆盖旧的预约；传 delay_minutes=0 可取消当前预约。**\n"
    "**预约不受勿扰时段限制**，即使是深夜或清晨设定的预约也会如期触发。\n"
    f"delay_minutes=0 取消预约；其他值范围限制为 "
    f"{MIN_SCHEDULE_DELAY_MINUTES:g}~{MAX_SCHEDULE_DELAY_MINUTES:g} 分钟"
    "（30 分钟~24 小时）。\n\n"
    "**reason（必填）：**\n"
    "记录此刻的真实想法。可以是一件具体的事，也可以只是「那个时间想找 Ta 说说话」"
    "——怎么自然怎么写，重要的是让未来的你看到时能自然接上。取消预约时可留空。"
)


class ScheduleProactiveAction(BaseAction):
    """预约下一次主动思考时间。"""

    name: str = "schedule_proactive"
    associated_types: list[str] = ["text"]
    description: str = _BASE_DESCRIPTION
    chatter_allow: list[str] = ["kokoro_flow_chatter"]

    _guidance: ClassVar[str] = ""
    """可配置的使用场景指导语，由插件加载时从配置写入。"""

    @classmethod
    def set_guidance(cls, guidance: str) -> None:
        """设置附加到工具描述末尾的使用指导语。

        Args:
            guidance: 指导语文本；传空串表示清除。
        """
        cls._guidance = guidance

    @classmethod
    def to_schema(cls) -> dict:  # type: ignore[override]
        """生成 schema，按需拼接可配置的指导语。"""
        schema = super().to_schema()
        if cls._guidance:
            schema.get("function", {})["description"] = (
                f"{_BASE_DESCRIPTION}\n\n{cls._guidance}"
            )
        return schema

    async def execute(
        self,
        delay_minutes: Annotated[
            int,
            "多少分钟后发起主动思考。传 0 表示取消当前预约；"
            "其他值范围 30~1440（30 分钟~24 小时）。",
        ] = 30,
        reason: Annotated[
            str,
            "此刻的真实想法：可以是一件具体的事，"
            "也可以只是「那个时间想找 Ta 说说话」。取消预约时可留空。",
        ] = "",
    ) -> tuple[bool, str]:
        """返回预约动作的执行回执。

        会话状态的实际写入由主循环通过 ``ProactiveService`` 完成，
        本方法只产出模型可见的回执文本。

        Args:
            delay_minutes: 延迟分钟数；0 表示取消预约。
            reason: 预约理由。

        Returns:
            tuple: ``(是否成功, 回执描述)``。
        """
        if delay_minutes == 0:
            logger.debug("取消主动思考预约")
            return True, "已取消当前主动思考预约"

        clamped_minutes = clamp_schedule_delay(float(delay_minutes))
        scheduled_at = time.time() + clamped_minutes * 60
        scheduled_text = datetime.fromtimestamp(scheduled_at).strftime("%H:%M:%S")
        logger.debug(
            f"预约主动思考: {scheduled_text}" + (f"（{reason}）" if reason else "")
        )
        return True, f"已预约在 {clamped_minutes:.0f} 分钟后主动思考"
