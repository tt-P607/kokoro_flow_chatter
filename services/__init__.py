"""KFC 运行时服务层。

封装带状态副作用的运行时能力，供 runtime 主循环与调度器调用：

- ``ProactiveService``：主动发起的预约管理与触发判定
- ``TimeoutService``：等待超时的判定与状态推进
- ``SummaryService``：近期记忆压缩的任务调度与去重
"""

from .proactive_service import (
    MAX_SCHEDULE_DELAY_MINUTES,
    MIN_SCHEDULE_DELAY_MINUTES,
    ProactiveService,
    clamp_schedule_delay,
)
from .summary_service import SummaryService
from .timeout_service import TimeoutResult, TimeoutService

__all__ = [
    "MAX_SCHEDULE_DELAY_MINUTES",
    "MIN_SCHEDULE_DELAY_MINUTES",
    "ProactiveService",
    "SummaryService",
    "TimeoutResult",
    "TimeoutService",
    "clamp_schedule_delay",
]
