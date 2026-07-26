"""KFC 专属动作。

``kfc_reply`` / ``do_nothing`` / ``pass_and_wait`` 是控制动作——它们由
执行层直接解释，``execute()`` 仅作为 schema 的形式入口存在；
``kfc_memo`` / ``kfc_memo_delete`` / ``schedule_proactive`` 则是常规动作，
执行体内含真实副作用。
"""

from .do_nothing import DoNothingAction
from .memo import KFCMemoAction, KFCMemoDeleteAction
from .pass_and_wait import PassAndWaitAction
from .reply import KFCReplyAction
from .schedule_proactive import ScheduleProactiveAction

__all__ = [
    "DoNothingAction",
    "KFCMemoAction",
    "KFCMemoDeleteAction",
    "KFCReplyAction",
    "PassAndWaitAction",
    "ScheduleProactiveAction",
]
