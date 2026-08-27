"""KFC 领域模型层。

存放不依赖运行时环境的纯数据结构与判定逻辑：决策对象与回合触发分类。
本层不做 IO，也不导入 runtime / context / execution。
"""

from .decision import Decision, ProactiveSchedule, ToolCallSpec
from .turn_trigger import TurnTrigger, classify_turn_trigger

__all__ = [
    "Decision",
    "ProactiveSchedule",
    "ToolCallSpec",
    "TurnTrigger",
    "classify_turn_trigger",
]
