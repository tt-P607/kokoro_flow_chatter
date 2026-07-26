"""KFC 领域模型层。

存放不依赖运行时环境的纯数据结构与判定逻辑：对话链条目、决策对象、
回合触发分类。本层不做 IO，也不导入 runtime / context / execution。
"""

from .chain_entry import ChainEntry
from .decision import Decision, ProactiveSchedule, ToolCallSpec
from .turn_trigger import TurnTrigger, classify_turn_trigger

__all__ = [
    "ChainEntry",
    "Decision",
    "ProactiveSchedule",
    "ToolCallSpec",
    "TurnTrigger",
    "classify_turn_trigger",
]
