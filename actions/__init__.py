"""KFC 动作组件模块。

提供核心动作：
- KFCReplyAction: 发送消息
- DoNothingAction: 选择不回复
- PassAndWaitAction: 完成当前动作后等待
- KFCMemoAction: 写入或刷新一条带过期时间的私人备忘录
- KFCMemoDeleteAction: 按 id 主动删除已不再需要的备忘录
"""

from __future__ import annotations

from .do_nothing import DoNothingAction
from .memo import KFCMemoAction, KFCMemoDeleteAction
from .pass_and_wait import PassAndWaitAction
from .reply import KFCReplyAction

__all__ = [
    "DoNothingAction",
    "KFCMemoAction",
    "KFCMemoDeleteAction",
    "KFCReplyAction",
    "PassAndWaitAction",
]
