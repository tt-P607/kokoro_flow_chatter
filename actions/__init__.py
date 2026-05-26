"""KFC 动作组件模块。

提供核心动作：
- KFCReplyAction: 发送消息
- DoNothingAction: 选择不回复
- PassAndWaitAction: 完成当前动作后等待
"""

from __future__ import annotations

from .do_nothing import DoNothingAction
from .pass_and_wait import PassAndWaitAction
from .reply import KFCReplyAction

__all__ = [
    "DoNothingAction",
    "KFCReplyAction",
    "PassAndWaitAction",
]
