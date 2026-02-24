"""KFC 动作组件模块。

提供核心动作：
- KFCReplyAction: 发送消息
- DoNothingAction: 选择不回复
"""

from __future__ import annotations

from .do_nothing import DoNothingAction
from .reply import KFCReplyAction

__all__ = [
    "DoNothingAction",
    "KFCReplyAction",
]
