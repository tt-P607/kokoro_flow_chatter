"""策略模块。

提供 ChatStrategy 协议定义和两种策略实现：
- UnifiedStrategy: 单次调用，LLM 同时完成规划和回复
- SplitStrategy: 拆分调用，先规划后回复
"""

from __future__ import annotations

from .base import ChatStrategy
from .unified import UnifiedStrategy
from .split import SplitStrategy

__all__ = [
    "ChatStrategy",
    "UnifiedStrategy",
    "SplitStrategy",
]
