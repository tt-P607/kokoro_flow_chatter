"""提示词模块。

提供 KFC 专用的提示词模板、模块函数和构建器。
"""

from __future__ import annotations

from .templates import (
    KFC_SYSTEM_PROMPT,
    KFC_PROACTIVE_PROMPT,
    KFC_CONTINUOUS_THINKING_PROMPT,
    KFC_TIMEOUT_PROMPT,
)
from .builder import KFCPromptBuilder

__all__ = [
    "KFC_SYSTEM_PROMPT",
    "KFC_PROACTIVE_PROMPT",
    "KFC_CONTINUOUS_THINKING_PROMPT",
    "KFC_TIMEOUT_PROMPT",
    "KFCPromptBuilder",
]
