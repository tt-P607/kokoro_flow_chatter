"""提示词模块。

提供 KFC 专用的提示词模板、模块函数和构建器。
"""

from __future__ import annotations

from .templates import (
    KFC_SYSTEM_PROMPT,
    KFC_PROACTIVE_PROMPT,
    KFC_TIMEOUT_PROMPT,
)

__all__ = [
    "KFC_SYSTEM_PROMPT",
    "KFC_PROACTIVE_PROMPT",
    "KFC_TIMEOUT_PROMPT",
    "KFCPromptBuilder",
]


def __getattr__(name: str) -> object:
    """惰性导出 builder，避免 context/planner 导入 templates 时触发循环导入。"""
    if name == "KFCPromptBuilder":
        from .builder import KFCPromptBuilder

        return KFCPromptBuilder
    raise AttributeError(name)
