"""KFC 兼容层导出。

该模块保留旧导入路径，具体实现已迁移到 protocol 子目录。
"""

from __future__ import annotations

from .protocol.compat_adapter import (
    is_deepseek_model_set,
    prepare_kfc_model_set,
    try_parse_tool_call_compat_response,
)
from .protocol.response_normalizer import resolve_response_text

__all__ = [
    "is_deepseek_model_set",
    "prepare_kfc_model_set",
    "resolve_response_text",
    "try_parse_tool_call_compat_response",
]