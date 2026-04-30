"""KFC 协议层导出。"""

from .compat_adapter import (
    build_tool_call_compat_retry_prompt,
    is_deepseek_model_set,
    prepare_kfc_model_set,
    rewrite_response_as_unsent_draft,
    try_parse_tool_call_compat_response,
)
from .decision_parser import build_decision, parse_response_decision
from .response_normalizer import NormalizedResponse, normalize_response, resolve_response_text

__all__ = [
    "NormalizedResponse",
    "build_decision",
    "build_tool_call_compat_retry_prompt",
    "is_deepseek_model_set",
    "normalize_response",
    "parse_response_decision",
    "prepare_kfc_model_set",
    "resolve_response_text",
    "rewrite_response_as_unsent_draft",
    "try_parse_tool_call_compat_response",
]