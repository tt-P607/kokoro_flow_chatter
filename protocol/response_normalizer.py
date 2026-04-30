"""KFC 响应标准化。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .compat_adapter import is_deepseek_model_set, try_parse_tool_call_compat_response


@dataclass(slots=True)
class NormalizedResponse:
    """标准化后的响应视图。"""

    response: Any
    text: str
    used_reasoning_content: bool = False
    used_compat_tool_calls: bool = False

    @property
    def has_tool_calls(self) -> bool:
        """是否已经形成标准工具调用。"""
        return bool(getattr(self.response, "call_list", None))


def resolve_response_text(response: Any) -> tuple[str, bool]:
    """统一提取响应正文；正文为空时回退到 reasoning_content。"""
    message = getattr(response, "message", None)
    if isinstance(message, str) and message.strip():
        return message.strip(), False

    reasoning_text = getattr(response, "reasoning_content", None)
    if isinstance(reasoning_text, str) and reasoning_text.strip():
        return reasoning_text.strip(), True

    if isinstance(message, str):
        return message.strip(), False

    return "", False


def normalize_response(response: Any) -> NormalizedResponse:
    """将 provider 原始响应标准化为 KFC 统一视图。"""
    resolved_text, used_reasoning = resolve_response_text(response)
    if used_reasoning and not (response.message or "").strip() and not getattr(response, "call_list", None):
        response.message = resolved_text

    used_compat_tool_calls = False
    if not getattr(response, "call_list", None) and is_deepseek_model_set(getattr(response, "model_set", None)):
        used_compat_tool_calls = try_parse_tool_call_compat_response(response)

    normalized_text, _ = resolve_response_text(response)
    return NormalizedResponse(
        response=response,
        text=normalized_text,
        used_reasoning_content=used_reasoning,
        used_compat_tool_calls=used_compat_tool_calls,
    )