"""KFC provider 兼容适配。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.kernel.llm import LLMPayload, ROLE, ReasoningText, Text, ToolCall
from src.kernel.llm.tool_call_compat import parse_tool_call_compat_response


def _is_deepseek_model_entry(model_entry: Any) -> bool:
    """判断模型条目是否指向 DeepSeek 提供商。"""
    if not isinstance(model_entry, dict):
        return False

    provider = str(model_entry.get("api_provider") or "").lower()
    base_url = str(model_entry.get("base_url") or "").lower()
    model_identifier = str(model_entry.get("model_identifier") or "").lower()
    return (
        "deepseek" in provider
        or "deepseek" in base_url
        or model_identifier.startswith("deepseek-")
    )


def is_deepseek_model_set(model_set: Any) -> bool:
    """判断模型集里是否包含 DeepSeek 条目。"""
    if isinstance(model_set, list):
        return any(_is_deepseek_model_entry(entry) for entry in model_set)
    return _is_deepseek_model_entry(model_set)


def prepare_kfc_model_set(model_set: Any) -> Any:
    """为 KFC 请求准备模型集，并对特定 provider 做请求级兼容。"""
    if not isinstance(model_set, list):
        return model_set

    prepared_model_set = deepcopy(model_set)
    for model_entry in prepared_model_set:
        if not _is_deepseek_model_entry(model_entry):
            continue

        extra_params = model_entry.get("extra_params")
        if not isinstance(extra_params, dict):
            extra_params = {}
        else:
            extra_params = dict(extra_params)

        extra_params["enable_thinking"] = False
        extra_params["thinking"] = {"type": "disabled"}
        model_entry["extra_params"] = extra_params

    return prepared_model_set


def try_parse_tool_call_compat_response(response: Any) -> bool:
    """尝试把正文中的 compat JSON 转成标准 call_list。"""
    if getattr(response, "call_list", None):
        return False

    message = getattr(response, "message", None)
    if not isinstance(message, str) or not message.strip():
        return False

    try:
        parsed_message, parsed_calls = parse_tool_call_compat_response(message)
    except Exception:
        return False

    if not parsed_calls:
        return False

    response.message = parsed_message
    response.call_list = [
        ToolCall(
            id=call.get("id"),
            name=call.get("name", ""),
            args=call.get("args", {}),
        )
        for call in parsed_calls
    ]
    _sync_last_assistant_payload(response)
    return True


def _sync_last_assistant_payload(response: Any) -> bool:
    """将最后一条 assistant payload 同步为当前 response 内容。"""
    payloads = getattr(response, "payloads", None)
    if not isinstance(payloads, list):
        return False

    content_parts: list[Any] = []
    reasoning_content = getattr(response, "reasoning_content", None)
    if isinstance(reasoning_content, str) and reasoning_content:
        content_parts.append(ReasoningText(reasoning_content))

    message = getattr(response, "message", None)
    if isinstance(message, str) and message:
        content_parts.append(Text(message))

    call_list = getattr(response, "call_list", None)
    if isinstance(call_list, list) and call_list:
        content_parts.extend(call_list)

    if not content_parts:
        content_parts.append(Text(""))

    if payloads and getattr(payloads[-1], "role", None) == ROLE.ASSISTANT:
        payloads[-1].content = content_parts
        return True

    payloads.append(LLMPayload(ROLE.ASSISTANT, content_parts))
    return True

