"""KFC 响应标准化。

不同 provider 返回 tool call 的形态不一致：部分模型会把工具调用写进
正文 JSON 而非结构化 ``call_list``。本模块在决策解析前统一把这类响应
就地修正为标准形态。

注意：``reasoning_content`` 不会被回填到 ``message``——思考内容仅供
日志与调试，不参与决策判定。
"""

from __future__ import annotations

from typing import Any

from src.kernel.llm import LLMPayload, ROLE, ReasoningText, Text, ToolCall
from src.kernel.llm.tool_call_compat import parse_tool_call_compat_response


def normalize_response(response: Any) -> bool:
    """就地标准化模型响应。

    仅在 ``call_list`` 为空时尝试从正文解析 compat 形态的工具调用；
    解析成功会同步改写 ``message`` / ``call_list`` 与最后一条
    ASSISTANT payload，使后续链路只需面对标准形态。

    Args:
        response: 模型响应对象。

    Returns:
        bool: 是否从正文补出了工具调用。
    """
    if response.call_list:
        return False

    message = response.message
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


def _sync_last_assistant_payload(response: Any) -> None:
    """把改写后的响应内容同步回最后一条 ASSISTANT payload。

    正文里的 compat JSON 已被替换成结构化调用，若不同步，上下文链中
    仍会保留原始 JSON 文本，导致模型在后续轮次看到自相矛盾的历史。
    """
    payloads = response.payloads
    if not isinstance(payloads, list):
        return

    content_parts: list[Any] = []
    reasoning_content = response.reasoning_content
    if isinstance(reasoning_content, str) and reasoning_content:
        content_parts.append(ReasoningText(reasoning_content))
    if isinstance(response.message, str) and response.message:
        content_parts.append(Text(response.message))
    if response.call_list:
        content_parts.extend(response.call_list)
    if not content_parts:
        content_parts.append(Text(""))

    if payloads and payloads[-1].role == ROLE.ASSISTANT:
        payloads[-1].content = content_parts
        return
    payloads.append(LLMPayload(ROLE.ASSISTANT, content_parts))
