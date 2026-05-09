"""KFC LLM 请求发送视图。

发送视图用于在本轮 LLM 调用中临时加入 transient payload，
但不把这些 payload 写入长期 response 链，避免直接 append/pop 修改主链。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.app.plugin_system.types import LLMPayload, ROLE
from src.kernel.llm.request import LLMRequest


@dataclass(slots=True)
class RequestView:
    """一次 LLM 调用的临时发送视图。"""

    source: Any
    payloads: list[LLMPayload] = field(default_factory=list)

    async def send(self, *, auto_append_response: bool = True, stream: bool = False) -> Any:
        """用视图 payloads 发送请求，并将非 transient 结果回写到 source。"""
        source_payloads = list(getattr(self.source, "payloads", []))
        transient_count = max(len(self.payloads) - len(source_payloads), 0)
        upper = getattr(self.source, "_upper", self.source)
        req = LLMRequest(
            self.source.model_set,
            request_name=getattr(upper, "request_name", ""),
            meta_data=dict(getattr(upper, "meta_data", {})),
            context_manager=getattr(self.source, "context_manager", None),
        )
        req.payloads = list(self.payloads)
        result = await req.send(auto_append_response=auto_append_response, stream=stream)
        if not getattr(result, "_consumed", False):
            await result

        persistent_payloads = _without_transient_payloads(
            result.payloads,
            source_payloads=source_payloads,
            transient_count=transient_count,
        )
        result.payloads = persistent_payloads

        if not hasattr(self.source, "message"):
            return result

        self.source.message = result.message
        self.source.reasoning_content = result.reasoning_content
        self.source.reasoning_parts = result.reasoning_parts
        self.source.call_list = result.call_list
        self.source.tool_call_compat = result.tool_call_compat
        self.source.payloads = persistent_payloads
        if hasattr(self.source, "_consumed"):
            self.source._consumed = getattr(result, "_consumed", True)
        if hasattr(self.source, "_appended_to_context"):
            self.source._appended_to_context = getattr(
                result,
                "_appended_to_context",
                auto_append_response,
            )
        return self.source


def _without_transient_payloads(
    payloads: list[LLMPayload],
    *,
    source_payloads: list[LLMPayload],
    transient_count: int,
) -> list[LLMPayload]:
    """移除发送视图追加的 transient payload 和其触发的 reminder 前缀。

    ``LLMContextManager`` 会在发送时把 system_reminder 前缀注入 USER payload，
    因此仅按索引切掉 extra payload 不够；还必须用发送前的 source payload
    覆盖同位置的持久 payload，避免 actor/dynamic reminder 被回写到主链。
    """
    base_count = len(source_payloads)
    if transient_count <= 0:
        persistent_payloads = list(payloads)
    else:
        persistent_payloads = (
            list(payloads[:base_count])
            + list(payloads[base_count + transient_count :])
        )

    for index, source_payload in enumerate(source_payloads):
        if index >= len(persistent_payloads):
            break
        if source_payload.role == ROLE.USER:
            persistent_payloads[index] = source_payload
    return persistent_payloads


def build_request_view(response: Any, transient_payloads: list[LLMPayload] | None = None) -> RequestView:
    """基于 response 构造只用于发送的一次性视图。"""
    payloads = list(getattr(response, "payloads", []))
    if transient_payloads:
        payloads.extend(transient_payloads)
    return RequestView(source=response, payloads=payloads)
