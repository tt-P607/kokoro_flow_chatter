"""KFC 一次性发送视图。

主循环需要在某轮请求中临时加入 payload（如第三方注入的附加上下文），
但这些内容不应污染长期维持的 ``response`` 链。``RequestView`` 以视图
方式承载「主链 + 临时 payload」，发送后只把持久部分回写主链。

之所以不能简单地 append/pop：框架的 context manager 会在发送时向 USER
payload 注入 system_reminder 前缀，按索引切掉临时项并不能还原被修改的
持久 payload，必须用发送前的快照覆盖回去。
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
    """被视图包装的原始 response / request 对象。"""

    payloads: list[LLMPayload] = field(default_factory=list)
    """本次实际发送的完整 payload 列表（主链 + 临时项）。"""

    async def send(
        self,
        *,
        auto_append_response: bool = True,
        stream: bool = False,
    ) -> Any:
        """发送请求并把持久结果回写 source。

        Args:
            auto_append_response: 是否自动把模型输出追加进上下文。
            stream: 是否使用流式响应。

        Returns:
            Any: source 支持回写时返回 source 本身，否则返回新结果对象。
        """
        source_payloads = list(self.source.payloads)
        transient_count = max(len(self.payloads) - len(source_payloads), 0)

        # LLMResponse 把请求元信息挂在 _upper 上；source 若本身就是
        # LLMRequest，则元信息直接位于自身。
        upper = self.source._upper if hasattr(self.source, "_upper") else self.source
        request = LLMRequest(
            self.source.model_set,
            request_name=upper.request_name,
            meta_data=dict(upper.meta_data),
            context_manager=self.source.context_manager,
        )
        request.payloads = list(self.payloads)

        result = await request.send(
            auto_append_response=auto_append_response, stream=stream
        )
        if not result._consumed:
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
        self.source._consumed = result._consumed
        self.source._appended_to_context = result._appended_to_context
        return self.source


def build_request_view(
    response: Any,
    transient_payloads: list[LLMPayload] | None = None,
) -> RequestView:
    """基于 response 构造一次性发送视图。

    Args:
        response: 主链对象。
        transient_payloads: 仅本次发送生效的临时 payload。

    Returns:
        RequestView: 发送视图。
    """
    payloads = list(response.payloads)
    if transient_payloads:
        payloads.extend(transient_payloads)
    return RequestView(source=response, payloads=payloads)


def _without_transient_payloads(
    payloads: list[LLMPayload],
    *,
    source_payloads: list[LLMPayload],
    transient_count: int,
) -> list[LLMPayload]:
    """剔除临时 payload，并还原被 reminder 修改过的持久 USER payload。

    Args:
        payloads: 发送后的完整 payload 列表。
        source_payloads: 发送前的主链快照。
        transient_count: 本次追加的临时 payload 数量。

    Returns:
        list[LLMPayload]: 应回写主链的持久 payload 列表。
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
