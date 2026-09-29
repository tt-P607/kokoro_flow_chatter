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

from src.app.plugin_system.types import LLMPayload, ROLE, Text
from src.kernel.llm.context_structure import validate_payload_sequence
from src.kernel.llm.exceptions import LLMContextError
from src.kernel.llm.request import LLMRequest

CONTINUATION_BOUNDARY_TEXT = "（会话继续）"


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
        transient_payloads = self.payloads[len(source_payloads) :]

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
            transient_payloads=transient_payloads,
        )
        validate_payload_sequence(persistent_payloads, allow_incomplete_tail=True)
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
    transient_payloads: list[LLMPayload],
) -> list[LLMPayload]:
    """剔除临时 payload，并保留由它触发的回复所需的回合边界。

    Args:
        payloads: 发送后的完整 payload 列表。
        source_payloads: 发送前的主链快照。
        transient_payloads: 本次追加的临时 payload；用于在发送前裁剪后仍
            能按身份区分临时输入与新输出。

    Returns:
        list[LLMPayload]: 应回写主链的持久 payload 列表。
    """
    source_by_identity = {id(payload): payload for payload in source_payloads}
    transient_identities = {id(payload) for payload in transient_payloads}
    persistent_payloads: list[LLMPayload] = []
    has_real_user = False
    removed_transient_user = False

    for payload in payloads:
        if id(payload) in transient_identities:
            removed_transient_user = removed_transient_user or payload.role == ROLE.USER
            continue

        source_payload = source_by_identity.get(id(payload))
        if source_payload is not None:
            persistent_payloads.append(source_payload if source_payload.role == ROLE.USER else payload)
        else:
            if (
                payload.role == ROLE.ASSISTANT
                and removed_transient_user
                and has_real_user
                and persistent_payloads
                and persistent_payloads[-1].role == ROLE.ASSISTANT
            ):
                persistent_payloads.append(
                    LLMPayload(ROLE.USER, Text(CONTINUATION_BOUNDARY_TEXT))
                )
            persistent_payloads.append(payload)
        has_real_user = has_real_user or payload.role == ROLE.USER
        removed_transient_user = False

    first_user_index = next(
        (
            index
            for index, payload in enumerate(persistent_payloads)
            if payload.role == ROLE.USER
        ),
        None,
    )
    if first_user_index is None and any(
        payload.role == ROLE.ASSISTANT for payload in persistent_payloads
    ):
        raise LLMContextError(
            "RequestView 裁剪后缺少真实 USER，不能回写包含 assistant 的响应"
        )

    return persistent_payloads
