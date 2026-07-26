"""KFC 初始上下文构建。

组装 ``execute()`` 启动时的第一份 LLM 请求：系统提示词、动态上下文、
历史对话链与工具注册。

所有 payload 统一经 ``add_payload()`` 注入，由 context manager 的
reminder 管线统一管理注入与剥离——若绕过管线直接构造 payload，
system_reminder 会在多轮循环中重复堆积。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.llm_api import (
    LLMContextManager,
    ReminderSourceSpec,
    create_llm_request,
)

from ..context import plan_initial_context, render_initial_context

if TYPE_CHECKING:
    from src.app.plugin_system.api.llm_api import ToolRegistry
    from src.app.plugin_system.types import ChatStream

    from ..chatter import KokoroFlowChatter
    from ..config import KFCConfig
    from ..session import KFCSession

REQUEST_NAME = "kokoro_flow_chatter"
"""LLM 请求名，用于统计与日志归类。"""

_ACTOR_BUCKET = "actor"
"""全局 actor reminder bucket。"""


def _build_reminder_sources(stream_id: str) -> list[ReminderSourceSpec]:
    """构建 reminder 订阅列表。

    同时订阅全局 actor bucket 与当前流的私有 bucket，与框架
    ``BaseChatter.create_request`` 的 ``with_reminder`` 行为保持一致。
    """
    sources = [ReminderSourceSpec(bucket=_ACTOR_BUCKET, wrap_with_system_tag=True)]
    if stream_id:
        sources.append(
            ReminderSourceSpec(
                bucket=f"stream:{stream_id}:{_ACTOR_BUCKET}",
                wrap_with_system_tag=True,
            )
        )
    return sources


async def build_initial_request(
    chatter: KokoroFlowChatter,
    chat_stream: ChatStream,
    config: KFCConfig,
    session: KFCSession,
    model_set: Any,
) -> tuple[Any, ToolRegistry]:
    """构建初始 LLM 请求并注册可用工具。

    Args:
        chatter: 当前 chatter 实例，提供工具注入能力。
        chat_stream: 当前聊天流。
        config: KFC 配置。
        session: 当前会话。
        model_set: 已解析的模型集。

    Returns:
        tuple: ``(request, usable_map)``。
    """
    request = create_llm_request(
        model_set,
        REQUEST_NAME,
        context_manager=LLMContextManager(
            reminder_sources=_build_reminder_sources(chatter.stream_id),
        ),
    )

    plan = plan_initial_context(
        chat_stream=chat_stream,
        config=config,
        session=session,
    )
    system_payloads, chain_payloads, _has_history = await render_initial_context(
        chat_stream=chat_stream,
        plan=plan,
        mental_log=session.mental_log,
        serialized_chain_payloads=list(session.chain_payloads),
    )
    for payload in (*system_payloads, *chain_payloads):
        request.add_payload(payload)

    usable_map = await chatter.inject_usables(request)
    return request, usable_map
