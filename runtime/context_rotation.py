"""在对话接近窗口上限时封存旧回合并重建活动请求。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.types import ROLE

from ..context_budget import context_budget, payload_tokens, should_rotate
from ..snapshot import deserialize_snapshot, serialize_payloads
from .context_builder import build_initial_request
from .request_view import build_request_view

if TYPE_CHECKING:
    from src.app.plugin_system.types import ChatStream, LLMPayload

    from ..chatter import KokoroFlowChatter
    from ..config import KFCConfig
    from ..session import KFCSession


async def rotate_if_needed(
    chatter: KokoroFlowChatter,
    response: Any,
    chat_stream: ChatStream,
    config: KFCConfig,
    session: KFCSession,
    model_set: Any,
    transient_payloads: list[LLMPayload],
) -> tuple[Any, Any, bool]:
    """重建候选请求并验证预算后，原子交接已完成前缀与活动链。"""
    full_payloads = build_request_view(response, transient_payloads).payloads
    if not should_rotate(full_payloads, model_set):
        return response, None, False
    entries = serialize_payloads(response.payloads)
    boundary = session.completed_context_length
    if not boundary:
        _check_request_budget(response, transient_payloads, model_set)
        return response, None, False
    if boundary > len(entries):
        raise ValueError("KFC 已完成回合边界超出活动链")

    sealed = entries[:boundary]
    active = entries[boundary:]
    for segment in (sealed, active):
        if not segment:
            continue
        restored = deserialize_snapshot(segment)
        if restored is None or len(restored) != len(segment):
            raise ValueError("KFC 对话链存在未闭合工具段，无法无损封存")
    store = chatter.session_store
    async with store.lock(session.stream_id):
        candidate = replace(
            session, context_snapshot=active or None,
            completed_context_length=0,
            sealed_segments=[*session.sealed_segments, sealed],
        )
        new_request, usable_map = await build_initial_request(
            chatter, chat_stream, config, candidate, model_set
        )
        _check_request_budget(new_request, transient_payloads, model_set)
        previous = session.context_snapshot
        previous_completed_length = session.completed_context_length
        previous_sealed = session.sealed_segments
        session.context_snapshot = candidate.context_snapshot
        session.completed_context_length = 0
        session.sealed_segments = candidate.sealed_segments
        try:
            await store.save_context(session)
        except BaseException as error:
            session.context_snapshot = previous
            session.completed_context_length = previous_completed_length
            session.sealed_segments = previous_sealed
            if isinstance(error, asyncio.CancelledError):
                await store.save_context(session)
            raise

    return new_request, usable_map, True


def _check_request_budget(
    response: Any, transient_payloads: list[LLMPayload], model_set: Any,
) -> None:
    """检查完整发送视图；真正超限时报告各类内容的估算占用。"""
    full_payloads = build_request_view(response, transient_payloads).payloads
    tokens = payload_tokens(full_payloads, model_set)
    budget = context_budget(model_set)
    if tokens <= budget:
        return
    pinned = [
        payload for payload in response.payloads
        if payload.role in {ROLE.SYSTEM, ROLE.TOOL}
    ]
    dialogue = [
        payload for payload in response.payloads
        if payload.role not in {ROLE.SYSTEM, ROLE.TOOL}
    ]
    background = build_request_view(response).transient_payloads
    raise ValueError(
        f"KFC 请求超出模型预算：{tokens} > {budget}"
        f"（系统与工具={payload_tokens(pinned, model_set)}，"
        f"动态背景={payload_tokens(background, model_set)}，"
        f"当前对话={payload_tokens(dialogue, model_set)}，"
        f"临时注入={payload_tokens(transient_payloads, model_set)}）"
    )