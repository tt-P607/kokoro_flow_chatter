"""在对话接近窗口上限时封存旧回合并重建活动请求。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.app.plugin_system.types import ROLE

from ..context_budget import should_rotate
from ..snapshot import deserialize_snapshot, serialize_payloads
from .context_builder import build_initial_request

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
    """先封存并提交旧完整回合，再重建只含当前回合的活动请求。"""
    if not should_rotate([*response.payloads, *transient_payloads], model_set):
        return response, None, False
    entries = serialize_payloads(response.payloads)
    user_starts = [
        index for index, entry in enumerate(entries)
        if entry["role"] == ROLE.USER.value
    ]
    if len(user_starts) < 2:
        raise ValueError("KFC 当前回合太长，缺少可封存的旧完整回合")

    boundary = user_starts[-1]
    sealed = entries[:boundary]
    active = entries[boundary:]
    for segment in (sealed, active):
        restored = deserialize_snapshot(segment)
        if restored is None or len(restored) != len(segment):
            raise ValueError("KFC 对话链存在未闭合工具段，无法无损封存")
    store = chatter.session_store
    async with store.lock(session.stream_id):
        previous = session.context_snapshot
        session.sealed_segments.append(sealed)
        session.context_snapshot = active
        try:
            await store.save_context(session)
        except BaseException:
            session.context_snapshot = previous
            session.sealed_segments.pop()
            raise

    new_request, usable_map = await build_initial_request(
        chatter, chat_stream, config, session, model_set
    )
    return new_request, usable_map, True