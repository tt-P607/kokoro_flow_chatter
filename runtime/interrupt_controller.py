"""KFC 打断控制运行时。"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.log_api import get_logger
from src.kernel.concurrency import get_task_manager

from .unread_policy import filter_interrupt_messages

if TYPE_CHECKING:
    from ..chatter import KokoroFlowChatter
    from ..config import KFCConfig


logger = get_logger("kfc_chatter")


async def send_interruptable_response(
    chatter: KokoroFlowChatter,
    response: Any,
    config: KFCConfig,
    known_unread_ids: frozenset[str],
) -> tuple[Any | None, list[Any]]:
    """以可打断方式发送 LLM 请求。

    只有真实用户消息可以打断正在生成的 LLM 响应；KFC 内部主动触发消息
    不应取消当前输出，否则定时任务与正常回复撞车时会“卡掉”模型返回。
    """

    async def _llm_work() -> Any:
        from src.kernel.concurrency import get_watchdog
        from ..protocol.response_normalizer import normalize_response

        watchdog = get_watchdog()
        watchdog.feed_dog(chatter.stream_id)
        result = await response.send(auto_append_response=True, stream=False)
        await result
        watchdog.feed_dog(chatter.stream_id)
        normalize_response(result)
        return result

    tm = get_task_manager()
    task_handle = tm.create_task(
        _llm_work(),
        name=f"kfc_llm_{chatter.stream_id[:8]}",
    )
    if task_handle.task is None:  # pragma: no cover
        raise RuntimeError("task_manager 未返回有效的 Task")
    llm_task: asyncio.Task[Any] = task_handle.task

    poll_interval = config.buffer.interrupt_poll_seconds

    try:
        while not llm_task.done():
            await asyncio.sleep(poll_interval)
            if llm_task.done():
                break

            _, current_msgs = await chatter.fetch_unreads(
                time_format="%Y-%m-%d %H:%M:%S"
            )
            interrupt_msgs = filter_interrupt_messages(
                current_msgs,
                known_unread_ids,
            )
            if interrupt_msgs:
                llm_task.cancel()
                try:
                    await llm_task
                except asyncio.CancelledError:
                    pass
                logger.info(
                    f"[打断] LLM 被取消，检测到 {len(interrupt_msgs)} 条新消息"
                )
                return None, interrupt_msgs

    except asyncio.CancelledError:
        llm_task.cancel()
        raise

    return llm_task.result(), []
