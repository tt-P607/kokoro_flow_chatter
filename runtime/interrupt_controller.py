"""KFC 可打断的 LLM 调用。

生成期间若对方发来新消息，继续用过时的上下文把话说完会显得答非所问。
本模块在后台轮询未读队列，一旦发现真实新消息就取消当前请求，交由主
循环合并消息后重新发起。

只有真实用户消息才能打断；KFC 自身的主动发起触发消息不应取消正在
进行的输出，否则定时任务与正常回复撞车时会"吃掉"模型响应。
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.log_api import get_logger
from src.kernel.concurrency import get_task_manager, get_watchdog

from ..protocol.response_normalizer import normalize_response
from .unread_policy import filter_interrupt_messages

if TYPE_CHECKING:
    from ..chatter import KokoroFlowChatter
    from ..config import KFCConfig

logger = get_logger("kfc_interrupt")

_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


async def send_interruptable_response(
    chatter: KokoroFlowChatter,
    send_target: Any,
    config: KFCConfig,
    known_unread_ids: frozenset[str],
) -> tuple[Any | None, list[Any]]:
    """以可打断方式发送 LLM 请求。

    Args:
        chatter: 当前 chatter 实例。
        send_target: 发送目标，通常是 ``RequestView``。
        config: KFC 配置，提供轮询间隔。
        known_unread_ids: 已纳入本轮上下文的消息 ID，不触发打断。

    Returns:
        tuple: 正常完成时为 ``(响应, [])``；被打断时为 ``(None, 打断消息列表)``。
    """

    async def llm_work() -> Any:
        """执行实际的 LLM 调用，并在首尾喂看门狗。"""
        watchdog = get_watchdog()
        watchdog.feed_dog(chatter.stream_id)
        result = await send_target.send(auto_append_response=True, stream=False)
        watchdog.feed_dog(chatter.stream_id)
        normalize_response(result)
        return result

    task_handle = get_task_manager().create_task(
        llm_work(),
        name=f"kfc_llm_{chatter.stream_id[:8]}",
    )
    if task_handle.task is None:  # pragma: no cover - 任务管理器契约保证非空
        raise RuntimeError("task_manager 未返回有效的 Task")

    llm_task: asyncio.Task[Any] = task_handle.task
    poll_interval = config.buffer.interrupt_poll_seconds

    try:
        while not llm_task.done():
            await asyncio.sleep(poll_interval)
            if llm_task.done():
                break

            _, current_msgs = await chatter.fetch_unreads(time_format=_TIME_FORMAT)
            interrupt_msgs = filter_interrupt_messages(current_msgs, known_unread_ids)
            if not interrupt_msgs:
                continue

            llm_task.cancel()
            try:
                await llm_task
            except asyncio.CancelledError:
                pass
            logger.info(f"LLM 被取消，检测到 {len(interrupt_msgs)} 条新消息")
            return None, interrupt_msgs
    except asyncio.CancelledError:
        llm_task.cancel()
        raise

    return llm_task.result(), []
