"""KFC 消息积累窗口运行时。"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..chatter import KokoroFlowChatter
    from ..config import KFCConfig


async def accumulate_message_buffer(
    chatter: KokoroFlowChatter,
    config: KFCConfig,
) -> tuple[str, list[Any]]:
    """在积累窗口内等待并聚合连发消息。"""
    window = config.buffer.accumulate_window
    max_window = config.buffer.accumulate_max_window

    if window <= 0:
        return await chatter.fetch_unreads(time_format="%Y-%m-%d %H:%M:%S")

    deadline = time.monotonic() + max_window
    last_count = 0

    while True:
        _, current_msgs = await chatter.fetch_unreads(time_format="%Y-%m-%d %H:%M:%S")
        current_count = len(current_msgs)

        if current_count > last_count:
            last_count = current_count
            next_check = time.monotonic() + window
        else:
            next_check = time.monotonic()

        remaining = min(deadline, next_check) - time.monotonic()
        if remaining <= 0:
            break

        await asyncio.sleep(min(0.2, remaining))

    return await chatter.fetch_unreads(time_format="%Y-%m-%d %H:%M:%S")