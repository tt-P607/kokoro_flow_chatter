"""KFC 未读消息策略。

集中处理未读队列中不同来源消息的优先级，避免主动发起判断散落在
回合控制与打断控制里。

核心规则：KFC 自身注入的主动发起触发消息是"内部信号"，一旦与真实
用户消息同时出现，就应让位给真实消息；它也永远不应打断正在进行的
LLM 生成。
"""

from __future__ import annotations

from typing import Any, Protocol

from src.app.plugin_system.api.log_api import get_logger

logger = get_logger("kfc_unread_policy")

PROACTIVE_MESSAGE_PREFIX = "proactive_"
"""主动发起触发消息的 ID 前缀，由 ``ProactiveHandler`` 生成时写入。"""

_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


class ChatterUnreadIO(Protocol):
    """未读策略所需的最小 Chatter 接口。"""

    async def flush_unreads(self, unread_messages: list[Any]) -> int:
        """把指定未读消息移出未读队列。"""
        ...

    @staticmethod
    def format_message_line(msg: Any, time_format: str = _TIME_FORMAT) -> str:
        """渲染单条消息为展示行。"""
        ...


def is_proactive_trigger_message(message: Any) -> bool:
    """判断消息是否为 KFC 主动发起注入的内部触发消息。"""
    message_id = message.message_id
    return isinstance(message_id, str) and message_id.startswith(
        PROACTIVE_MESSAGE_PREFIX
    )


async def prefer_real_unreads(
    chatter: ChatterUnreadIO,
    unread_msgs: list[Any],
) -> list[Any]:
    """真实消息与主动触发撞车时只保留真实消息。

    被丢弃的主动触发会被 flush，避免后续回合重复处理同一条内部信号。

    Args:
        chatter: 提供 flush 能力的 chatter。
        unread_msgs: 当前未读快照。

    Returns:
        list[Any]: 过滤后的未读消息列表。
    """
    real_messages: list[Any] = []
    proactive_messages: list[Any] = []
    for message in unread_msgs:
        if is_proactive_trigger_message(message):
            proactive_messages.append(message)
        else:
            real_messages.append(message)

    if not real_messages or not proactive_messages:
        return unread_msgs

    await chatter.flush_unreads(proactive_messages)
    logger.info(
        f"主动触发与真实消息撞车，已丢弃 {len(proactive_messages)} 条主动触发消息"
    )
    return real_messages


def format_unread_messages(
    chatter: ChatterUnreadIO,
    unread_msgs: list[Any],
    time_format: str = _TIME_FORMAT,
) -> str:
    """按 KFC 格式渲染未读消息快照。"""
    return "\n".join(
        chatter.format_message_line(message, time_format) for message in unread_msgs
    )


def filter_interrupt_messages(
    current_msgs: list[Any],
    known_unread_ids: frozenset[str],
) -> list[Any]:
    """筛选真正应该打断 LLM 生成的新消息。

    已纳入本轮上下文的消息与 KFC 内部主动触发都不构成打断理由。

    Args:
        current_msgs: 当前未读快照。
        known_unread_ids: 已知消息 ID 集合。

    Returns:
        list[Any]: 应触发打断的消息列表。
    """
    return [
        message
        for message in current_msgs
        if message.message_id not in known_unread_ids
        and not is_proactive_trigger_message(message)
    ]
