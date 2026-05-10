"""KFC 未读消息过滤策略。

本模块集中处理 unread 队列中不同来源消息的优先级，避免在回合控制、
打断控制等运行时模块里散落主动触发判断逻辑。
"""

from __future__ import annotations

from typing import Any, Protocol

from src.app.plugin_system.api.log_api import get_logger

logger = get_logger("kfc_unread_policy")


class ChatterUnreadIO(Protocol):
    """未读消息策略所需的最小 Chatter IO 接口。"""

    async def flush_unreads(self, unread_messages: list[Any]) -> int:
        """将指定未读消息从 unread 队列移入 history。"""
        ...

    @staticmethod
    def format_message_line(msg: Any, time_format: str = "%Y-%m-%d %H:%M:%S") -> str:
        """按指定时间格式渲染单条消息。"""
        ...


def is_proactive_trigger_message(message: Any) -> bool:
    """判断消息是否为 KFC 主动发起注入的内部触发消息。"""
    message_id = str(getattr(message, "message_id", "") or "")
    return message_id.startswith("proactive_")


def split_proactive_triggers(messages: list[Any]) -> tuple[list[Any], list[Any]]:
    """按真实消息与主动触发消息拆分 unread 快照。

    Returns:
        tuple[list[Any], list[Any]]: ``(real_messages, proactive_messages)``。
    """
    real_messages: list[Any] = []
    proactive_messages: list[Any] = []
    for message in messages:
        if is_proactive_trigger_message(message):
            proactive_messages.append(message)
        else:
            real_messages.append(message)
    return real_messages, proactive_messages


async def prefer_real_unreads(
    chatter: ChatterUnreadIO,
    unread_msgs: list[Any],
) -> list[Any]:
    """真实消息与主动触发撞车时，只保留真实消息。

    主动发起是内部系统触发；当它和真实用户消息同时出现时，真实消息优先。
    被丢弃的主动触发会被 flush 掉，避免后续回合再次处理同一条内部触发。
    """
    real_msgs, proactive_msgs = split_proactive_triggers(unread_msgs)
    if not real_msgs or not proactive_msgs:
        return unread_msgs

    await chatter.flush_unreads(proactive_msgs)
    logger.info(
        f"[KFC] 主动触发与真实消息撞车，已丢弃 {len(proactive_msgs)} 条主动触发消息"
    )
    return real_msgs


def format_unread_messages(
    chatter: ChatterUnreadIO,
    unread_msgs: list[Any],
    time_format: str = "%Y-%m-%d %H:%M:%S",
) -> str:
    """按 KFC 指定时间格式渲染 unread 快照。"""
    return "\n".join(
        chatter.format_message_line(message, time_format)
        for message in unread_msgs
    )


def filter_interrupt_messages(
    current_msgs: list[Any],
    known_unread_ids: frozenset[str],
) -> list[Any]:
    """筛选真正应该打断 LLM 生成的新消息。

    已知消息和 KFC 主动发起内部触发都不应取消当前 LLM 输出。
    """
    return [
        message
        for message in current_msgs
        if getattr(message, "message_id", None) not in known_unread_ids
        and not is_proactive_trigger_message(message)
    ]
