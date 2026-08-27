"""KFC 上下文规划。

规划阶段只决定"本轮上下文由哪些内容组成"，产出纯数据的
``ContextPlan`` / ``InitialContextPlan``；具体的 payload 组装由
``context.renderer`` 完成。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.app.plugin_system.types import LLMPayload, ROLE, Text

from .sources.initial_source import build_initial_context_plan
from .sources.memo_source import build_memo_contribution
from .sources.plugin_source import collect_plugin_turn_contributions
from .types import ContextPlan, InitialContextPlan

if TYPE_CHECKING:
    from src.app.plugin_system.types import ChatStream

    from ..config import KFCConfig
    from ..session import KFCSession

USER_PROMPT_NAME = "kfc_user_prompt"
"""``on_prompt_build`` 事件中标识 KFC 用户提示词的模板名。

当前请求级行为强调不在这里持久化为伪 USER；稳定协议规则应留在系统模板，
临时提醒由 RequestView 动态注入。
"""


def plan_initial_context(
    *,
    chat_stream: ChatStream,
    config: KFCConfig,
    session: KFCSession,
) -> InitialContextPlan:
    """规划 ``execute()`` 启动时所需的初始上下文。

    Args:
        chat_stream: 当前聊天流。
        config: KFC 配置。
        session: 当前会话。

    Returns:
        InitialContextPlan: 系统模板变量、摘要与叙事截断点。
    """
    return build_initial_context_plan(
        chat_stream=chat_stream,
        config=config,
        session=session,
    )


async def plan_user_turn(
    *,
    formatted_unreads: str,
    stream_id: str = "",
    session: KFCSession | None = None,
) -> ContextPlan:
    """规划本轮用户输入及 turn 级上下文贡献。

    Args:
        formatted_unreads: 已格式化的未读消息文本。
        stream_id: 当前聊天流 ID，供 ``on_prompt_build`` 订阅者读取。
        session: 当前会话；为 ``None`` 时跳过备忘录注入。

    Returns:
        ContextPlan: 当前真实用户输入与上下文贡献列表。
    """
    user_text = f"[新消息]\n{formatted_unreads}"

    contributions = await collect_plugin_turn_contributions(
        prompt_name=USER_PROMPT_NAME,
        content=user_text,
        stream_id=stream_id,
    )

    if session is not None and session.memos:
        memo_contribution = build_memo_contribution(session.memos)
        if memo_contribution is not None:
            contributions.append(memo_contribution)

    return ContextPlan(user_text=user_text, contributions=contributions)


async def plan_followup_contributions(stream_id: str) -> ContextPlan:
    """规划续轮/超时路径的 turn 级上下文贡献。

    这两条路径不新增用户消息，但仍需收集第三方注入——否则
    ``prompt_injector`` 等插件提供的内容会在续轮中丢失。

    Args:
        stream_id: 当前聊天流 ID。

    Returns:
        ContextPlan: 仅含贡献列表，文本字段为空。
    """
    contributions = await collect_plugin_turn_contributions(
        prompt_name=USER_PROMPT_NAME,
        content="",
        stream_id=stream_id,
    )
    return ContextPlan(user_text="", contributions=contributions)


def build_last_mile_payload() -> LLMPayload:
    """构造仅当前请求可见的收尾行为指令。"""
    last_mile_instructions = (
        "请基于上述信息决定接下来你要调用的工具或动作。\n"
        "重申：请务必使用工具来实现你的任何行为，不要直接在文本里写出你想说的话。\n"
        "请务必保持你的回复符合你的人设和表达风格，\n"
        "同时请确保你的回复有理有据，禁止无根据地编造信息或胡乱回复。"
    )
    return LLMPayload(ROLE.SYSTEM, Text(last_mile_instructions))
