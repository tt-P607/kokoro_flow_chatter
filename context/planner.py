"""KFC 上下文规划器。"""

from __future__ import annotations

from typing import Any

from .sources.initial_source import build_initial_context_plan
from .sources.plugin_source import collect_plugin_turn_contributions
from .types import ContextPlan, InitialContextPlan


class ContextPlanner:
    """负责把单轮输入转换成结构化上下文计划。"""

    def plan_initial_context(
        self,
        *,
        chat_stream: Any,
        config: Any,
        session: Any,
    ) -> InitialContextPlan:
        """规划 execute 启动时所需的初始上下文数据。"""
        return build_initial_context_plan(
            chat_stream=chat_stream,
            config=config,
            session=session,
        )

    async def plan_user_turn(
        self,
        *,
        formatted_unreads: str,
        stream_id: str = "",
    ) -> ContextPlan:
        """规划本轮用户输入和第三方 turn 级上下文贡献。"""
        user_text = f"[新消息]\n{formatted_unreads}"
        contributions = await collect_plugin_turn_contributions(
            prompt_name="kfc_user_prompt",
            content=user_text,
            stream_id=stream_id,
        )
        return ContextPlan(user_text=user_text, contributions=contributions)