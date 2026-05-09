"""KFC 上下文规划器。"""

from __future__ import annotations

from typing import Any

from src.app.plugin_system.api.log_api import get_logger

from .sources.initial_source import build_initial_context_plan
from .sources.plugin_source import collect_plugin_turn_contributions
from .types import ContextPlan, InitialContextPlan

logger = get_logger("kfc_context_planner")


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

    @staticmethod
    def _build_last_mile_instructions() -> str:
        """构建 user 消息末尾的行为强调指令。"""
        return (
            "请基于上述信息决定接下来你要调用的工具或动作。\n"
            "重申：请务必使用工具来实现你的任何行为，不要直接在文本里写出你想说的话。\n"
            "请务必保持你的回复符合你的人设和表达风格，\n"
            "同时请确保你的回复有理有据，禁止无根据地编造信息或胡乱回复。"
        )

    async def plan_user_turn(
        self,
        *,
        formatted_unreads: str,
        stream_id: str = "",
        chat_stream: Any = None,
    ) -> ContextPlan:
        """规划本轮用户输入和第三方 turn 级上下文贡献。"""
        last_mile = self._build_last_mile_instructions()
        chain_text = f"[新消息]\n{formatted_unreads}"
        user_text = f"{chain_text}\n\n---\n{last_mile}"

        contributions = await collect_plugin_turn_contributions(
            prompt_name="kfc_user_prompt",
            content=user_text,
            stream_id=stream_id,
        )
        return ContextPlan(user_text=user_text, contributions=contributions, chain_text=chain_text)