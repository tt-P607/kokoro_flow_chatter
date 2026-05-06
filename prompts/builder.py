"""KFC 提示词构建器。

KFCPromptBuilder 负责在 execute() 中构建完整的系统提示词，
以及对话循环中的 User Payload 和 Timeout Payload。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.app.plugin_system.types import LLMPayload, ROLE, Text

from ..context import ContextPlanner, ContextRenderer

if TYPE_CHECKING:
    from src.app.plugin_system.types import ChatStream


class KFCPromptBuilder:
    """KFC 提示词构建器。

    从 PromptManager 中获取已注册的模板，
    填入动态变量后构建最终的系统提示词。
    """

    def __init__(self) -> None:
        """初始化上下文规划器与渲染器。"""
        self._planner = ContextPlanner()
        self._renderer = ContextRenderer()

    async def build_system_prompt(
        self,
        chat_stream: ChatStream,
        extra_vars: dict[str, Any] | None = None,
    ) -> str:
        """构建系统提示词。

        Args:
            chat_stream: 当前聊天流
            extra_vars: 额外模板变量

        Returns:
            str: 完整的系统提示词
        """
        return await self._renderer.build_system_prompt(
            chat_stream,
            extra_vars=extra_vars,
        )

    async def build_initial_payloads(
        self,
        chat_stream: ChatStream,
        config: Any,
        session: Any,
    ) -> tuple[list[LLMPayload], bool]:
        """构建 execute 启动所需的初始 payload 列表。"""
        plan = self._planner.plan_initial_context(
            chat_stream=chat_stream,
            config=config,
            session=session,
        )
        return await self._renderer.render_initial_context(
            chat_stream=chat_stream,
            plan=plan,
            mental_log=session.mental_log,
            serialized_chain_payloads=list(session.chain_payloads or []),
            build_system_prompt_fn=self.build_system_prompt,
            build_fused_narrative_fn=self.build_fused_narrative,
        )

    async def build_user_payload(
        self,
        formatted_unreads: str,
        media_items: list[Any] | None = None,
        stream_id: str = "",
    ) -> tuple[LLMPayload, LLMPayload | None]:
        """构建用户消息 Payload。

        将格式化的未读消息构建为一个 USER 角色的 Payload。
        心理活动已在融合叙事时间线中展示，不再单独注入摘要。
        如果携带多模态图片，则打包为 Text + Image 混合内容。

        触发 ``on_prompt_build`` 事件（模板名 ``kfc_user_prompt``），
        允许外部插件（如 prompt_injector）向历史末尾追加额外的独立 USER payload。
        注入内容不会拼入 user_text，而是作为单独的第二个 payload 返回，
        由调用方在发送前临时追加、发送后移除，从而不进入持久历史。

        Args:
            formatted_unreads: 格式化后的未读消息文本
            media_items: 多模态图片列表（可选，来自 extract_media_from_messages）
            stream_id: 当前聊天流 ID（供 on_prompt_build 事件处理器读取）

        Returns:
            tuple: (user_payload, extra_payload | None)
                - user_payload: USER 角色的 Payload（进入持久历史）
                - extra_payload: 注入内容的独立 USER Payload（临时，不进历史），无注入时为 None
        """
        plan = await self._planner.plan_user_turn(
            formatted_unreads=formatted_unreads,
            stream_id=stream_id,
        )
        return self._renderer.render_user_payload(plan, media_items=media_items)

    @staticmethod
    def build_timeout_payload(
        elapsed_seconds: float,
        expected_reaction: str,
        consecutive_timeouts: int,
        last_bot_message: str = "",
        max_consecutive_timeouts: int = 3,
    ) -> LLMPayload:
        """构建等待超时 Payload。

        使用丰富的超时决策提示词，引导 LLM 根据消息类型分析
        做出合理的下一步决策（继续等待、追问或结束）。

        Args:
            elapsed_seconds: 已等待秒数
            expected_reaction: 之前预期的对方反应
            consecutive_timeouts: 连续超时次数
            last_bot_message: 最后一条 Bot 发送的消息
            max_consecutive_timeouts: 配置的连续超时上限

        Returns:
            LLMPayload: USER 角色的超时 Payload
        """
        from .modules import build_timeout_context

        timeout_text = build_timeout_context(
            elapsed_seconds=elapsed_seconds,
            expected_reaction=expected_reaction,
            consecutive_timeouts=consecutive_timeouts,
            last_bot_message=last_bot_message,
            max_consecutive_timeouts=max_consecutive_timeouts,
        )

        return LLMPayload(ROLE.USER, Text(timeout_text))

    def build_fused_narrative(
        self,
        chat_stream: ChatStream,
        mental_log: Any,
        before_ts: float | None = None,
    ) -> str:
        """构建聊天历史与内心独白的融合叙事。

        将数据库聊天记录和心理活动日志按时间交织在一起，
        形成一个包含"说了什么"和"想了什么"的统一时间线。
        这是老版 KFC 的核心设计——让 LLM 在回顾历史时不仅看到
        对话内容，还能看到每个节点上自己当时的内心活动。

        消息来源：context.history_messages（受 core.toml max_context_size 管控）。
        chain_payloads 直接追加到 LLM request，不占此配额；
        二者通过 before_ts（chain_cutoff_ts）分界、互不重叠。

        Args:
            chat_stream: 当前聊天流
            mental_log: 心理活动日志（MentalLog 实例）
            before_ts: 若指定，只包含时间戳严格小于该值的消息，
                用于将叙事截止到链起始时间戳之前，避免与 chain_payloads 内容重叠。

        Returns:
            str: 融合叙事文本，无内容时返回空串
        """
        _ = self
        return ContextRenderer.build_fused_narrative(
            chat_stream,
            mental_log,
            before_ts=before_ts,
        )
