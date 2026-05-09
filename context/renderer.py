"""KFC 上下文渲染器。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.prompt_api import get_template as _get_prompt_template
from src.app.plugin_system.types import Content, LLMPayload, ROLE, Text

from .sources.history_source import (
    build_current_time_payload,
    build_fused_narrative as build_history_narrative,
    build_history_summary_payload,
    restore_chain_payloads as restore_history_chain_payloads,
)
from .types import ContextContribution, ContextPlan, InitialContextPlan

if TYPE_CHECKING:
    from src.app.plugin_system.types import ChatStream


class ContextRenderer:
    """负责把 ContextPlan 和历史状态渲染成 LLM payload。"""

    _OWNER_RENDER_ORDER: tuple[str, ...] = (
        "policy",
        "self_state",
        "user_state",
        "relationship_state",
        "scene_evidence",
        "notice",
    )
    _OWNER_SECTION_TITLES: dict[str, str] = {
        "policy": "[策略约束]",
        "self_state": "[你的状态]",
        "user_state": "[对方状态]",
        "relationship_state": "[关系状态]",
        "scene_evidence": "[场景证据]",
    }

    async def render_initial_context(
        self,
        *,
        chat_stream: ChatStream,
        plan: InitialContextPlan,
        mental_log: Any,
        serialized_chain_payloads: list[dict[str, Any]],
        build_system_prompt_fn: Callable[
            [ChatStream, dict[str, Any] | None], Awaitable[str]
        ]
        | None = None,
        build_fused_narrative_fn: Callable[[ChatStream, Any, float | None], str]
        | None = None,
    ) -> tuple[list[LLMPayload], list[LLMPayload], bool]:
        """渲染 execute 启动时需要的初始 payload 列表。

        Returns:
            tuple: (system_payloads, chain_payloads, has_history)
                - system_payloads: 系统 Payload 列表（通过 add_payload 注入，触发 reminder）
                - chain_payloads: 历史链 Payload 列表（直接追加，绕过 context manager 避免
                  对历史 USER 重复注入 system_reminder）
                - has_history: 是否存在历史内容
        """
        system_payloads: list[LLMPayload] = []

        system_prompt_builder = build_system_prompt_fn or self.build_system_prompt
        fused_narrative_builder = (
            build_fused_narrative_fn or self.build_fused_narrative
        )

        # 1. 构建各部分文本
        system_prompt = await system_prompt_builder(
            chat_stream,
            plan.system_extra_vars,
        )

        from .sources.history_source import (
            build_history_summary_payload,
            build_current_time_payload,
        )

        summary_text = ""
        summary_payload = build_history_summary_payload(
            chat_stream,
            plan.history_summary,
        )
        if summary_payload is not None:
            for item in summary_payload.content:
                if hasattr(item, "text"):
                    summary_text += item.text  # type: ignore[attr-defined]

        history_text = fused_narrative_builder(
            chat_stream,
            mental_log,
            plan.history_before_ts,
        )
        if not history_text:
            time_payload = build_current_time_payload()
            for item in time_payload.content:
                if hasattr(item, "text"):
                    history_text += item.text  # type: ignore[attr-defined]

        # 2. 拆分为两个 SYSTEM Payload 以优化缓存命中
        # Payload 1: 静态部分 (人设/准则/平台信息) - 几乎永远不变，缓存价值最高
        system_payloads.append(LLMPayload(ROLE.SYSTEM, Text(system_prompt)))

        # Payload 2: 动态部分 (记忆摘要/融合叙事) - 随对话进度变化
        dynamic_parts: list[str] = []
        if summary_text:
            dynamic_parts.append(summary_text)
        if history_text:
            dynamic_parts.append(history_text)

        if dynamic_parts:
            merged_dynamic_text = "\n\n---\n\n".join(dynamic_parts)
            system_payloads.append(LLMPayload(ROLE.SYSTEM, Text(merged_dynamic_text)))

        # 3. 恢复工具调用链 — 独立返回，绕过 add_payload 避免
        #    context manager 对历史 USER 重复注入 system_reminder
        chain_payloads = restore_history_chain_payloads(serialized_chain_payloads)

        return system_payloads, chain_payloads, bool(history_text) or bool(chain_payloads)

    async def build_system_prompt(
        self,
        chat_stream: ChatStream,
        extra_vars: dict[str, Any] | None = None,
    ) -> str:
        """构建系统提示词。"""
        from ..prompts.modules import build_mental_log_hint

        tmpl_base = _get_prompt_template("kfc_system_prompt")
        if not tmpl_base:
            return ""

        tmpl = tmpl_base.clone()
        tmpl.set("platform", chat_stream.platform or "unknown")
        tmpl.set("chat_type", str(chat_stream.chat_type or "unknown"))
        tmpl.set("bot_id", chat_stream.bot_id or "")
        tmpl.set("stream_id", chat_stream.stream_id or "")
        tmpl.set("mental_log_hint", build_mental_log_hint())
        tmpl.set("theme_guide", "")

        if extra_vars:
            for key, value in extra_vars.items():
                tmpl.set(key, value)

        return await tmpl.build()

    def render_user_payload(
        self,
        plan: ContextPlan,
        media_items: list[Any] | None = None,
    ) -> tuple[LLMPayload, LLMPayload | None, str]:
        """将 ContextPlan 渲染为用户 payload。

        Returns:
            tuple: (user_payload, extra_payload | None, chain_text)
                - chain_text: 仅含原始消息内容，不含末尾强调指令，用于链持久化
        """
        content: Content | list[Content]
        if media_items:
            from ..multimodal import build_multimodal_content

            content = build_multimodal_content(plan.user_text, media_items)
        else:
            content = Text(plan.user_text)

        user_payload = LLMPayload(ROLE.USER, content)  # type: ignore[arg-type]
        extra_payload = self.render_turn_contributions(plan.contributions)
        return user_payload, extra_payload, plan.chain_text or plan.user_text

    def render_turn_contributions(
        self,
        contributions: list[ContextContribution],
    ) -> LLMPayload | None:
        """将 turn 级上下文贡献渲染为临时 USER payload。"""
        turn_contributions = [
            contribution
            for contribution in contributions
            if contribution.scope == "turn" and contribution.content.strip()
        ]
        if not turn_contributions:
            return None

        joined_contents = "\n\n".join(
            self._render_owner_contribution_block(owner, turn_contributions)
            for owner in self._OWNER_RENDER_ORDER
            if self._render_owner_contribution_block(owner, turn_contributions)
        )
        if not joined_contents:
            return None

        return LLMPayload(
            ROLE.USER,
            Text(f"[附加上下文]\n{joined_contents}"),
        )

    def _render_owner_contribution_block(
        self,
        owner: str,
        contributions: list[ContextContribution],
    ) -> str:
        """按 owner 渲染 turn contribution 分区。"""
        owner_contributions = sorted(
            (
                contribution
                for contribution in contributions
                if contribution.owner == owner
            ),
            key=lambda contribution: (
                -contribution.priority,
                contribution.source,
                contribution.content,
            ),
        )
        rendered_contents = [
            rendered
            for rendered in (
                self._render_contribution_text(contribution)
                for contribution in owner_contributions
            )
            if rendered
        ]
        if not rendered_contents:
            return ""

        joined_contents = "\n\n".join(rendered_contents)
        section_title = self._OWNER_SECTION_TITLES.get(owner, "")
        if not section_title:
            return joined_contents
        return f"{section_title}\n{joined_contents}"

    @staticmethod
    def _render_contribution_text(contribution: ContextContribution) -> str:
        """渲染单条上下文贡献文本。"""
        content = contribution.content.strip()
        if not content:
            return ""

        if contribution.source == "legacy.on_prompt_build.extra":
            if content.startswith("[SYSTEM REMINDER]"):
                return content
            return f"[SYSTEM REMINDER]\n{content}"

        return content

    @staticmethod
    def build_fused_narrative(
        chat_stream: ChatStream,
        mental_log: Any,
        before_ts: float | None = None,
    ) -> str:
        """构建聊天历史与内心独白的融合叙事。"""
        return build_history_narrative(
            chat_stream,
            mental_log,
            before_ts=before_ts,
        )