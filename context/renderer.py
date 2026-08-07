"""KFC 上下文渲染。

把规划阶段产出的纯数据组装成 LLM payload。核心约束是**保护前缀缓存**：
SYSTEM payload 只放稳定的人设与行为规范，一切每轮变化的内容（时间、
摘要、叙事、第三方注入）都作为 USER payload 进入对话链。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.prompt_api import get_template
from src.app.plugin_system.types import Content, LLMPayload, ROLE, Text

from .sources.history_source import (
    build_channel_payload,
    build_current_time_payload,
    build_fused_narrative,
    build_history_summary_payload,
    restore_chain_payloads,
)
from .types import ContextContribution, ContextOwner, ContextPlan, InitialContextPlan

if TYPE_CHECKING:
    from src.app.plugin_system.types import ChatStream

    from ..mental_log import MentalLog

SECTION_SEPARATOR = "\n\n---\n\n"
"""动态 USER payload 内各段落的分隔符。"""

_SYSTEM_TEMPLATE_NAME = "kfc_system_prompt"

_OWNER_RENDER_ORDER: tuple[ContextOwner, ...] = (
    "policy",
    "self_state",
    "user_state",
    "relationship_state",
    "notice",
)
"""上下文贡献的渲染顺序：约束在前、状态居中、通知在后。"""

_OWNER_SECTION_TITLES: dict[str, str] = {
    "policy": "[策略约束]",
    "self_state": "[你的状态]",
    "user_state": "[对方状态]",
    "relationship_state": "[关系状态]",
}
"""各分区的标题；``notice`` 无标题，直接平铺。"""

_LEGACY_EXTRA_SOURCE = "legacy.on_prompt_build.extra"
_LEGACY_EXTRA_PREFIX = "[SYSTEM REMINDER]"


def _collect_payload_text(payload: LLMPayload) -> str:
    """把 payload 内所有文本片段拼接为单个字符串。"""
    content = payload.content
    if not isinstance(content, list):
        content = [content]
    return "".join(item.text for item in content if isinstance(item, Text))


async def build_system_prompt(
    chat_stream: ChatStream,
    extra_vars: dict[str, str] | None = None,
) -> str:
    """构建稳定的系统提示词。

    KFC 的系统提示词是前缀缓存的锚点，因此**不发布** ``on_prompt_build``
    事件——第三方动态注入统一走 ``kfc_user_prompt`` 的上下文贡献通道。

    Args:
        chat_stream: 当前聊天流。
        extra_vars: 额外模板变量，如工具说明、预约状态。

    Returns:
        str: 渲染后的系统提示词；模板缺失时返回空串。
    """
    from ..prompts.modules import build_mental_log_hint

    template_base = get_template(_SYSTEM_TEMPLATE_NAME)
    if not template_base:
        return ""

    template = template_base.clone()
    template.set("platform", chat_stream.platform or "unknown")
    template.set("chat_type", str(chat_stream.chat_type or "unknown"))
    template.set("bot_id", chat_stream.bot_id or "")
    template.set("stream_id", chat_stream.stream_id or "")
    template.set("mental_log_hint", build_mental_log_hint())
    for key, value in (extra_vars or {}).items():
        template.set(key, value)

    # 直接调用内部渲染以跳过 on_prompt_build 事件，保证系统前缀稳定。
    return template._render(  # noqa: SLF001
        template.template,
        dict(template.values),
        dict(template.policies),
        strict=False,
    )


async def render_initial_context(
    *,
    chat_stream: ChatStream,
    plan: InitialContextPlan,
    mental_log: MentalLog | None,
    serialized_chain_payloads: list[dict[str, Any]],
    skip_narrative: bool = False,
    build_system_prompt_fn: Callable[
        [ChatStream, dict[str, str] | None], Awaitable[str]
    ]
    | None = None,
    build_fused_narrative_fn: Callable[[ChatStream, Any, float | None], str]
    | None = None,
) -> tuple[list[LLMPayload], list[LLMPayload], bool]:
    """渲染 ``execute()`` 启动所需的初始 payload。

    产出分为两组：``system_payloads`` 只含稳定系统提示词；
    ``chain_payloads`` 依次是「当前通道 + 记忆摘要 + 融合叙事」合并的
    动态 USER payload，以及从存档还原的历史对话链。先说明当前背景、
    再展开过去对话，语义顺序更自然。

    当以完整上下文快照恢复历史链时，应传 ``skip_narrative=True``：快照链
    本身已是最新最完整的近期对话（含工具调用与结果），此时再生成融合叙事
    会与快照链重叠（同进程内 ``history_messages`` 仍在内存）。

    Args:
        chat_stream: 当前聊天流。
        plan: 初始上下文规划结果。
        mental_log: 心理活动流，供融合叙事使用。
        serialized_chain_payloads: 存档中的对话链条目。
        skip_narrative: 是否跳过融合叙事（快照恢复历史链时传 True）。
        build_system_prompt_fn: 系统提示词构建器，默认用本模块实现。
        build_fused_narrative_fn: 融合叙事构建器，默认用本模块实现。

    Returns:
        tuple: ``(system_payloads, chain_payloads, has_history)``。
    """
    system_prompt_builder = build_system_prompt_fn or build_system_prompt
    narrative_builder = build_fused_narrative_fn or build_fused_narrative

    system_prompt = await system_prompt_builder(chat_stream, plan.system_extra_vars)
    system_payloads = [LLMPayload(ROLE.SYSTEM, Text(system_prompt))]

    dynamic_parts: list[str] = [
        _collect_payload_text(build_channel_payload(chat_stream))
    ]

    summary_payload = build_history_summary_payload(chat_stream, plan.history_summary)
    if summary_payload is not None:
        dynamic_parts.append(_collect_payload_text(summary_payload))

    has_history = False
    if not skip_narrative:
        history_text = narrative_builder(chat_stream, mental_log, plan.history_before_ts)
        if not history_text:
            history_text = _collect_payload_text(build_current_time_payload())
        dynamic_parts.append(history_text)
        has_history = bool(history_text)
    else:
        # 快照恢复路径跳过融合叙事，但仍需当前时间锚点，
        # 让恢复后的模型知道"现在"是什么时候。
        dynamic_parts.append(_collect_payload_text(build_current_time_payload()))

    chain_payloads: list[LLMPayload] = [
        LLMPayload(ROLE.USER, Text(SECTION_SEPARATOR.join(dynamic_parts)))
    ]
    restored_payloads = restore_chain_payloads(serialized_chain_payloads)
    chain_payloads.extend(restored_payloads)

    return (
        system_payloads,
        chain_payloads,
        has_history or bool(restored_payloads),
    )


def render_user_payload(
    plan: ContextPlan,
    media_items: list[dict[str, Any]] | None = None,
) -> tuple[LLMPayload, LLMPayload | None, str]:
    """把用户回合规划渲染为 payload。

    Args:
        plan: 用户回合规划结果。
        media_items: 原生多模态图片列表；非空时打包为图文混合内容。

    Returns:
        tuple: ``(user_payload, extra_payload | None, chain_text)``。
        ``extra_payload`` 为本轮临时注入，发送后不入对话链。
    """
    content: Content | list[Content]
    if media_items:
        from ..multimodal import build_multimodal_content

        content = build_multimodal_content(plan.user_text, media_items)
    else:
        content = Text(plan.user_text)

    user_payload = LLMPayload(ROLE.USER, content)  # type: ignore[arg-type]
    extra_payload = render_turn_contributions(plan.contributions)
    return user_payload, extra_payload, plan.chain_text or plan.user_text


def render_turn_contributions(
    contributions: list[ContextContribution],
) -> LLMPayload | None:
    """把 turn 级上下文贡献渲染为临时 USER payload。

    Args:
        contributions: 待渲染的贡献列表。

    Returns:
        LLMPayload | None: 渲染结果；无有效内容时返回 ``None``。
    """
    valid_contributions = [
        contribution for contribution in contributions if contribution.content.strip()
    ]
    if not valid_contributions:
        return None

    blocks = [
        block
        for block in (
            _render_owner_block(owner, valid_contributions)
            for owner in _OWNER_RENDER_ORDER
        )
        if block
    ]
    if not blocks:
        return None

    return LLMPayload(ROLE.USER, Text("[附加上下文]\n" + "\n\n".join(blocks)))


def _render_owner_block(
    owner: ContextOwner,
    contributions: list[ContextContribution],
) -> str:
    """渲染单个 owner 分区；无内容时返回空串。"""
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
    rendered = [
        _render_contribution_text(contribution) for contribution in owner_contributions
    ]
    if not rendered:
        return ""

    body = "\n\n".join(rendered)
    section_title = _OWNER_SECTION_TITLES.get(owner, "")
    return f"{section_title}\n{body}" if section_title else body


def _render_contribution_text(contribution: ContextContribution) -> str:
    """渲染单条贡献；legacy 文本补上系统提醒前缀以区分来源。"""
    content = contribution.content.strip()
    if contribution.source != _LEGACY_EXTRA_SOURCE:
        return content
    if content.startswith(_LEGACY_EXTRA_PREFIX):
        return content
    return f"{_LEGACY_EXTRA_PREFIX}\n{content}"
