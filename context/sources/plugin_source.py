"""KFC 第三方上下文接入点。

通过 ``on_prompt_build`` 事件收集其他插件为本轮提供的上下文，
并统一归一化为 ``ContextContribution``，使主流程无需关心 legacy
的裸文本拼接形态。
"""

from __future__ import annotations

from typing import Any, cast, get_args

from src.app.plugin_system.api.log_api import get_logger

from ..types import ContextContribution, ContextOwner

logger = get_logger("kfc_context_source")

_VALID_OWNERS = frozenset(get_args(ContextOwner))
_DEFAULT_OWNER: ContextOwner = "notice"
_LEGACY_SOURCE = "legacy.on_prompt_build.extra"
_EVENT_TEMPLATE = "{content}\n{extra}"


def _normalize_contribution(raw: Any) -> ContextContribution | None:
    """把第三方返回值归一化为 ``ContextContribution``。

    Args:
        raw: 订阅者提供的贡献，可为实例或字典。

    Returns:
        ContextContribution | None: 归一化结果；内容为空或结构非法时返回 ``None``。
    """
    if isinstance(raw, ContextContribution):
        return raw
    if not isinstance(raw, dict):
        return None

    content = str(raw.get("content", "") or "").strip()
    if not content:
        return None

    owner = str(raw.get("owner", _DEFAULT_OWNER) or _DEFAULT_OWNER)
    try:
        priority = int(raw.get("priority", 0) or 0)
    except (TypeError, ValueError):
        priority = 0

    return ContextContribution(
        source=str(raw.get("source", "plugin.on_prompt_build") or "plugin.on_prompt_build"),
        owner=cast(ContextOwner, owner if owner in _VALID_OWNERS else _DEFAULT_OWNER),
        priority=priority,
        content=content,
    )


async def collect_plugin_turn_contributions(
    *,
    prompt_name: str,
    content: str,
    stream_id: str = "",
) -> list[ContextContribution]:
    """收集第三方为本轮提交的上下文贡献。

    Args:
        prompt_name: 提示词模板名，供订阅者区分注入目标。
        content: 当前用户提示词正文，供订阅者参考。
        stream_id: 当前聊天流 ID。

    Returns:
        list[ContextContribution]: 归一化后的贡献列表；事件失败时返回空列表。
    """
    try:
        from src.app.plugin_system.api.event_api import publish_event

        values: dict[str, Any] = {
            "content": content,
            "extra": "",
            "stream_id": stream_id,
        }
        result = await publish_event(
            "on_prompt_build",
            {
                "name": prompt_name,
                "template": _EVENT_TEMPLATE,
                "values": values,
                "policies": {},
                "strict": False,
            },
        )
        final_params: dict[str, Any] = result.get("params", {})

        contributions: list[ContextContribution] = []
        for raw in final_params.get("context_contributions", []) or []:
            normalized = _normalize_contribution(raw)
            if normalized is not None:
                contributions.append(normalized)

        rendered_values = dict(final_params.get("values", values))
        legacy_extra = str(rendered_values.get("extra", "") or "").strip()
        if legacy_extra:
            contributions.append(
                ContextContribution(
                    source=_LEGACY_SOURCE,
                    owner=_DEFAULT_OWNER,
                    priority=0,
                    content=legacy_extra,
                )
            )
        return contributions
    except Exception as error:
        logger.warning(f"on_prompt_build 注入失败，忽略额外上下文: {error}")
        return []
