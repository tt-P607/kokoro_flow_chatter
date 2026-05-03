"""KFC 第三方上下文贡献接入点。"""

from __future__ import annotations

from typing import Any, cast, get_args

from src.app.plugin_system.api.log_api import get_logger

from ..types import ContextContribution, ContextOwner, ContextScope


logger = get_logger("kfc_context_plugin_source")

_VALID_CONTEXT_OWNERS = frozenset(get_args(ContextOwner))
_VALID_CONTEXT_SCOPES = frozenset(get_args(ContextScope))


def _normalize_context_contribution(raw: Any) -> ContextContribution | None:
    """将第三方返回值归一化为 ContextContribution。"""
    if isinstance(raw, ContextContribution):
        return raw
    if not isinstance(raw, dict):
        return None

    try:
        content = str(raw.get("content", "") or "").strip()
        if not content:
            return None

        owner = str(raw.get("owner", "notice") or "notice")
        scope = str(raw.get("scope", "turn") or "turn")
        normalized_owner = owner if owner in _VALID_CONTEXT_OWNERS else "notice"
        normalized_scope = scope if scope in _VALID_CONTEXT_SCOPES else "turn"

        return ContextContribution(
            source=str(raw.get("source", "plugin.on_prompt_build") or "plugin.on_prompt_build"),
            owner=cast(ContextOwner, normalized_owner),
            scope=cast(ContextScope, normalized_scope),
            priority=int(raw.get("priority", 0) or 0),
            ttl_turns=(
                int(raw["ttl_turns"])
                if raw.get("ttl_turns") is not None
                else None
            ),
            content=content,
            evidence_only=bool(raw.get("evidence_only", False)),
        )
    except Exception:
        return None


async def collect_plugin_turn_contributions(
    *,
    prompt_name: str,
    content: str,
    stream_id: str = "",
) -> list[ContextContribution]:
    """收集第三方在本轮提交的上下文贡献。

    兼容期内继续监听 on_prompt_build，但会把 legacy extra 文本
    立即归一化成 notice/turn 的 ContextContribution，避免主流程继续
    直接拼接 raw extra user payload。
    """
    try:
        from src.app.plugin_system.api.event_api import publish_event

        template = "{content}\n{extra}"
        values: dict[str, Any] = {"content": content, "extra": "", "stream_id": stream_id}
        result = await publish_event(
            "on_prompt_build",
            {
                "name": prompt_name,
                "template": template,
                "values": values,
                "policies": {},
                "strict": False,
            },
        )
        final_params: dict[str, Any] = result.get("params", {})

        contributions: list[ContextContribution] = []
        for raw in final_params.get("context_contributions", []) or []:
            normalized = _normalize_context_contribution(raw)
            if normalized is not None:
                contributions.append(normalized)

        rendered_values = dict(final_params.get("values", values))
        legacy_extra = str(rendered_values.get("extra", "") or "").strip()
        if legacy_extra:
            contributions.append(
                ContextContribution(
                    source="legacy.on_prompt_build.extra",
                    owner="notice",
                    scope="turn",
                    priority=0,
                    ttl_turns=1,
                    content=legacy_extra,
                )
            )

        return contributions
    except Exception as exc:
        logger.warning(f"on_prompt_build 注入失败，将忽略额外上下文: {exc}")
        return []