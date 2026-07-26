"""KFC 调试日志格式化。

把 LLM 请求的 payload 列表渲染成可读面板，以及输出模型决策的美化摘要。
仅用于排查问题，不参与任何业务判定。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.types import Image, ROLE, Text, ToolCall

from ..models import DO_NOTHING, KFC_REPLY

if TYPE_CHECKING:
    from ..config import KFCConfig
    from ..models import ToolCallResult

logger = get_logger("kfc_debug")

_EMPTY_PROMPT_TEXT = "（无 payload）"
_EMPTY_PAYLOAD_TEXT = "（空）"

_IMAGE_PREVIEW_LIMIT = 40
_TOOL_ARGS_PREVIEW_LIMIT = 200
_CHAIN_ARGS_PREVIEW_LIMIT = 400
_RAW_ITEM_PREVIEW_LIMIT = 300
_ACTION_CONTENT_PREVIEW_LIMIT = 60


def _dump_args(args: Any, limit: int) -> str:
    """把工具参数序列化为受限长度的字符串。"""
    try:
        text = json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(args)
    return text if len(text) <= limit else text[:limit] + "..."


def _extract_payload_parts(payload: Any) -> tuple[list[str], list[dict[str, Any]]]:
    """从单个 payload 提取展示文本与工具 schema。

    Returns:
        tuple: ``(文本片段列表, 工具 schema 列表)``。
    """
    content = payload.content
    if not isinstance(content, list):
        content = [content]

    text_parts: list[str] = []
    tool_schemas: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, Text):
            text_parts.append(item.text)
        elif isinstance(item, Image):
            text_parts.append(f"[图片: {str(item.value)[:_IMAGE_PREVIEW_LIMIT]}...]")
        elif isinstance(item, ToolCall):
            args_text = _dump_args(item.args, _TOOL_ARGS_PREVIEW_LIMIT)
            text_parts.append(f"ToolCall(name={item.name!r}, args={args_text})")
        elif hasattr(item, "to_schema"):
            tool_schemas.append(item.to_schema())
        elif hasattr(item, "to_text"):
            text_parts.append(item.to_text())
        else:
            raw = str(item)
            if len(raw) > _RAW_ITEM_PREVIEW_LIMIT:
                raw = raw[:_RAW_ITEM_PREVIEW_LIMIT] + "..."
            text_parts.append(raw)
    return text_parts, tool_schemas


def _format_tools_section(tool_schemas: list[dict[str, Any]]) -> str:
    """渲染工具列表段落。"""
    lines = [
        "── TOOLS (API 参数，不进入消息流) ──",
        f"[共 {len(tool_schemas)} 个工具]",
        "",
    ]
    for index, schema in enumerate(tool_schemas, start=1):
        func_info = schema.get("function", schema)
        lines.append(f"{index}. {func_info.get('name', 'unknown')}")
        lines.append(f"   描述: {func_info.get('description', '（无描述）')}")

        parameters = func_info.get("parameters", {})
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        if properties:
            required = parameters.get("required", [])
            lines.append("   参数:")
            for param_name, param_info in properties.items():
                requirement = "必需" if param_name in required else "可选"
                lines.append(
                    f"     - {param_name} ({param_info.get('type', 'unknown')}) "
                    f"[{requirement}]: {param_info.get('description', '')}"
                )
        lines.append("")
    return "\n".join(lines).rstrip()


def format_prompt_for_log(
    response: Any,
    chain_payloads: list[dict[str, Any]] | None = None,
) -> str:
    """把请求 payload 渲染成可读的提示词面板。

    渲染顺序按阅读习惯组织：人设与规范在最前，工具列表居中，历史对话
    与最新消息贴在末尾——这样上下文脉络是连贯的。

    Args:
        response: LLM 请求或响应对象。
        chain_payloads: 存档对话链，用于把历史 assistant 文本还原为
            工具调用展示，便于对照模型当时的真实动作。

    Returns:
        str: 渲染后的提示词文本。
    """
    payloads = response.payloads
    if not payloads:
        return _EMPTY_PROMPT_TEXT

    chain_tool_calls: dict[str, list[dict[str, Any]]] = {}
    for entry in chain_payloads or []:
        if entry.get("role") == "assistant" and entry.get("tool_calls"):
            chain_tool_calls[str(entry.get("text", "") or "")] = entry["tool_calls"]

    system_parts: list[str] = []
    convo_parts: list[str] = []
    all_tool_schemas: list[dict[str, Any]] = []

    for payload in payloads:
        role = payload.role
        text_parts, tool_schemas = _extract_payload_parts(payload)

        # TOOL role 对应 API 的 tools 参数，不进入消息流
        if role == ROLE.TOOL:
            all_tool_schemas.extend(tool_schemas)
            continue

        role_name = str(role.value).upper()
        restored = _restore_chain_tool_calls(role, text_parts, chain_tool_calls)
        if restored is not None:
            convo_parts.append(f"── {role_name} ──\n{restored}")
            continue

        text = "\n".join(text_parts) if text_parts else _EMPTY_PAYLOAD_TEXT
        line = f"── {role_name} ──\n{text}"
        if role == ROLE.SYSTEM:
            system_parts.append(line)
        else:
            convo_parts.append(line)

    sections: list[str] = list(system_parts)
    if all_tool_schemas:
        sections.append(_format_tools_section(all_tool_schemas))
    sections.extend(convo_parts)
    return "\n\n".join(sections) if sections else _EMPTY_PROMPT_TEXT


def _restore_chain_tool_calls(
    role: Any,
    text_parts: list[str],
    chain_tool_calls: dict[str, list[dict[str, Any]]],
) -> str | None:
    """尝试把历史 assistant 文本还原为工具调用展示。

    Returns:
        str | None: 还原后的展示文本；无对应记录时返回 ``None``。
    """
    if role != ROLE.ASSISTANT or len(text_parts) != 1:
        return None
    tool_calls = chain_tool_calls.get(text_parts[0])
    if not tool_calls:
        return None

    return "\n".join(
        f"ToolCall(name={call.get('name', '?')!r}, "
        f"args={_dump_args(call.get('args', {}), _CHAIN_ARGS_PREVIEW_LIMIT)})"
        for call in tool_calls
    )


def log_kfc_result(result: ToolCallResult, config: KFCConfig) -> None:
    """输出模型本轮决策的美化摘要。

    Args:
        result: 执行层产出的工具执行结果。
        config: KFC 配置，据此决定是否输出。
    """
    if not config.debug.show_response:
        return

    if result.thought:
        logger.info(f"[bold magenta]💭[/bold magenta] {result.thought}")

    for action in result.actions:
        action_type = action.get("type", "")
        if action_type == KFC_REPLY:
            _log_reply_action(action.get("content"))
        elif action_type == DO_NOTHING:
            logger.info("[bold yellow]⏳[/bold yellow] 选择不回复")
        elif action_type:
            logger.info(f"[bold cyan]{action_type}[/bold cyan]")

    meta_parts: list[str] = []
    if result.max_wait_seconds > 0:
        meta_parts.append(f"⏱ {result.max_wait_seconds:.0f}s")
    if result.expected_reaction:
        meta_parts.append(f"预期: {result.expected_reaction}")
    if result.mood:
        meta_parts.append(f"心情: {result.mood}")
    if meta_parts:
        logger.info(f"[dim]{' | '.join(meta_parts)}[/dim]")


def _log_reply_action(content: Any) -> None:
    """输出回复动作的内容，多段时带序号。"""
    if not content:
        return
    if not isinstance(content, list):
        logger.info(f"[bold green]💬[/bold green] {content}")
        return

    total = len(content)
    for index, segment in enumerate(content, start=1):
        prefix = f"[{index}/{total}] " if total > 1 else ""
        logger.info(f"[bold green]💬[/bold green] {prefix}{segment}")
