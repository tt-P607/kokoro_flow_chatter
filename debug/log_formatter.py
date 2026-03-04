"""KFC 调试日志格式化工具。

将 LLM 请求/响应的 payload 列表格式化为人类可读的面板输出，
以及 ToolCallResult 的美化摘要。
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from src.app.plugin_system.api.log_api import get_logger

from ..models import KFC_REPLY, DO_NOTHING

if TYPE_CHECKING:
    from ..config import KFCConfig
    from ..models import ToolCallResult

logger = get_logger("kfc_debug")


def format_prompt_for_log(response: Any) -> str:
    """从 LLM request/response 的 payload 列表中提取并格式化提示词。

    Args:
        response: LLMRequest 或 LLMResponse 对象

    Returns:
        str: 格式化后的提示词文本
    """
    payloads = getattr(response, "payloads", None)
    if not payloads:
        return "（无 payload）"

    parts: list[str] = []
    for payload in payloads:
        role = getattr(payload, "role", None)
        role_name = str(role.value).upper() if hasattr(role, "value") else str(role)

        content_list = getattr(payload, "content", [])
        if not isinstance(content_list, list):
            content_list = [content_list]

        text_parts: list[str] = []
        tool_names: list[str] = []
        for item in content_list:
            if hasattr(item, "text"):
                text_parts.append(item.text)
            elif (
                hasattr(item, "value")
                and hasattr(item, "__class__")
                and item.__class__.__name__ == "Image"
            ):
                data_preview = str(item.value)[:40]
                text_parts.append(f"[图片: {data_preview}...]")
            elif hasattr(item, "to_text"):
                text_parts.append(item.to_text())
            elif hasattr(item, "to_schema"):
                schema = item.to_schema()
                func_info = schema.get("function", schema)
                name = func_info.get("name", type(item).__name__)
                tool_names.append(name)
            else:
                text_parts.append(str(item))

        tool_count = len(tool_names)
        tool_summary = (
            f"[{tool_count} 个工具: {', '.join(tool_names)}]"
            if tool_names
            else ""
        )
        if tool_count > 0 and not text_parts:
            text = tool_summary
        elif tool_count > 0:
            text = "\n".join(text_parts) + f"\n[+ {tool_summary}]"
        elif text_parts:
            text = "\n".join(text_parts)
        else:
            text = "（空）"

        parts.append(f"── {role_name} ──\n{text}")

    return "\n\n".join(parts)


def log_kfc_result(result: ToolCallResult, config: KFCConfig) -> None:
    """美化输出 LLM 响应摘要。

    Args:
        result: 工具调用解析结果
        config: KFC 配置
    """
    if not config.debug.show_response:
        return

    if result.thought:
        logger.info(f"[bold magenta]💭[/bold magenta] {result.thought}")

    for action in result.actions:
        action_type = action.get("type", "")
        if action_type in (KFC_REPLY, "respond"):
            content = action.get("content", "")
            if content:
                logger.info(f"[bold green]💬[/bold green] {content}")
        elif action_type == DO_NOTHING:
            logger.info("[bold yellow]⏳[/bold yellow] 选择不回复")
        elif action_type not in ("no_action",):
            logger.info(f"[bold cyan]🎯[/bold cyan] {action_type}")

    meta_parts: list[str] = []
    if result.max_wait_seconds > 0:
        meta_parts.append(f"⏱ {result.max_wait_seconds:.0f}s")
    if result.expected_reaction:
        meta_parts.append(f"预期: {result.expected_reaction}")
    if result.mood:
        meta_parts.append(f"心情: {result.mood}")
    if meta_parts:
        logger.info(f"[dim]{' | '.join(meta_parts)}[/dim]")
