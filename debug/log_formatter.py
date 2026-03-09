"""KFC 调试日志格式化工具。

将 LLM 请求/响应的 payload 列表格式化为人类可读的面板输出，
以及 ToolCallResult 的美化摘要。
"""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from src.app.plugin_system.api.log_api import get_logger
from src.kernel.llm import ROLE

from ..models import KFC_REPLY, DO_NOTHING

if TYPE_CHECKING:
    from ..config import KFCConfig
    from ..models import ToolCallResult

logger = get_logger("kfc_debug")


def _extract_payload_text(payload: Any) -> tuple[list[str], list[dict[str, Any]]]:
    """从单个 payload 中提取文本片段和工具 schema。

    Returns:
        tuple: (text_parts, tool_schemas)
    """
    content_list = getattr(payload, "content", [])
    if not isinstance(content_list, list):
        content_list = [content_list]

    text_parts: list[str] = []
    tool_schemas: list[dict[str, Any]] = []
    for item in content_list:
        if hasattr(item, "text"):
            text_parts.append(item.text)
        elif hasattr(item, "value") and hasattr(item, "__class__") and item.__class__.__name__ == "Image":
            data_preview = str(item.value)[:40]
            text_parts.append(f"[图片: {data_preview}...]")
        elif hasattr(item, "to_text"):
            text_parts.append(item.to_text())
        elif hasattr(item, "to_schema"):
            schema = item.to_schema()
            tool_schemas.append(schema)
        elif hasattr(item, "name") and hasattr(item, "args") and hasattr(item, "id"):
            # ToolCall：仅展示名称和参数，截断 id 中的 base64
            name = getattr(item, "name", "?")
            args = getattr(item, "args", {})
            try:
                args_str = json.dumps(args, ensure_ascii=False)
            except Exception:
                args_str = str(args)
            if len(args_str) > 200:
                args_str = args_str[:200] + "..."
            text_parts.append(f"ToolCall(name={name!r}, args={args_str})")
        else:
            raw = str(item)
            if len(raw) > 300:
                raw = raw[:300] + "..."
            text_parts.append(raw)
    return text_parts, tool_schemas


def format_prompt_for_log(response: Any) -> str:
    """从 LLM request/response 的 payload 列表中提取并格式化提示词。

    渲染顺序（符合阅读习惯，首尾最重要）：
    1. SYSTEM（人设 / 关系）：固定在最前，快速定位角色设定
    2. TOOLS（API tools 参数，不进入消息流）：放中间，了解可用能力
    3. SYSTEM（历史叙事）+ 对话轮次 + 新消息：贴在一块放末尾，
       历史叙事与最新消息上下文相邻，便于阅读完整的对话脉络

    历史叙事识别方式：context 构建时注入顺序固定为
    人设 → 关系文本 → 历史叙事，因此最后一个 SYSTEM payload
    即为历史叙事（仅当 SYSTEM > 1 条时才拆分）。

    Args:
        response: LLMRequest 或 LLMResponse 对象

    Returns:
        str: 格式化后的提示词文本
    """
    payloads = getattr(response, "payloads", None)
    if not payloads:
        return "（无 payload）"

    system_parts: list[str] = []
    convo_parts: list[str] = []
    all_tool_schemas: list[dict[str, Any]] = []

    for payload in payloads:
        role = getattr(payload, "role", None)
        text_parts, tool_schemas = _extract_payload_text(payload)

        if role == ROLE.TOOL:
            # TOOL role 对应 API 的 tools 参数，不进入消息流，单独收集
            all_tool_schemas.extend(tool_schemas)
            continue

        role_name = str(role.value).upper() if hasattr(role, "value") else str(role)
        text = "\n".join(text_parts) if text_parts else "（空）"
        line = f"── {role_name} ──\n{text}"

        if role == ROLE.SYSTEM:
            system_parts.append(line)
        else:
            convo_parts.append(line)

    # 拆分 SYSTEM：当只有 1 条时全归人设段；≥2 条时最后一条为历史叙事
    if len(system_parts) >= 2:
        persona_parts = system_parts[:-1]
        history_part = system_parts[-1]
    else:
        persona_parts = system_parts
        history_part = None

    sections: list[str] = []

    # 1. 人设 / 关系（最前）
    if persona_parts:
        sections.extend(persona_parts)

    # 2. 工具列表（中间）
    if all_tool_schemas:
        tool_count = len(all_tool_schemas)
        tools_section = f"── TOOLS (API 参数，不进入消息流) ──\n"
        tools_section += f"[共 {tool_count} 个工具]\n\n"
        
        for i, schema in enumerate(all_tool_schemas, 1):
            func_info = schema.get("function", schema)
            name = func_info.get("name", "unknown")
            desc = func_info.get("description", "（无描述）")
            params = func_info.get("parameters", {})
            
            tools_section += f"{i}. {name}\n"
            tools_section += f"   描述: {desc}\n"
            
            if params and isinstance(params, dict):
                required = params.get("required", [])
                properties = params.get("properties", {})
                if properties:
                    tools_section += f"   参数:\n"
                    for param_name, param_info in properties.items():
                        param_desc = param_info.get("description", "")
                        param_type = param_info.get("type", "unknown")
                        is_required = "必需" if param_name in required else "可选"
                        tools_section += f"     - {param_name} ({param_type}) [{is_required}]: {param_desc}\n"
            tools_section += "\n"
        
        sections.append(tools_section.rstrip())

    # 3. 历史叙事 + 对话轮次 + 新消息（末尾，紧密相邻）
    if history_part:
        sections.append(history_part)
    if convo_parts:
        sections.extend(convo_parts)

    return "\n\n".join(sections) if sections else "（无 payload）"


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
            content = action.get("content")
            if content:
                if isinstance(content, list):
                    for i, seg in enumerate(content, 1):
                        prefix = f"[{i}/{len(content)}] " if len(content) > 1 else ""
                        logger.info(f"[bold green]💬[/bold green] {prefix}{seg}")
                else:
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
