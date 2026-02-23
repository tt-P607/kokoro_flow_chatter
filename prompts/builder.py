"""KFC 提示词构建器。

KFCPromptBuilder 负责在 execute() 中构建完整的系统提示词。
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

from src.core.prompt import get_prompt_manager

from .modules import build_mental_log_hint

if TYPE_CHECKING:
    from src.core.models.stream import ChatStream


def build_extra_action_types(tool_schemas: list[dict[str, Any]]) -> str:
    """将第三方工具 schema 转为 JSON 动作类型描述文本。

    每个工具生成一行描述 + 参数说明，让 LLM 可以通过 JSON actions 调用。

    Args:
        tool_schemas: 工具的 OpenAI Tool schema 列表

    Returns:
        str: 动作类型描述文本，为空时返回空字符串
    """
    if not tool_schemas:
        return ""

    lines: list[str] = [
        "",
        "以下是你可以使用的额外能力（直接在 actions 中使用即可）："
    ]
    for schema in tool_schemas:
        func = schema.get("function", schema)
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = set(params.get("required", []))

        # 构建参数描述
        param_parts: list[str] = []
        for pname, pinfo in props.items():
            if pname == "reason":
                # reason 由框架自动注入的元参数，不暴露给 LLM
                continue
            ptype = pinfo.get("type", "string")
            pdesc = pinfo.get("description", "")
            is_required = pname in required
            suffix = "" if is_required else "，可选"
            param_parts.append(f"{pname}({ptype}{suffix}): {pdesc}")

        param_text = "；".join(param_parts) if param_parts else "无参数"
        lines.append(f"- {name} — {desc}。参数: {param_text}")

    return "\n".join(lines)


class KFCPromptBuilder:
    """KFC 提示词构建器。

    从 PromptManager 中获取已注册的模板，
    填入动态变量后构建最终的系统提示词。
    """

    def __init__(self, log_format: str = "narrative") -> None:
        self._log_format = log_format

    def build_system_prompt(
        self,
        chat_stream: ChatStream,
        extra_vars: dict[str, Any] | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> str:
        """构建系统提示词。

        Args:
            chat_stream: 当前聊天流
            extra_vars: 额外模板变量
            tool_schemas: 第三方工具 schema 列表，会生成动作类型描述注入提示词

        Returns:
            str: 完整的系统提示词
        """
        pm = get_prompt_manager()
        tmpl = pm.get_template("kfc_system_prompt")
        if not tmpl:
            return ""

        # 复制模板以避免污染全局状态
        tmpl = tmpl.clone()

        # 设置动态变量
        # 只用 chat_stream 属性补充模板中未设置的字段
        # nickname/alias_names 等已在 modules.py 的 register_kfc_prompts() 中
        # 从 core_config.personality 正确设置，此处不再覆盖
        tmpl.set("platform", chat_stream.platform or "unknown")
        tmpl.set("chat_type", str(chat_stream.chat_type or "unknown"))
        tmpl.set("bot_id", chat_stream.bot_id or "")
        tmpl.set(
            "current_time",
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # 活动流格式提示
        tmpl.set("mental_log_hint", build_mental_log_hint(self._log_format))

        # 场景引导
        theme_guide = self._get_theme_guide(chat_stream)
        tmpl.set("theme_guide", theme_guide)

        # 第三方工具 → 动态动作类型描述
        extra_action_text = build_extra_action_types(tool_schemas or [])
        tmpl.set("extra_action_types", extra_action_text)

        # 额外变量
        if extra_vars:
            for key, value in extra_vars.items():
                tmpl.set(key, value)

        return tmpl.build()

    @staticmethod
    def _get_theme_guide(chat_stream: ChatStream) -> str:
        """根据聊天类型返回场景引导文本。"""
        chat_type = str(chat_stream.chat_type or "").lower()

        if chat_type == "private":
            return (
                "你当前处于私聊环境。你可以更亲近地和对方交流，"
                "关注对方情绪并提供更直接、细腻的回应。"
            )
        if chat_type == "group":
            return (
                "你当前处于群聊环境。注意多人对话的上下文，"
                "确认对方确实在和你说话后再做出回应。"
                "群聊中不要总是抢话，保持自然。"
            )
        return ""
