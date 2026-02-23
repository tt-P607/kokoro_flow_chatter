"""KFC 提示词模块函数。

提供基于 PromptManager 的模板注入和上下文构建辅助函数。
"""

from __future__ import annotations

import datetime
from typing import Any

from src.core.config import get_core_config
from src.core.prompt import get_prompt_manager, optional, wrap, min_len

from .templates import (
    KFC_SYSTEM_PROMPT,
    KFC_PROACTIVE_PROMPT,
    KFC_CONTINUOUS_THINKING_PROMPT,
)


def register_kfc_prompts() -> None:
    """注册 KFC 所有提示词模板到 PromptManager。

    在 plugin.on_plugin_loaded() 中调用一次即可。
    """
    config = get_core_config()
    personality = config.personality

    pm = get_prompt_manager()

    # 主系统提示词
    pm.get_or_create(
        name="kfc_system_prompt",
        template=KFC_SYSTEM_PROMPT,
        policies={
            "nickname": optional(personality.nickname),
            "alias_names": optional("、".join(personality.alias_names)),
            "personality_core": optional(personality.personality_core),
            "personality_side": optional(personality.personality_side),
            "identity": optional(personality.identity),
            "background_story": optional(personality.background_story)
            .then(min_len(10))
            .then(
                wrap(
                    "# 背景故事\n",
                    "\n- （以上为背景知识，请理解并作为行动依据，但不要在对话中直接复述。）",
                )
            ),
            "reply_style": optional(personality.reply_style),
            "safety_guidelines": optional(
                "\n".join(personality.safety_guidelines)
            ),
            "current_time": optional(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ),
            "extra_action_types": optional(""),
        },
    )

    # 主动发起提示词
    pm.get_or_create(
        name="kfc_proactive_prompt",
        template=KFC_PROACTIVE_PROMPT,
        policies={
            "silence_duration": optional("未知"),
            "recent_activity": optional("（无近期活动记录）"),
        },
    )

    # 连续思考提示词
    pm.get_or_create(
        name="kfc_continuous_thinking_prompt",
        template=KFC_CONTINUOUS_THINKING_PROMPT,
        policies={
            "elapsed_seconds": optional("0"),
            "progress": optional("0"),
            "expected_reaction": optional("无特定期望"),
        },
    )


def build_mental_log_hint(log_format: str) -> str:
    """构建活动流格式提示。"""
    if log_format == "table":
        return (
            "你的活动流会以表格形式呈现在消息中，"
            "帮助你回顾之前的互动和内心活动。"
        )
    return (
        "你的活动流会以线性叙事的形式呈现在消息中，"
        "帮助你回顾之前的互动和内心活动。"
    )


def build_proactive_context(
    silence_minutes: float,
    recent_activity: str,
) -> str:
    """构建主动发起上下文。"""
    pm = get_prompt_manager()
    tmpl = pm.get_template("kfc_proactive_prompt")
    if not tmpl:
        return f"已沉默 {silence_minutes:.0f} 分钟"

    return (
        tmpl.clone()
        .set("silence_duration", f"{silence_minutes:.0f}")
        .set("recent_activity", recent_activity or "（无近期活动记录）")
        .build()
    )


def build_continuous_thinking_context(
    elapsed_seconds: float,
    progress: float,
    expected_reaction: str,
) -> str:
    """构建连续思考上下文。"""
    pm = get_prompt_manager()
    tmpl = pm.get_template("kfc_continuous_thinking_prompt")
    if not tmpl:
        return f"已等待 {elapsed_seconds:.0f} 秒"

    return (
        tmpl.clone()
        .set("elapsed_seconds", f"{elapsed_seconds:.0f}")
        .set("progress", f"{progress:.0%}")
        .set("expected_reaction", expected_reaction or "无特定期望")
        .build()
    )
