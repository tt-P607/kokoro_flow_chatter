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
    KFC_TIMEOUT_PROMPT,
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
        },
    )

    # 主动发起提示词
    pm.get_or_create(
        name="kfc_proactive_prompt",
        template=KFC_PROACTIVE_PROMPT,
        policies={
            "current_time": optional(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            ),
            "silence_duration": optional("未知"),
            "recent_activity": optional("（无近期活动记录）"),
        },
    )

    # 连续思考提示词
    pm.get_or_create(
        name="kfc_continuous_thinking_prompt",
        template=KFC_CONTINUOUS_THINKING_PROMPT,
        policies={
            "last_bot_message": optional("（消息内容不可用）"),
            "expected_reaction": optional("无特定期望"),
            "elapsed_minutes": optional("0"),
            "progress": optional("0%"),
        },
    )


def build_mental_log_hint() -> str:
    """构建活动流格式提示。"""
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

    # 格式化沉默持续时间为可读文本
    if silence_minutes >= 60:
        hours = silence_minutes / 60
        silence_str = f"{hours:.1f} 小时"
    else:
        silence_str = f"{silence_minutes:.0f} 分钟"

    return (
        tmpl.clone()
        .set("current_time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        .set("silence_duration", silence_str)
        .set("recent_activity", recent_activity or "（无近期活动记录）")
        .build()
    )


def build_continuous_thinking_context(
    elapsed_seconds: float,
    progress: float,
    expected_reaction: str,
    last_bot_message: str = "",
) -> str:
    """构建连续思考上下文。

    Args:
        elapsed_seconds: 已等待秒数
        progress: 等待进度 (0.0~1.0)
        expected_reaction: 预期对方的反应
        last_bot_message: 最后一条 Bot 发送的消息
    """
    pm = get_prompt_manager()
    tmpl = pm.get_template("kfc_continuous_thinking_prompt")
    if not tmpl:
        return f"已等待 {elapsed_seconds:.0f} 秒"

    return (
        tmpl.clone()
        .set("last_bot_message", last_bot_message or "（消息内容不可用）")
        .set("expected_reaction", expected_reaction or "无特定期望")
        .set("elapsed_minutes", f"{elapsed_seconds / 60:.1f}")
        .set("progress", f"{progress:.0%}")
        .build()
    )


def build_timeout_context(
    elapsed_seconds: float,
    expected_reaction: str,
    consecutive_timeouts: int,
    last_bot_message: str = "",
    pending_thoughts: list[str] | None = None,
) -> str:
    """构建等待超时决策上下文。

    Args:
        elapsed_seconds: 已等待秒数
        expected_reaction: 预期对方的反应
        consecutive_timeouts: 连续超时次数（含本次）
        last_bot_message: 最后一条 Bot 发送的消息
        pending_thoughts: 等待期间产生的想法列表
    """
    elapsed_minutes = elapsed_seconds / 60

    # 根据追问次数生成递进式警告
    followup_count = max(0, consecutive_timeouts - 1)
    if followup_count >= 2:
        followup_warning = (
            f"\n⚠️ **强烈建议**: 你已经连续追问了 {followup_count} 次，对方仍未回复。"
            "**极度推荐选择 do_nothing 并将 max_wait_seconds 设为 0**。"
            "对方可能在忙或需要空间，给彼此一些空间会更好。"
        )
    elif followup_count == 1:
        followup_warning = (
            "\n📝 温馨提醒：这是你第 2 次等待回复（已追问 1 次）。"
            "可以再试着追问一次，但如果对方还是没回复，"
            "**强烈建议**之后选择 do_nothing 结束等待。"
        )
    else:
        followup_warning = (
            "\n💭 这是第一次等待超时。如果觉得话题还没结束，"
            "可以适当追问一下，但也要考虑对方可能在忙。"
        )

    # 等待期间的想法
    if pending_thoughts:
        thoughts_text = "、".join(pending_thoughts)
        pending_block = f"\n你等待期间的想法：{thoughts_text}"
    else:
        pending_block = ""

    from .templates import KFC_TIMEOUT_PROMPT

    return KFC_TIMEOUT_PROMPT.format(
        elapsed_seconds=elapsed_seconds,
        elapsed_minutes=elapsed_minutes,
        expected_reaction=expected_reaction or "对方能回复点什么",
        last_bot_message=last_bot_message or "（消息内容不可用）",
        followup_warning=followup_warning,
        pending_thoughts_block=pending_block,
    )
