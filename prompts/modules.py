"""KFC 提示词模块函数。

提供基于 PromptManager 的模板注入和上下文构建辅助函数。
"""

from __future__ import annotations

import datetime

from src.core.config import get_core_config  # TODO: 待 prompt_api 暴露 get_bot_personality() 后迁移
from src.core.prompt import optional, wrap, min_len  # 纯工具函数，无状态副作用

from src.app.plugin_system.api.prompt_api import get_or_create as _pm_get_or_create
from src.app.plugin_system.api.prompt_api import get_template as _pm_get_template

from .templates import (
    KFC_SYSTEM_PROMPT,
    KFC_PROACTIVE_PROMPT,
    KFC_TIMEOUT_PROMPT,
    KFC_PROACTIVE_DECISION_TOOL_CALLING,
    KFC_REPLY_MODE_TOOL_CALLING,
)


def register_kfc_prompts() -> None:
    """注册 KFC 所有提示词模板到 PromptManager。

    在 plugin.on_plugin_loaded() 中调用一次即可。
    """
    config = get_core_config()
    personality = config.personality

    # 主系统提示词
    _pm_get_or_create(
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
            "negative_behaviors": optional(
                "\n".join(personality.negative_behaviors)
            ),
            "custom_decision_prompt": optional(""),
            "scene_state_info": optional(""),
            # reply_mode_instruction 由 _build_initial_context 动态注入，此处提供 tool calling 兜底
            "reply_mode_instruction": optional(KFC_REPLY_MODE_TOOL_CALLING),
            "current_time": optional(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ),
        },
    )

    # 主动发起提示词
    _pm_get_or_create(
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


def build_mental_log_hint() -> str:
    """构建活动流格式提示。"""
    return (
        "你的活动流会以线性叙事的形式呈现在消息中，"
        "帮助你回顾之前的互动和内心活动。"
    )


async def build_proactive_context(
    silence_minutes: float,
    recent_activity: str,
    scheduled_reason: str = "",
) -> str:
    """构建主动发起上下文。"""
    tmpl_base = _pm_get_template("kfc_proactive_prompt")
    if not tmpl_base:
        return f"已沉默 {silence_minutes:.0f} 分钟"

    # 格式化沉默持续时间为可读文本
    if silence_minutes >= 60:
        hours = silence_minutes / 60
        silence_str = f"{hours:.1f} 小时"
    else:
        silence_str = f"{silence_minutes:.0f} 分钟"

    decision_instruction = KFC_PROACTIVE_DECISION_TOOL_CALLING

    result = await (
        tmpl_base.clone()
        .set("current_time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        .set("silence_duration", silence_str)
        .set("recent_activity", recent_activity or "（无近期活动记录）")
        .set("proactive_decision_instruction", decision_instruction)
        .build()
    )

    if scheduled_reason:
        result = f"【你在上次对话结束时为这次主动发起做了预约，预约理由：{scheduled_reason}】\n\n" + result

    return result


def build_timeout_context(
    elapsed_seconds: float,
    expected_reaction: str,
    consecutive_timeouts: int,
    last_bot_message: str = "",
    max_consecutive_timeouts: int = 3,
) -> str:
    """构建等待超时决策上下文。

    Args:
        elapsed_seconds: 已等待秒数
        expected_reaction: 预期对方的反应
        consecutive_timeouts: 连续超时次数（含本次）
        last_bot_message: 最后一条 Bot 发送的消息
        max_consecutive_timeouts: 配置的连续超时上限
    """
    elapsed_minutes = elapsed_seconds / 60
    is_first = consecutive_timeouts == 1
    is_last = consecutive_timeouts >= max_consecutive_timeouts
    msg_snippet = last_bot_message or "（消息内容不可用）"

    # ── 情境描述 ──
    if is_first:
        timeout_situation = (
            f"你发出消息已经过去 {elapsed_minutes:.0f} 分钟了，对方还没有回应。\n"
            f"**你发的最后一条消息**：「{msg_snippet}」"
        )
    else:
        timeout_situation = (
            f"你已经主动说了 {consecutive_timeouts} 次，对方一直没有回应。\n"
            f"距上次发消息已有 {elapsed_minutes:.0f} 分钟。\n"
            f"**你最后说的**：「{msg_snippet}」"
        )

    # ── 引导语 ──
    if is_last:
        timeout_guidance = (
            "你已经等了很久，对方始终没有出现。\n"
            "这种时候，你会怎么做？"
        )
    elif is_first:
        timeout_guidance = (
            "你想想：有没有什么没说完的话，或者忽然想到什么想跟对方说的？\n"
            "如果有，发出去就好；如果脑子里没什么，继续等一等也无妨。"
        )
    else:
        timeout_guidance = (
            "对方一直没有回复。\n"
            "你有没有真的需要说的内容——还是只是想打破沉默？"
        )

    # ── 操作指令 ──
    if is_last:
        decision_instructions = (
            "本次等待到此为止，**不得**再设置新的等待（`max_wait_seconds` 必须为 0）。"
        )
    elif is_first:
        decision_instructions = (
            "可以调用 `kfc_reply(...)` 发送消息，"
            "或调用 `do_nothing(max_wait_seconds>0)` 继续等待，"
            "或调用 `do_nothing(max_wait_seconds=0)` 结束等待。"
        )
    else:
        decision_instructions = (
            "如果确实有话说，可以调用 `kfc_reply(...)` 发送消息；"
            "或调用 `do_nothing(max_wait_seconds=0)` 结束等待。"
        )

    return KFC_TIMEOUT_PROMPT.format(
        timeout_situation=timeout_situation,
        timeout_guidance=timeout_guidance,
        decision_instructions=decision_instructions,
    )
