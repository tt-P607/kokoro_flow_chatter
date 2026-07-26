"""KFC 提示词模板注册与动态上下文构建。

``register_kfc_prompts()`` 在插件加载时把模板注册进 PromptManager；
其余函数按运行时状态渲染主动发起、超时决策等动态提示词。
"""

from __future__ import annotations

import datetime

from src.app.plugin_system.api.prompt_api import get_or_create, get_template
from src.app.plugin_system.types import LLMPayload, ROLE, Text
from src.core.config import get_core_config  # 公开 API 尚未提供人格配置读取
from src.core.prompt import min_len, optional, wrap  # 无状态的模板策略工具

from .templates import (
    KFC_PROACTIVE_DECISION_TOOL_CALLING,
    KFC_PROACTIVE_PROMPT,
    KFC_REPLY_MODE_TOOL_CALLING,
    KFC_SYSTEM_PROMPT,
    KFC_TIMEOUT_PROMPT,
)

SYSTEM_PROMPT_NAME = "kfc_system_prompt"
PROACTIVE_PROMPT_NAME = "kfc_proactive_prompt"

_BACKGROUND_STORY_MIN_LEN = 10
"""背景故事短于此长度时视为未配置，不注入模板。"""

_BACKGROUND_STORY_SUFFIX = (
    "\n- （以上为背景知识，请理解并作为行动依据，但不要在对话中直接复述。）"
)


def register_kfc_prompts() -> None:
    """把 KFC 全部提示词模板注册到 PromptManager。

    在 ``plugin.on_plugin_loaded()`` 中调用一次即可；重复调用因
    ``get_or_create`` 语义而幂等。
    """
    personality = get_core_config().personality

    get_or_create(
        name=SYSTEM_PROMPT_NAME,
        template=KFC_SYSTEM_PROMPT,
        policies={
            "nickname": optional(personality.nickname),
            "alias_names": optional("、".join(personality.alias_names)),
            "personality_core": optional(personality.personality_core),
            "personality_side": optional(personality.personality_side),
            "identity": optional(personality.identity),
            "background_story": optional(personality.background_story)
            .then(min_len(_BACKGROUND_STORY_MIN_LEN))
            .then(wrap("# 背景故事\n", _BACKGROUND_STORY_SUFFIX)),
            "reply_style": optional(personality.reply_style),
            "safety_guidelines": optional("\n".join(personality.safety_guidelines)),
            "negative_behaviors": optional("\n".join(personality.negative_behaviors)),
            "custom_decision_prompt": optional(""),
            "scene_state_info": optional(""),
            "scheduled_proactive_info": optional(""),
            # 实际值由初始上下文规划注入，此处提供 tool calling 兜底
            "reply_mode_instruction": optional(KFC_REPLY_MODE_TOOL_CALLING),
        },
    )

    get_or_create(
        name=PROACTIVE_PROMPT_NAME,
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
    """构建活动流格式说明，注入系统提示词。"""
    return (
        "你的活动流会以线性叙事的形式呈现在消息中，"
        "帮助你回顾之前的互动和内心活动。"
    )


async def build_proactive_context(
    silence_minutes: float,
    recent_activity: str,
    scheduled_reason: str = "",
) -> str:
    """构建主动发起的触发上下文。

    Args:
        silence_minutes: 距上次互动的沉默分钟数。
        recent_activity: 近期互动摘要。
        scheduled_reason: 模型此前预约时给出的理由，非空时置于开头。

    Returns:
        str: 渲染后的主动发起提示词；模板缺失时返回简短兜底文本。
    """
    template_base = get_template(PROACTIVE_PROMPT_NAME)
    if not template_base:
        return f"已沉默 {silence_minutes:.0f} 分钟"

    if silence_minutes >= 60:
        silence_text = f"{silence_minutes / 60:.1f} 小时"
    else:
        silence_text = f"{silence_minutes:.0f} 分钟"

    rendered = await (
        template_base.clone()
        .set("current_time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        .set("silence_duration", silence_text)
        .set("recent_activity", recent_activity or "（无近期活动记录）")
        .set("proactive_decision_instruction", KFC_PROACTIVE_DECISION_TOOL_CALLING)
        .build()
    )

    if scheduled_reason:
        return (
            "【你在上次对话结束时为这次主动发起做了预约，"
            f"预约理由：{scheduled_reason}】\n\n{rendered}"
        )
    return rendered


def build_timeout_payload(
    elapsed_seconds: float,
    expected_reaction: str,
    consecutive_timeouts: int,
    last_bot_message: str = "",
    max_consecutive_timeouts: int = 3,
) -> LLMPayload:
    """构建等待超时的决策 payload。

    按"首次超时 / 中间超时 / 最后一次超时"三种情境给出不同的情绪描述与
    可选动作，引导模型做出符合当下心境的决定，而非机械追问。

    Args:
        elapsed_seconds: 已等待秒数。
        expected_reaction: 此前预期的对方反应。
        consecutive_timeouts: 连续超时次数（含本次）。
        last_bot_message: 最后一条 Bot 消息，用于唤起上下文。
        max_consecutive_timeouts: 配置的连续超时上限。

    Returns:
        LLMPayload: USER 角色的超时决策 payload。
    """
    _ = expected_reaction
    elapsed_minutes = elapsed_seconds / 60
    is_first = consecutive_timeouts == 1
    is_last = consecutive_timeouts >= max_consecutive_timeouts
    message_snippet = last_bot_message or "（消息内容不可用）"

    if is_first:
        situation = (
            f"你发出消息已经过去 {elapsed_minutes:.0f} 分钟了，对方还没有回应。\n"
            f"**你发的最后一条消息**：「{message_snippet}」"
        )
    else:
        situation = (
            f"你已经主动说了 {consecutive_timeouts} 次，对方一直没有回应。\n"
            f"距上次发消息已有 {elapsed_minutes:.0f} 分钟。\n"
            f"**你最后说的**：「{message_snippet}」"
        )

    if is_last:
        guidance = "你已经等了很久，对方始终没有出现。\n这种时候，你会怎么做？"
        instructions = (
            "本次等待到此为止，**不得**再设置新的等待"
            "（`max_wait_seconds` 必须为 0）。"
        )
    elif is_first:
        guidance = (
            "你想想：有没有什么没说完的话，或者忽然想到什么想跟对方说的？\n"
            "如果有，发出去就好；如果脑子里没什么，继续等一等也无妨。"
        )
        instructions = (
            "可以调用 `action-kfc_reply(...)` 发送消息，"
            "或调用 `action-do_nothing(max_wait_seconds>0)` 继续等待，"
            "或调用 `action-do_nothing(max_wait_seconds=0)` 结束等待。"
        )
    else:
        guidance = (
            "对方一直没有回复。\n"
            "你有没有真的需要说的内容——还是只是想打破沉默？"
        )
        instructions = (
            "如果确实有话说，可以调用 `action-kfc_reply(...)` 发送消息；"
            "或调用 `action-do_nothing(max_wait_seconds=0)` 结束等待。"
        )

    timeout_text = KFC_TIMEOUT_PROMPT.format(
        timeout_situation=situation,
        timeout_guidance=guidance,
        decision_instructions=instructions,
    )
    return LLMPayload(ROLE.USER, Text(timeout_text))
