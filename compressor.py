"""KFC 近期记忆压缩模块。

异步生成"近期记忆摘要"（history_summary），覆盖最近 N 天的对话。
使用与正常对话相同的主聊天模型，以完整人设 + 第一人称书写，直接替换旧摘要。

压缩时机由 KFCChatter 在每轮对话结束后检查并触发（见 chatter.py）。
"""

from __future__ import annotations

import datetime
import time
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.stream_api import get_stream_messages
from src.kernel.llm import LLMPayload, LLMRequest, ROLE, Text

if TYPE_CHECKING:
    from .config import KFCConfig
    from .session import KFCSession
    from .prompts.builder import KFCPromptBuilder

logger = get_logger("kfc_compressor")


async def compress_history(
    session: "KFCSession",
    prompt_builder: "KFCPromptBuilder",
    config: "KFCConfig",
    model_set: Any,
    chat_stream: Any,
) -> None:
    """对最近 N 天的对话生成近期记忆摘要，更新 session.history_summary。

    该函数为"替换式"：每次调用都基于原始消息重新生成摘要，不累积旧摘要。
    应在 task_manager 中异步调用，不阻塞主对话流程。

    Args:
        session: 当前用户的 KFCSession（会被直接修改）
        prompt_builder: KFC prompt 构建器（用于获取 system_prompt）
        config: KFC 配置
        model_set: LLM 模型集合（与正常对话一致）
        chat_stream: 当前聊天流（用于 system_prompt 构建）
    """
    # 立即标记压缩时间，防止异步并发重复触发
    session.last_compress_at = time.time()

    stream_id = session.stream_id
    days = config.prompt.compress_days_window
    since_ts = time.time() - days * 86400

    # ── 1. 从 DB 读取时间窗口内的历史消息 ──
    # 私聊流消息量有限，以大上限一次性拉取后按时间过滤，无需分页。
    _FETCH_LIMIT = 10000
    try:
        all_msgs = await get_stream_messages(stream_id=stream_id, limit=_FETCH_LIMIT)
    except Exception as exc:
        logger.warning(f"[KFC] 压缩：读取 DB 消息失败：{exc}")
        return

    # 过滤到时间窗口内
    window_msgs = [m for m in all_msgs if _msg_time(m) >= since_ts]

    if not window_msgs:
        logger.debug(f"[KFC] 压缩：流 {stream_id} 最近 {days} 天无消息，跳过")
        return

    # ── 2. 格式化消息文本（同 fused_narrative 格式）──
    bot_id = str(getattr(chat_stream, "bot_id", "") or "")
    formatted_lines: list[str] = []

    for msg in window_msgs:
        ts = _msg_time(msg)
        try:
            time_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        except (OSError, ValueError, OverflowError):
            continue

        text = (getattr(msg, "processed_plain_text", "") or "").strip()
        if not text:
            continue

        sender_id = str(getattr(msg, "sender_id", ""))
        message_id = str(getattr(msg, "message_id", "") or "")
        sender = getattr(msg, "sender_name", "用户")

        is_bot = bool(
            (bot_id and sender_id == bot_id)
            or message_id.startswith("action_kfc_reply")
        )
        if is_bot:
            formatted_lines.append(f"[{time_str}] 你回复：{text}")
        else:
            formatted_lines.append(f"[{time_str}] {sender}说：{text}")

    if not formatted_lines:
        logger.debug(f"[KFC] 压缩：流 {stream_id} 格式化后无有效内容，跳过")
        return

    # 同时从 mental_log 中加入内心活动（BOT_PLANNING 的 thought）
    _merge_mental_log(formatted_lines, session, since_ts)
    formatted_lines.sort()  # 按 "[时间戳]" 字符串排序（格式一致时等价于按时间排序）

    history_text = "\n".join(formatted_lines)

    # ── 3. 构建 LLM 请求 ──
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    user_name = (
        getattr(chat_stream, "partner_name", None)
        or getattr(chat_stream, "group_name", None)
        or "对方"
    )

    try:
        system_prompt = await prompt_builder.build_system_prompt(chat_stream)
    except Exception as exc:
        logger.warning(f"[KFC] 压缩：构建 system_prompt 失败：{exc}")
        return

    compress_instruction = (
        f"当前时间：{now_str}\n\n"
        f"以下是你与{user_name}之间最近 {days:.0f} 天的对话记录。\n\n"
        f"【近期对话记录】\n{history_text}\n\n"
        "请你以第一人称（'我'）写一段简洁的近期记忆摘要，要求：\n"
        "1. 覆盖上述对话中你认为值得记住的内容\n"
        "2. 保留时间感：无需精确到秒，但重要节点要有相对时间描述（如'昨天深夜'、'今天下午'、'前天'）\n"
        "3. 保留关键情感节点、对方的重要信息、承诺与期待\n"
        "4. 字数控制在 800-1200 字，用感性而真实的自然语言\n"
        "5. 不要包含任何 JSON 或结构化标记，直接输出摘要正文"
    )

    # 注入 actor_reminder（如有）
    from src.core.prompt import get_system_reminder_store
    actor_reminder = get_system_reminder_store().get("actor")

    llm_request = LLMRequest(model_set, f"kfc_compress_{stream_id}")
    llm_request.add_payload(LLMPayload(ROLE.SYSTEM, Text(system_prompt)))
    if actor_reminder:
        llm_request.add_payload(LLMPayload(ROLE.SYSTEM, Text(actor_reminder)))
    llm_request.add_payload(LLMPayload(ROLE.USER, Text(compress_instruction)))

    # ── 4. 调用 LLM（非流式收集全文）──
    try:
        llm_response = await llm_request.send()
        summary = (await llm_response or "").strip()
    except Exception as exc:
        logger.warning(f"[KFC] 压缩：LLM 调用失败：{exc}")
        return

    if not summary:
        logger.warning(f"[KFC] 压缩：LLM 返回空摘要，跳过")
        return

    # ── 5. 更新 session（直接替换）──
    session.history_summary = summary
    session.last_compress_at = time.time()
    session.compress_round_count = 0
    logger.info(
        f"[KFC] 近期记忆压缩完成：流 {stream_id}，"
        f"覆盖 {len(formatted_lines)} 条消息，"
        f"摘要 {len(summary)} 字"
    )


def should_compress(session: "KFCSession", config: "KFCConfig") -> bool:
    """判断是否应触发压缩。

    Args:
        session: 当前 KFCSession
        config: KFC 配置

    Returns:
        bool: 是否应触发压缩
    """
    every_n = config.prompt.compress_every_n_rounds
    if every_n <= 0:
        return False

    if session.compress_round_count < every_n:
        return False

    # 最短间隔检查
    min_interval = config.prompt.min_compress_interval_minutes * 60
    if time.time() - session.last_compress_at < min_interval:
        return False

    return True


# ── 私有辅助函数 ──────────────────────────────────────────

def _msg_time(msg: Any) -> float:
    t = getattr(msg, "time", None)
    return float(t) if isinstance(t, (int, float)) else 0.0


def _merge_mental_log(
    lines: list[str],
    session: "KFCSession",
    since_ts: float,
) -> None:
    """将 mental_log 中的 BOT_PLANNING thought 合并入 lines。"""
    from .models import KFCEventType

    mental_log = getattr(session, "mental_log", None)
    if not mental_log:
        return

    for entry in mental_log.entries:
        if entry.timestamp < since_ts:
            continue
        if entry.event_type != KFCEventType.BOT_PLANNING or not entry.thought:
            continue
        try:
            time_str = datetime.datetime.fromtimestamp(entry.timestamp).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except (OSError, ValueError, OverflowError):
            continue
        lines.append(f"[{time_str}] （你的内心：{entry.thought}）")
