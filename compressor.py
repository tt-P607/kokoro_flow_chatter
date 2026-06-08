"""KFC 近期记忆压缩模块。

异步生成"近期记忆摘要"（history_summary），覆盖最近 N 天的对话。
使用 actor 模型任务（config.general.model_task），以完整人设 + 第一人称书写，直接替换旧摘要。

压缩时机由 KFCChatter 在每轮对话结束后检查并触发（见 chatter.py）。
"""

from __future__ import annotations

import datetime
import time
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.stream_api import get_stream_messages
from src.app.plugin_system.types import LLMPayload, ROLE, Text

from .models import KFCEventType

if TYPE_CHECKING:
    from .config import KFCConfig
    from .session import KFCSession
    from .prompts.builder import KFCPromptBuilder

from src.app.plugin_system.api.llm_api import get_model_set_by_task, create_llm_request

logger = get_logger("kfc_compressor")


async def compress_history(
    session: "KFCSession",
    prompt_builder: "KFCPromptBuilder",
    config: "KFCConfig",
    chat_stream: Any,
) -> None:
    """对最近 N 天的对话生成近期记忆摘要，更新 session.history_summary。

    该函数为"替换式"：每次调用都基于原始消息重新生成摘要，不累积旧摘要。
    应在 task_manager 中异步调用，不阻塞主对话流程。
    使用 config.general.model_task 对应的 actor 模型，避免继承对话 model_set 的 max_tokens 限制。

    Args:
        session: 当前用户的 KFCSession（会被直接修改）
        prompt_builder: KFC prompt 构建器（用于获取 system_prompt）
        config: KFC 配置
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

    # ── 2. 收集 (时间戳, 单行文本) 元组 ──
    # 同时容纳消息与内心活动，按时间戳统一排序，再按天分组渲染
    bot_id = chat_stream.bot_id or ""
    collected: list[tuple[float, str]] = []

    for msg in window_msgs:
        ts = _msg_time(msg)
        if ts <= 0:
            continue
        try:
            time_str = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        except (OSError, ValueError, OverflowError):
            continue

        text = (msg.processed_plain_text or "").strip()
        if not text:
            continue

        sender_id = msg.sender_id or ""
        message_id = msg.message_id or ""
        sender = msg.sender_name or "用户"

        is_bot = bool(
            (bot_id and sender_id == bot_id)
            or message_id.startswith("action_kfc_reply")
        )
        if is_bot:
            collected.append((ts, f"[{time_str}] 你回复：{text}"))
        else:
            collected.append((ts, f"[{time_str}] {sender}说：{text}"))

    # 同时从 mental_log 中加入内心活动（BOT_PLANNING 的 thought）
    _merge_mental_log(collected, session, since_ts)

    if not collected:
        logger.debug(f"[KFC] 压缩：流 {stream_id} 格式化后无有效内容，跳过")
        return

    collected.sort(key=lambda item: item[0])
    history_text = _render_by_day(collected)

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

    min_chars = config.prompt.compress_min_chars
    max_chars = config.prompt.compress_max_chars
    # 防御式校正：保证 max >= min 且二者均为正数
    if min_chars < 0:
        min_chars = 0
    if max_chars < min_chars:
        max_chars = min_chars

    compress_instruction = (
        f"当前时间：{now_str}\n\n"
        f"以下是你与{user_name}之间最近 {days:.0f} 天的对话记录，"
        "已按自然天分组：每个 `=== 日期（星期，相对时间）===` 标题之下，"
        "都是当天的消息与你当时的内心活动，行首 `[HH:MM:SS]` 是当天的时间。\n\n"
        f"【近期对话记录】\n{history_text}\n\n"
        "请你以第一人称（'我'）写一段近期记忆摘要（Memory Stream），要求：\n"
        "1. 【按重要性分配篇幅】：不要把笔墨平均分配给每天。对于关键情感节点、重要的约定、影响关系的事件、对方吐露的心声，应分配较大篇幅详细记录（甚至保留核心原话）；对于日常寒暄、琐碎水文、流水账，一笔带过或直接忽略。\n"
        "2. 【保留时间感】：直接使用'今天下午'、'昨天深夜'、'前天'、'三天前'这类相对描述（对应分组标题里的相对时间），不要写出具体数字日期。\n"
        "3. 【主观真实感】：这是你脑海中流淌的真实记忆，用感性且符合你人设的自然语言叙述，体现你对这些事的感受与想法。\n"
        f"4. 【字数限制】：总字数控制在 {min_chars}-{max_chars} 字。\n"
        "5. 【纯文本输出】：不要包含任何 JSON 或结构化标记，直接输出记忆正文。"
    )

    # 注入 actor_reminder（如有）
    from src.app.plugin_system.api.prompt_api import get_system_reminder
    actor_reminder = get_system_reminder("actor")

    # 直接使用 actor 模型任务，避免继承对话 model_set 的 max_tokens 限制
    model_set = get_model_set_by_task(config.general.model_task)
    llm_request = create_llm_request(model_set, f"kfc_compress_{stream_id}")
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
        logger.warning("[KFC] 压缩：LLM 返回空摘要，跳过")
        return

    # ── 5. 更新 session（直接替换）──
    session.history_summary = summary
    session.last_compress_at = time.time()
    session.compress_round_count = 0
    logger.info(
        f"[KFC] 近期记忆压缩完成：流 {stream_id}，"
        f"覆盖 {len(collected)} 条消息，"
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
    """从消息对象中提取时间戳，不存在或类型错误时返回 0.0。"""
    # msg 可能来自 DB 层原始记录不一定是 Message 实例，这里保留 getattr。
    t = getattr(msg, "time", None)
    return float(t) if isinstance(t, (int, float)) else 0.0


def _merge_mental_log(
    collected: list[tuple[float, str]],
    session: "KFCSession",
    since_ts: float,
) -> None:
    """将 mental_log 中的 BOT_PLANNING thought 以 (ts, line) 元组合并入列表。"""
    mental_log = session.mental_log
    if not mental_log:
        return

    for entry in mental_log.entries:
        if entry.timestamp < since_ts:
            continue
        if entry.event_type != KFCEventType.BOT_PLANNING or not entry.thought:
            continue
        try:
            time_str = datetime.datetime.fromtimestamp(entry.timestamp).strftime(
                "%H:%M:%S"
            )
        except (OSError, ValueError, OverflowError):
            continue
        collected.append(
            (entry.timestamp, f"[{time_str}] （你的内心：{entry.thought}）")
        )


# 中文星期，下标从 weekday() 直接取
_WEEKDAY_ZH: tuple[str, ...] = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def _render_by_day(collected: list[tuple[float, str]]) -> str:
    """将 (时间戳, 行) 序列按"自然天"分组渲染。

    每天一个小标题：``=== YYYY-MM-DD（周X，N 天前 / 今天 / 昨天）===``，
    段内每行只保留 ``[HH:MM:SS]``。
    LLM 据此即可分辨同一天内与跨天的时间关系。
    """
    if not collected:
        return ""

    today = datetime.date.today()
    grouped: dict[datetime.date, list[str]] = {}
    order: list[datetime.date] = []

    for ts, line in collected:
        try:
            day = datetime.datetime.fromtimestamp(ts).date()
        except (OSError, ValueError, OverflowError):
            continue
        if day not in grouped:
            grouped[day] = []
            order.append(day)
        grouped[day].append(line)

    sections: list[str] = []
    for day in sorted(order):
        delta_days = (today - day).days
        if delta_days == 0:
            relative = "今天"
        elif delta_days == 1:
            relative = "昨天"
        elif delta_days == 2:
            relative = "前天"
        elif delta_days > 0:
            relative = f"{delta_days} 天前"
        else:
            # 极端情况：消息时间晚于今天（系统时钟漂移），降级为日期描述
            relative = f"{-delta_days} 天后"
        weekday = _WEEKDAY_ZH[day.weekday()]
        header = f"=== {day.isoformat()}（{weekday}，{relative}）==="
        body = "\n".join(grouped[day])
        sections.append(f"{header}\n{body}")

    return "\n\n".join(sections)
