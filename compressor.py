"""KFC 近期记忆压缩。

把最近 N 天的对话与内心活动压缩成一段第一人称的记忆摘要，替换式更新
``session.history_summary``。摘要以完整人设书写，注入后续每轮上下文，
使长期对话不因上下文窗口限制而失忆。

压缩是替换而非累积——每次都基于原始消息重新生成，避免摘要在多次
迭代中逐渐失真。调度与去重由 ``services.summary_service`` 负责。
"""

from __future__ import annotations

import datetime
import json
import time
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.llm_api import create_llm_request, get_model_set_by_task
from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.prompt_api import get_system_reminder
from src.app.plugin_system.api.stream_api import get_stream_messages
from src.app.plugin_system.types import LLMPayload, ROLE, Text

from .context.renderer import build_system_prompt
from .models import KFCEventType

if TYPE_CHECKING:
    from src.app.plugin_system.types import ChatStream, Message

    from .config import KFCConfig
    from .mental_log import MentalLog
    from .session import KFCSession, KFCSessionStore

logger = get_logger("kfc_compressor")

_FETCH_LIMIT = 10000
"""单次拉取的消息上限。私聊消息量有限，一次取够后按时间过滤即可。"""

_MAX_RETRIES = 3
"""LLM 调用与 JSON 解析的最大重试次数。"""

_SECONDS_PER_DAY = 86400

_WEEKDAY_ZH: tuple[str, ...] = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
"""中文星期，下标直接取自 ``date.weekday()``。"""

_ACTOR_REMINDER_BUCKET = "actor"


def should_compress(session: KFCSession, config: KFCConfig) -> bool:
    """判断会话是否满足周期性压缩条件。

    Args:
        session: 当前会话。
        config: KFC 配置。

    Returns:
        bool: 是否应触发压缩。
    """
    every_n_rounds = config.prompt.compress_every_n_rounds
    if every_n_rounds <= 0:
        return False
    if session.compress_round_count < every_n_rounds:
        return False

    min_interval = config.prompt.min_compress_interval_minutes * 60
    return time.time() - session.last_compress_at >= min_interval


async def compress_history(
    session: KFCSession,
    config: KFCConfig,
    chat_stream: ChatStream,
    session_store: KFCSessionStore | None = None,
) -> None:
    """生成近期记忆摘要并更新会话。

    应在后台任务中调用，不阻塞对话主流程。使用独立的压缩模型任务，
    避免继承对话模型的 ``max_tokens`` 限制。

    Args:
        session: 目标会话，会被就地修改。
        config: KFC 配置。
        chat_stream: 当前聊天流，用于构建系统提示词。
        session_store: 会话存储；传入时摘要更新后立即持久化。
    """
    # 先占位，防止并发调度重复触发
    session.last_compress_at = time.time()

    stream_id = session.stream_id
    days = config.prompt.compress_days_window
    since_ts = time.time() - days * _SECONDS_PER_DAY

    collected = await _collect_timeline(stream_id, chat_stream, session, since_ts)
    if not collected:
        logger.debug(f"压缩：流 {stream_id[:8]} 窗口内无有效内容，跳过")
        return

    collected.sort(key=lambda item: item[0])
    history_text = _render_by_day(collected)

    try:
        system_prompt = await build_system_prompt(chat_stream)
    except Exception as error:
        logger.warning(f"压缩：构建系统提示词失败：{error}")
        return

    instruction = _build_compress_instruction(chat_stream, config, days, history_text)
    summary = await _request_summary(stream_id, config, system_prompt, instruction)
    if not summary:
        return

    session.history_summary = summary
    session.last_compress_at = time.time()
    session.compress_round_count = 0
    logger.info(
        f"近期记忆压缩完成：流 {stream_id[:8]}，"
        f"覆盖 {len(collected)} 条记录，摘要 {len(summary)} 字"
    )

    if session_store is not None:
        try:
            async with session_store.lock(stream_id):
                await session_store.save(session)
        except Exception as error:
            logger.warning(f"近期记忆摘要持久化失败：{error}")


# ── 时间线收集 ────────────────────────────────────────────


def _message_timestamp(message: Message) -> float:
    """提取消息时间戳；类型非法时返回 0。"""
    raw_time = message.time
    return float(raw_time) if isinstance(raw_time, (int, float)) else 0.0


def _format_clock(timestamp: float) -> str | None:
    """把时间戳格式化为 ``HH:MM:SS``；越界时返回 ``None``。"""
    try:
        return datetime.datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
    except (OSError, ValueError, OverflowError):
        return None


async def _collect_timeline(
    stream_id: str,
    chat_stream: ChatStream,
    session: KFCSession,
    since_ts: float,
) -> list[tuple[float, str]]:
    """收集窗口内的消息与内心活动，合并为 ``(时间戳, 单行文本)`` 序列。"""
    try:
        all_messages = await get_stream_messages(stream_id=stream_id, limit=_FETCH_LIMIT)
    except Exception as error:
        logger.warning(f"压缩：读取消息失败：{error}")
        return []

    bot_id = chat_stream.bot_id or ""
    collected: list[tuple[float, str]] = []

    for message in all_messages:
        timestamp = _message_timestamp(message)
        if timestamp < since_ts or timestamp <= 0:
            continue
        text = (message.processed_plain_text or "").strip()
        if not text:
            continue
        clock = _format_clock(timestamp)
        if clock is None:
            continue

        message_id = message.message_id or ""
        is_bot = bool(
            (bot_id and message.sender_id == bot_id)
            or message_id.startswith("action_kfc_reply")
        )
        if is_bot:
            collected.append((timestamp, f"[{clock}] 你回复：{text}"))
        else:
            sender = message.sender_name or "用户"
            collected.append((timestamp, f"[{clock}] {sender}说：{text}"))

    collected.extend(_collect_thoughts(session.mental_log, since_ts))
    return collected


def _collect_thoughts(
    mental_log: MentalLog | None,
    since_ts: float,
) -> list[tuple[float, str]]:
    """收集窗口内的内心独白。"""
    if mental_log is None:
        return []

    collected: list[tuple[float, str]] = []
    for entry in mental_log.entries:
        if entry.timestamp < since_ts:
            continue
        if entry.event_type != KFCEventType.BOT_PLANNING or not entry.thought:
            continue
        clock = _format_clock(entry.timestamp)
        if clock is None:
            continue
        collected.append((entry.timestamp, f"[{clock}] （你的内心：{entry.thought}）"))
    return collected


def _render_by_day(collected: list[tuple[float, str]]) -> str:
    """把时间线按自然天分组渲染。

    每天一个标题 ``=== YYYY-MM-DD（周X，相对时间）===``，段内每行只保留
    ``[HH:MM:SS]``。模型据此即可分辨同日内与跨日的时间关系。
    """
    today = datetime.date.today()
    grouped: dict[datetime.date, list[str]] = {}

    for timestamp, line in collected:
        try:
            day = datetime.datetime.fromtimestamp(timestamp).date()
        except (OSError, ValueError, OverflowError):
            continue
        grouped.setdefault(day, []).append(line)

    sections: list[str] = []
    for day in sorted(grouped):
        header = (
            f"=== {day.isoformat()}"
            f"（{_WEEKDAY_ZH[day.weekday()]}，{_describe_relative_day(today, day)}）==="
        )
        sections.append(header + "\n" + "\n".join(grouped[day]))
    return "\n\n".join(sections)


def _describe_relative_day(today: datetime.date, day: datetime.date) -> str:
    """把日期描述为相对今天的自然语言。"""
    delta_days = (today - day).days
    if delta_days == 0:
        return "今天"
    if delta_days == 1:
        return "昨天"
    if delta_days == 2:
        return "前天"
    if delta_days > 0:
        return f"{delta_days} 天前"
    # 系统时钟漂移导致消息时间晚于今天时的降级描述
    return f"{-delta_days} 天后"


# ── LLM 调用 ──────────────────────────────────────────────


def _build_compress_instruction(
    chat_stream: ChatStream,
    config: KFCConfig,
    days: float,
    history_text: str,
) -> str:
    """构建压缩指令。"""
    now_text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    user_name = chat_stream.stream_name or "对方"

    min_chars = max(0, config.prompt.compress_min_chars)
    max_chars = max(min_chars, config.prompt.compress_max_chars)

    return (
        f"当前时间：{now_text}\n\n"
        f"以下是你与{user_name}之间最近 {days:.0f} 天的对话记录，"
        "已按自然天分组：每个 `=== 日期（星期，相对时间）===` 标题之下，"
        "都是当天的消息与你当时的内心活动，行首 `[HH:MM:SS]` 是当天的时间。\n\n"
        f"【近期对话记录】\n{history_text}\n\n"
        "请你以第一人称（'我'）写一段近期记忆摘要（Memory Stream），要求：\n"
        "1. 【按重要性分配篇幅】：不要把笔墨平均分配给每天。对于关键情感节点、"
        "重要的约定、影响关系的事件、对方吐露的心声，应分配较大篇幅详细记录"
        "（甚至保留核心原话）；对于日常寒暄、琐碎水文、流水账，一笔带过或直接忽略。\n"
        "2. 【保留时间感】：直接使用'今天下午'、'昨天深夜'、'前天'、'三天前'这类"
        "相对描述（对应分组标题里的相对时间），不要写出具体数字日期。\n"
        "3. 【主观真实感】：这是你脑海中流淌的真实记忆，用感性且符合你人设的"
        "自然语言叙述，体现你对这些事的感受与想法。\n"
        f"4. 【字数限制】：总字数控制在 {min_chars}-{max_chars} 字。\n"
        "5. 【输出格式】：以 JSON 格式输出，只包含一个 `content` 字段，"
        "值为记忆正文字符串。示例：\n"
        '```json\n{"content": "你的记忆正文..."}\n```'
    )


async def _request_summary(
    stream_id: str,
    config: KFCConfig,
    system_prompt: str,
    instruction: str,
) -> str:
    """调用 LLM 生成摘要，带重试。

    Returns:
        str: 摘要正文；全部尝试失败时返回空串。
    """
    model_set = get_model_set_by_task(config.prompt.compress_model_task)
    llm_request = create_llm_request(model_set, f"kfc_compress_{stream_id}")
    llm_request.add_payload(LLMPayload(ROLE.SYSTEM, Text(system_prompt)))

    actor_reminder = get_system_reminder(_ACTOR_REMINDER_BUCKET)
    if actor_reminder:
        llm_request.add_payload(LLMPayload(ROLE.SYSTEM, Text(actor_reminder)))
    llm_request.add_payload(LLMPayload(ROLE.USER, Text(instruction)))

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            llm_response = await llm_request.send()
            raw_summary = (await llm_response or "").strip()
        except Exception as error:
            logger.warning(f"压缩：LLM 调用失败（{attempt}/{_MAX_RETRIES}）：{error}")
            continue

        if not raw_summary:
            logger.warning(f"压缩：LLM 返回空（{attempt}/{_MAX_RETRIES}）")
            continue

        summary = _extract_summary_content(raw_summary)
        if summary:
            return summary
        logger.warning(f"压缩：摘要内容解析后为空（{attempt}/{_MAX_RETRIES}）")

    logger.warning("压缩：达到最大重试次数仍未取得有效摘要，跳过")
    return ""


def _extract_summary_content(raw: str) -> str:
    """从 LLM 输出中提取摘要正文。

    期望格式为 ``{"content": "..."}``。JSON 解析失败时回退使用清理过
    markdown 围栏的原始文本——摘要本身是自然语言，即便格式不合规，
    正文通常仍然可用。

    Args:
        raw: LLM 返回的原始文本。

    Returns:
        str: 提取出的摘要正文。
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        try:
            data: Any = json.loads(cleaned[brace_start : brace_end + 1])
        except (json.JSONDecodeError, TypeError):
            logger.debug("压缩：JSON 解析失败，回退使用原始文本")
        else:
            content = data.get("content", "") if isinstance(data, dict) else ""
            if isinstance(content, list):
                return "\n".join(str(item) for item in content if str(item).strip())
            return str(content).strip()

    return cleaned
