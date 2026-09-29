"""把已封存的完整回合写入第一人称日记。"""

from __future__ import annotations

import datetime
import json
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.llm_api import create_llm_request, get_model_set_by_task
from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.types import LLMPayload, ROLE, Text

from .context_budget import KFCContextManager
from .context.renderer import build_diary_system_prompt

if TYPE_CHECKING:
    from src.app.plugin_system.types import ChatStream

    from .config import KFCConfig
    from .session import KFCSession, KFCSessionStore


logger = get_logger("kfc_diary_compressor")
_MAX_ATTEMPTS = 3


async def bootstrap_diary(
    session: KFCSession,
    config: KFCConfig,
    chat_stream: ChatStream,
    history_text: str,
    *,
    session_store: KFCSessionStore,
) -> bool:
    """从近期框架聊天记录生成初始日记，不修改活动链与封存队列。"""
    instruction = (
        f"当前时间：{datetime.datetime.now():%Y-%m-%d %H:%M}\n\n"
        f"最近的聊天记录：\n{history_text}\n\n"
        "请根据以上记录写一份第一人称'我'的日记，直接输出日记正文。"
        "保留重要事件、约定、关系变化和明确的日期时间；"
        "区分对方说过的话与我的推测，不编造没有发生的事情。"
        "不用'昨天'或'前天'代替明确日期，将这些记录整理为简短记忆。"
    )
    try:
        system_prompt = await build_diary_system_prompt(chat_stream)
        model_set = get_model_set_by_task(config.prompt.compress_model_task)
    except Exception as error:
        logger.error(f"构建初始日记请求失败：{error}")
        return False

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            request = create_llm_request(
                model_set,
                f"kfc_diary_{session.stream_id}",
                context_manager=KFCContextManager(),
            )
            request.add_payload(LLMPayload(ROLE.SYSTEM, Text(system_prompt)))
            request.add_payload(LLMPayload(ROLE.USER, Text(instruction)))
            result = await request.send(stream=False)
            diary = (await result or "").strip()
            if not diary:
                raise ValueError("初始日记为空")

            async with session_store.lock(session.stream_id):
                if session.history_summary.strip():
                    return True
                session.history_summary = diary
                try:
                    await session_store.save_context(session)
                except BaseException:
                    session.history_summary = ""
                    raise
            logger.info(f"初始日记生成完成：流 {session.stream_id[:8]}")
            return True
        except Exception as error:
            logger.warning(f"初始日记生成失败 ({attempt}/{_MAX_ATTEMPTS})：{error}")
    return False


def _render_segment(entries: list[dict[str, Any]]) -> str:
    """以保留角色及顺序的文本表示封存段，不复制媒体二进制。"""
    lines: list[str] = []
    for entry in entries:
        parts: list[str] = []
        for part in entry["content"]:
            kind = part["type"]
            if kind in {"text", "reasoning"}:
                parts.append(part["text"])
            elif kind == "tool_call":
                parts.append(f"调用 {part['name']}：{json.dumps(part['args'], ensure_ascii=False)}")
            elif kind == "tool_result":
                parts.append(f"工具结果：{part['value']}")
            else:
                parts.append(f"[{kind} 媒体]")
        lines.append(f"{entry['role']}：{' '.join(parts)}")
    return "\n".join(lines)


async def compress_history(
    session: KFCSession,
    config: KFCConfig,
    chat_stream: ChatStream,
    *,
    session_store: KFCSessionStore,
) -> bool:
    """三次以内生成新日记；成功后只替换日记和队首封存段。"""
    if not session.sealed_segments:
        return True

    segment = session.sealed_segments[0]
    previous_diary = session.history_summary
    source_text = _render_segment(segment)
    instruction = (
        f"当前时间：{datetime.datetime.now():%Y-%m-%d %H:%M}\n\n"
        f"已有的第一人称日记：\n{previous_diary or '（暂无）'}\n\n"
        f"接下来发生的对话与当时的思考：\n{source_text}\n\n"
        "请以第一人称'我'续写并整合成一份完整的日记，直接输出日记正文。"
        "忠实保留重要事件、约定、关系变化和关键原话；优先保留明确的日期和时间，"
        "不要用会随时间失效的'昨天'、'前天'替代明确日期，也不要编造时间。"
        "区分对方明确说过的话和我自己的推测。不按天机械分组，不规定篇幅，"
        "但须把旧日记和这段原话压缩为比输入更短的记忆。"
    )
    try:
        system_prompt = await build_diary_system_prompt(chat_stream)
        model_set = get_model_set_by_task(config.prompt.compress_model_task)
    except Exception as error:
        logger.error(f"构建日记压缩请求失败：{error}")
        return False

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            request = create_llm_request(
                model_set,
                f"kfc_diary_{session.stream_id}",
                context_manager=KFCContextManager(),
            )
            request.add_payload(LLMPayload(ROLE.SYSTEM, Text(system_prompt)))
            request.add_payload(LLMPayload(ROLE.USER, Text(instruction)))
            result = await request.send(stream=False)
            summary = (await result or "").strip()
            if not summary or len(summary) >= len(previous_diary) + len(source_text):
                raise ValueError("新日记为空或没有压缩原文")

            async with session_store.lock(session.stream_id):
                if not session.sealed_segments or session.sealed_segments[0] != segment:
                    raise RuntimeError("压缩期间封存段已变化")
                session.history_summary = summary
                session.sealed_segments.pop(0)
                try:
                    await session_store.save_context(session)
                except BaseException:
                    session.history_summary = previous_diary
                    session.sealed_segments.insert(0, segment)
                    raise
            logger.info(f"日记压缩完成：流 {session.stream_id[:8]}")
            return True
        except Exception as error:
            logger.warning(f"日记压缩失败 ({attempt}/{_MAX_ATTEMPTS})：{error}")
    return False