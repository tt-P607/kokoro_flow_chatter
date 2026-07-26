"""KFC 回合的输入准备与决策提交。

一轮循环分为两个阶段，分别对应本模块的两个入口：

- ``prepare_turn_input()``：判定触发原因，据此准备本轮的 LLM 输入；
- ``commit_turn_decision()``：把决策结果落到会话状态，并告诉主循环
  接下来该续轮、等待还是收口。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.base import Stop, Wait
from src.app.plugin_system.types import LLMPayload, ROLE, Text

from ..context import (
    plan_followup_contributions,
    plan_user_turn,
    render_turn_contributions,
    render_user_payload,
)
from ..domain.chain_entry import ChainEntry
from ..domain.turn_trigger import TurnTrigger, classify_turn_trigger
from ..models import KFCEventType, WaitingConfig
from ..services import SummaryService
from .unread_policy import format_unread_messages, prefer_real_unreads

if TYPE_CHECKING:
    from src.app.plugin_system.types import ChatStream

    from ..chatter import KokoroFlowChatter
    from ..config import KFCConfig
    from ..domain.decision import Decision
    from ..services.timeout_service import TimeoutService
    from ..session import KFCSession

logger = get_logger("kfc_turn")

_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(slots=True)
class TurnInputResult:
    """一轮 LLM 调用前的输入准备结果。"""

    response: Any
    unread_msgs: list[Any] = field(default_factory=list)
    extra_payload: LLMPayload | None = None
    """本轮临时注入的上下文贡献，发送后不入对话链。"""

    chain_text: str = ""
    """本轮待写入对话链的用户文本；非新消息路径为空。"""

    next_signal: Wait | None = None
    continue_loop: bool = False
    has_pending_tool_results: bool = False
    is_final_timeout: bool = False


@dataclass(slots=True)
class TurnControlResult:
    """一轮决策提交后的主循环控制指令。"""

    next_signal: Wait | Stop | None = None
    continue_loop: bool = False
    return_after_yield: bool = False
    has_pending_tool_results: bool = False
    is_final_timeout: bool = False
    chain_assistant_saved: bool = False


def build_chain_assistant_entry(
    assistant_text: str,
    serialized_tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """构造写入对话链的 assistant 条目。

    占位补齐与字段过滤由 ``ChainEntry.assistant`` 内部完成。
    """
    return ChainEntry.assistant(assistant_text, serialized_tool_calls).to_dict()


async def prepare_turn_input(
    chatter: KokoroFlowChatter,
    response: Any,
    chat_stream: ChatStream,
    config: KFCConfig,
    session: KFCSession,
    timeout_service: TimeoutService,
    has_pending_tool_results: bool,
) -> TurnInputResult:
    """准备本轮 LLM 调用的输入。

    先做备忘录懒清理，再按触发原因分派到三条路径之一：新消息、工具
    续轮、等待超时；都不成立时让出本 tick。

    Args:
        chatter: 当前 chatter 实例。
        response: 当前 LLM 响应链。
        chat_stream: 当前聊天流。
        config: KFC 配置。
        session: 当前会话。
        timeout_service: 超时服务。
        has_pending_tool_results: 上轮是否留下待消化的工具结果。

    Returns:
        TurnInputResult: 本轮输入准备结果。
    """
    # 每轮请求前清掉过期备忘并补事件，让模型能看到"我记的某事到期了"
    for expired_memo in session.prune_expired_memos():
        session.add_memo_event(KFCEventType.MEMO_EXPIRED, expired_memo)

    _, raw_unreads = await chatter.fetch_unreads(time_format=_TIME_FORMAT)
    unread_msgs = await prefer_real_unreads(chatter, raw_unreads)

    is_timeout = (
        not unread_msgs
        and not has_pending_tool_results
        and session.is_waiting()
        and timeout_service.check_timeout(session)
    )
    trigger = classify_turn_trigger(
        has_unread=bool(unread_msgs),
        has_pending_tool_results=has_pending_tool_results,
        session=session,
        is_timeout=is_timeout,
    )

    if trigger is TurnTrigger.NEW_MESSAGES:
        return await _prepare_new_messages(
            chatter, response, chat_stream, config, session, unread_msgs
        )
    if trigger is TurnTrigger.FOLLOWUP_TOOL_RESULT:
        return TurnInputResult(
            response=response,
            extra_payload=await _collect_followup_payload(chatter.stream_id),
            has_pending_tool_results=False,
        )
    if trigger is TurnTrigger.TIMEOUT_EXPIRED:
        return await _prepare_timeout(chatter, response, session, timeout_service)

    return TurnInputResult(
        response=response,
        next_signal=Wait(0),
        continue_loop=True,
        has_pending_tool_results=has_pending_tool_results,
    )


async def _prepare_new_messages(
    chatter: KokoroFlowChatter,
    response: Any,
    chat_stream: ChatStream,
    config: KFCConfig,
    session: KFCSession,
    unread_msgs: list[Any],
) -> TurnInputResult:
    """处理新消息路径：记录消息、构建用户 payload。"""
    for message in unread_msgs:
        sender_id = message.sender_id or ""
        session.add_user_message(
            content=message.processed_plain_text or str(message.content or ""),
            user_name=message.sender_name or "用户",
            user_id=sender_id,
            timestamp=chatter.extract_timestamp(message),
            message_id=message.message_id or "",
        )
        if sender_id:
            session.user_id = sender_id
        if chat_stream.platform:
            session.platform = chat_stream.platform

    if session.is_waiting():
        chatter.record_reply_timing(session)
        session.clear_waiting()

    media_items = None
    if config.general.native_multimodal:
        from ..multimodal import extract_images_from_messages

        media_items = extract_images_from_messages(unread_msgs) or None

    plan = await plan_user_turn(
        formatted_unreads=format_unread_messages(chatter, unread_msgs),
        stream_id=chatter.stream_id,
        session=session,
    )
    user_payload, extra_payload, chain_text = render_user_payload(plan, media_items)
    _append_or_merge_user_payload(response, user_payload, allow_merge=not media_items)

    return TurnInputResult(
        response=response,
        unread_msgs=unread_msgs,
        extra_payload=extra_payload,
        chain_text=chain_text,
        has_pending_tool_results=False,
    )


async def _prepare_timeout(
    chatter: KokoroFlowChatter,
    response: Any,
    session: KFCSession,
    timeout_service: TimeoutService,
) -> TurnInputResult:
    """处理超时路径：注入超时决策提示。"""
    timeout_result = timeout_service.build_timeout_result(session)
    _append_or_merge_user_payload(response, timeout_result.payload, allow_merge=True)
    return TurnInputResult(
        response=response,
        extra_payload=await _collect_followup_payload(chatter.stream_id),
        is_final_timeout=timeout_result.is_final_timeout,
    )


async def _collect_followup_payload(stream_id: str) -> LLMPayload | None:
    """为续轮/超时路径收集第三方上下文贡献。

    这两条路径不新增用户消息，但仍需收集注入——否则 ``prompt_injector``
    等插件提供的内容会在续轮中丢失。
    """
    plan = await plan_followup_contributions(stream_id)
    return render_turn_contributions(plan.contributions)


def _append_or_merge_user_payload(
    response: Any,
    payload: LLMPayload,
    *,
    allow_merge: bool,
) -> None:
    """追加 USER payload；末尾已是纯文本 USER 时合并进去。

    合并可避免"打断后重来"或"超时叠加"场景在链尾堆出多条连续 USER。
    含图片的 payload 不合并，以免破坏图文顺序。
    """
    new_text = _extract_text(payload)
    if not allow_merge or not new_text:
        response.add_payload(payload)
        return

    payloads = response.payloads
    if not payloads or payloads[-1].role != ROLE.USER:
        response.add_payload(payload)
        return

    last_content = payloads[-1].content
    if not isinstance(last_content, list) or not last_content:
        response.add_payload(payload)
        return
    if not isinstance(last_content[-1], Text):
        response.add_payload(payload)
        return

    last_content[-1] = Text(f"{last_content[-1].text}\n{new_text}")
    logger.debug("已把新 USER 内容合并进链尾 payload")


def _extract_text(payload: LLMPayload) -> str:
    """提取 payload 中的全部文本片段。"""
    content = payload.content
    if not isinstance(content, list):
        content = [content]
    return "".join(item.text for item in content if isinstance(item, Text))


async def commit_turn_decision(
    chatter: KokoroFlowChatter,
    decision: Decision,
    response: Any,
    session: KFCSession,
    config: KFCConfig,
    chat_stream: ChatStream,
    *,
    pending_user_text: str,
    last_user_ts: float,
    chain_user_saved: bool,
    is_final_timeout: bool,
) -> TurnControlResult:
    """把本轮决策提交到会话，并给出主循环的下一步指令。

    Args:
        chatter: 当前 chatter 实例。
        decision: 本轮决策。
        response: 当前 LLM 响应。
        session: 当前会话。
        config: KFC 配置。
        chat_stream: 当前聊天流。
        pending_user_text: 待入链的用户文本。
        last_user_ts: 该用户文本对应的时间戳。
        chain_user_saved: 用户条目是否已提前入链。
        is_final_timeout: 本轮是否为最后一次超时。

    Returns:
        TurnControlResult: 主循环控制指令。
    """
    session.add_bot_planning(
        thought=decision.thought,
        actions=decision.actions,
        expected_reaction=decision.expected_reaction,
        max_wait_seconds=decision.wait_seconds,
        raw_response=response.message or "",
    )

    chain_assistant_saved = await _save_assistant_chain(
        chatter,
        decision,
        response,
        session,
        config,
        chat_stream,
        pending_user_text=pending_user_text,
        last_user_ts=last_user_ts,
        chain_user_saved=chain_user_saved,
    )

    if not decision.has_meaningful_action:
        raw_message = (response.message or "").strip()
        if raw_message:
            logger.warning(f"LLM 返回未形成有效决策: {raw_message[:100]}")
        else:
            logger.warning("LLM 未返回有效动作且消息为空，强制终止本次对话循环")
        await chatter.save_session(session)
        return TurnControlResult(
            next_signal=Stop(0),
            return_after_yield=True,
            is_final_timeout=is_final_timeout,
        )

    # 信息类工具有返回值，无论是否同时回复都要续轮让模型看到结果
    if decision.has_info_tool_calls:
        logger.debug("信息工具已执行，工具结果已入链，立即续轮")
        return TurnControlResult(
            continue_loop=True,
            has_pending_tool_results=True,
            is_final_timeout=is_final_timeout,
            chain_assistant_saved=chain_assistant_saved,
        )

    # action 类工具无返回值，只有在没回复也没选择沉默时才需要续轮补话
    if (
        decision.has_third_party_calls
        and not decision.should_reply
        and not decision.chose_silence
    ):
        logger.debug("第三方动作已执行但本轮未回复，续轮让模型补充表达")
        return TurnControlResult(
            continue_loop=True,
            has_pending_tool_results=True,
            is_final_timeout=is_final_timeout,
            chain_assistant_saved=chain_assistant_saved,
        )

    wait_seconds = config.wait.apply_rules(
        decision.wait_seconds,
        session.consecutive_timeout_count,
    )
    if is_final_timeout and wait_seconds > 0:
        logger.info("最后一次超时决策完成，强制结束等待")
        wait_seconds = 0
        is_final_timeout = False

    if wait_seconds > 0:
        session.set_waiting(
            WaitingConfig(
                expected_reaction=decision.expected_reaction,
                max_wait_seconds=wait_seconds,
                started_at=time.time(),
            )
        )
        await chatter.save_session(session)
        return TurnControlResult(
            next_signal=Wait(0),
            continue_loop=True,
            is_final_timeout=is_final_timeout,
            chain_assistant_saved=chain_assistant_saved,
        )

    session.clear_waiting()
    await chatter.save_session(session)
    return TurnControlResult(
        next_signal=Stop(0),
        return_after_yield=True,
        is_final_timeout=is_final_timeout,
        chain_assistant_saved=chain_assistant_saved,
    )


async def _save_assistant_chain(
    chatter: KokoroFlowChatter,
    decision: Decision,
    response: Any,
    session: KFCSession,
    config: KFCConfig,
    chat_stream: ChatStream,
    *,
    pending_user_text: str,
    last_user_ts: float,
    chain_user_saved: bool,
) -> bool:
    """把本轮 assistant 输出写入对话链，并按需调度记忆压缩。

    Returns:
        bool: 是否写入了 assistant 条目。
    """
    if not pending_user_text:
        return False

    assistant_text = (response.message or "").strip() or decision.reply_text
    serialized_tool_calls = [
        {"name": call.name, "args": call.args, "id": call.id}
        for call in response.call_list or []
    ]
    if not assistant_text and not serialized_tool_calls:
        return False

    entries = [build_chain_assistant_entry(assistant_text, serialized_tool_calls)]
    if not chain_user_saved:
        entries.insert(
            0, ChainEntry.user(pending_user_text, ts=last_user_ts).to_dict()
        )
    session.update_chain(entries, config.prompt.max_context_payloads)
    await chatter.save_session(session)

    session.compress_round_count += 1
    SummaryService.maybe_schedule_compression(
        session,
        config,
        chat_stream,
        session_store=chatter.session_store,
    )
    return True
