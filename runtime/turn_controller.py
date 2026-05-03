"""KFC 回合提交控制。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.base import Stop, Wait
from src.app.plugin_system.types import LLMPayload, ROLE, Text

from ..models import WaitingConfig
from ..services import SummaryService

if TYPE_CHECKING:
    from ..config import KFCConfig
    from ..domain.decision import Decision
    from ..prompts.builder import KFCPromptBuilder
    from ..session import KFCSession
    from ..chatter import KokoroFlowChatter
    from src.app.plugin_system.types import ChatStream
    from ..services.timeout_service import TimeoutService


logger = get_logger("kfc_chatter")


@dataclass(slots=True)
class TurnControlResult:
    """一轮决策提交后的控制结果。"""

    next_signal: Wait | Stop | None = None
    continue_loop: bool = False
    return_after_yield: bool = False
    has_pending_tool_results: bool = False
    is_final_timeout: bool = False


@dataclass(slots=True)
class TurnInputResult:
    """一轮主循环在发起 LLM 请求前的准备结果。"""

    response: Any
    unread_msgs: list[Any]
    extra_payload: LLMPayload | None = None
    next_signal: Wait | None = None
    continue_loop: bool = False
    history_images_injected: bool = False
    has_pending_tool_results: bool = False
    is_final_timeout: bool = False
    formatted_unreads: str = ""


async def prepare_turn_input(
    chatter: KokoroFlowChatter,
    response: Any,
    chat_stream: ChatStream,
    config: KFCConfig,
    session: KFCSession,
    prompt_builder: KFCPromptBuilder,
    timeout_service: TimeoutService,
    image_budget: Any,
    has_history: bool,
    history_images_injected: bool,
    has_pending_tool_results: bool,
) -> TurnInputResult:
    """准备一轮 LLM 调用前的触发输入。"""
    formatted_text, unread_msgs = await chatter.fetch_unreads(
        time_format="%Y-%m-%d %H:%M:%S"
    )
    extra_payload: LLMPayload | None = None
    is_final_timeout = False

    if formatted_text and unread_msgs:
        formatted_text, unread_msgs = await chatter._accumulate_messages(config)
        has_pending_tool_results = False
        for msg in unread_msgs:
            sender_id = getattr(msg, "sender_id", "")
            session.add_user_message(
                content=getattr(msg, "processed_plain_text", "")
                or str(getattr(msg, "content", "")),
                user_name=getattr(msg, "sender_name", "用户"),
                user_id=sender_id,
                timestamp=chatter._extract_timestamp(msg),
                message_id=getattr(msg, "message_id", ""),
            )
            if sender_id:
                session.user_id = sender_id
            if chat_stream.platform:
                session.platform = chat_stream.platform

        if session.is_waiting():
            chatter._record_reply_timing(session)
            session.clear_waiting()

        media_items = chatter._extract_media(
            unread_msgs,
            config,
            image_budget,
        )

        if (
            not history_images_injected
            and has_history
            and image_budget is not None
            and not image_budget.is_exhausted()
        ):
            history_images_injected = True
            history_imgs = chatter._extract_history_media(
                chat_stream,
                image_budget,
            )
            from ..services import MultimodalService

            MultimodalService.append_history_reference(response, history_imgs or [])

        user_payload, extra_payload = await prompt_builder.build_user_payload(
            formatted_unreads=formatted_text,
            media_items=media_items,
            stream_id=chatter.stream_id,
        )

        if response.payloads and response.payloads[-1].role == ROLE.TOOL_RESULT:
            logger.debug(
                "新消息到达时 response 尾部为 tool_result，"
                "插入 assistant 桥接 payload 以闭合工具链"
            )
            response.add_payload(LLMPayload(ROLE.ASSISTANT, Text("好的。")))

        upserted = False
        if (
            not media_items
            and response.payloads
            and response.payloads[-1].role == ROLE.USER
        ):
            last_payload = response.payloads[-1]
            if last_payload.content and isinstance(last_payload.content[-1], Text):
                existing = last_payload.content[-1].text  # type: ignore[attr-defined]
                last_payload.content[-1] = Text(
                    f"{existing}\n{user_payload.content[-1].text}"  # type: ignore[attr-defined]
                    if isinstance(user_payload.content, list)
                    else f"{existing}\n{user_payload.content.text}"  # type: ignore[attr-defined]
                )
                upserted = True
                logger.debug("[KFC] Upsert USER payload（打断重来合并新消息）")
        if not upserted:
            response.add_payload(user_payload)
    elif has_pending_tool_results:
        has_pending_tool_results = False
    elif session.is_waiting():
        if timeout_service.check_timeout(session):
            timeout_result = timeout_service.build_timeout_result(
                response,
                session,
            )
            is_final_timeout = timeout_result.is_final_timeout
            timeout_upserted = False
            if response.payloads and response.payloads[-1].role == ROLE.USER:
                last_payload = response.payloads[-1]
                timeout_text = (
                    timeout_result.payload.content.text  # type: ignore[attr-defined]
                    if isinstance(timeout_result.payload.content, Text)
                    else ""
                )
                if timeout_text and last_payload.content and isinstance(last_payload.content[-1], Text):
                    last_payload.content[-1] = Text(
                        f"{last_payload.content[-1].text}\n{timeout_text}"  # type: ignore[attr-defined]
                    )
                    timeout_upserted = True
            if not timeout_upserted:
                response.add_payload(timeout_result.payload)
        else:
            return TurnInputResult(
                response=response,
                unread_msgs=[],
                next_signal=Wait(0),
                continue_loop=True,
                history_images_injected=history_images_injected,
                has_pending_tool_results=has_pending_tool_results,
                is_final_timeout=is_final_timeout,
            )
    else:
        return TurnInputResult(
            response=response,
            unread_msgs=[],
            next_signal=Wait(0),
            continue_loop=True,
            history_images_injected=history_images_injected,
            has_pending_tool_results=has_pending_tool_results,
            is_final_timeout=is_final_timeout,
        )

    return TurnInputResult(
        response=response,
        unread_msgs=unread_msgs,
        extra_payload=extra_payload,
        history_images_injected=history_images_injected,
        has_pending_tool_results=has_pending_tool_results,
        is_final_timeout=is_final_timeout,
        formatted_unreads=formatted_text,
    )


async def commit_turn_decision(
    chatter: KokoroFlowChatter,
    decision: Decision,
    response: Any,
    session: KFCSession,
    config: KFCConfig,
    prompt_builder: KFCPromptBuilder,
    chat_stream: ChatStream,
    pre_send_user_text: str,
    last_user_ts: float,
    chain_user_pre_saved: bool,
    is_final_timeout: bool,
) -> TurnControlResult:
    """提交本轮 Decision 对 session 与主循环的影响。"""
    session.add_bot_planning(
        thought=decision.thought,
        actions=decision.actions,
        expected_reaction=decision.expected_reaction,
        max_wait_seconds=decision.wait_seconds,
        raw_response=getattr(response, "message", "") or "",
    )

    assistant_text = (getattr(response, "message", "") or "").strip()
    if not assistant_text:
        assistant_text = decision.reply_text
    call_list = getattr(response, "call_list", None) or []
    serialized_tool_calls = [
        {"name": tc.name, "args": tc.args, "id": tc.id}
        for tc in call_list
        if hasattr(tc, "name") and hasattr(tc, "args")
    ]
    if pre_send_user_text and (assistant_text or serialized_tool_calls):
        assistant_entry: dict = {"role": "assistant", "text": assistant_text}
        if serialized_tool_calls:
            assistant_entry["tool_calls"] = serialized_tool_calls
        if chain_user_pre_saved:
            session.update_chain(
                [assistant_entry],
                config.prompt.max_context_payloads,
            )
        else:
            session.update_chain(
                [
                    {"role": "user", "text": pre_send_user_text, "ts": last_user_ts},
                    assistant_entry,
                ],
                config.prompt.max_context_payloads,
            )
        await chatter._save_session(session)
        session.compress_round_count += 1
        SummaryService.maybe_schedule_compression(
            session,
            prompt_builder,
            config,
            chat_stream,
        )

    if not decision.has_meaningful_action:
        if response.message and response.message.strip():
            logger.warning(
                f"LLM 返回未形成有效决策: {response.message[:100]}"
            )
        await chatter._save_session(session)
        return TurnControlResult(
            next_signal=Stop(0),
            return_after_yield=True,
            is_final_timeout=is_final_timeout,
        )

    if decision.chose_silence and not decision.should_reply:
        if decision.wait_seconds <= 0:
            logger.debug("do_nothing（无等待），结束对话")
            await chatter._save_session(session)
            return TurnControlResult(
                next_signal=Stop(0),
                return_after_yield=True,
                is_final_timeout=is_final_timeout,
            )

    if decision.has_info_tool_calls and not decision.should_reply:
        logger.debug("信息工具调用完成，tool_result 已积累到 response 链，立即续轮")
        return TurnControlResult(
            continue_loop=True,
            has_pending_tool_results=True,
            is_final_timeout=is_final_timeout,
        )
    if (
        decision.has_third_party_calls
        and not decision.should_reply
        and not decision.chose_silence
    ):
        logger.debug(
            "第三方工具调用完成，tool_result 已积累到 response 链，下轮循环继续"
        )
        return TurnControlResult(
            continue_loop=True,
            has_pending_tool_results=True,
            is_final_timeout=is_final_timeout,
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
        waiting_config = WaitingConfig(
            expected_reaction=decision.expected_reaction,
            max_wait_seconds=wait_seconds,
            started_at=time.time(),
        )
        session.set_waiting(waiting_config)
        await chatter._save_session(session)
        return TurnControlResult(
            next_signal=Wait(0),
            continue_loop=True,
            is_final_timeout=is_final_timeout,
        )

    session.clear_waiting()
    await chatter._save_session(session)
    return TurnControlResult(
        next_signal=Stop(0),
        return_after_yield=True,
        is_final_timeout=is_final_timeout,
    )