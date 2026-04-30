"""KFC 运行时总控。"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, AsyncGenerator

from src.app.plugin_system.api.llm_api import (
    get_model_set_by_name,
    get_model_set_by_task,
)
from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.base import Failure, Stop, Success, Wait
from src.kernel.llm import LLMPayload, ROLE, Text

from ..debug.log_formatter import log_kfc_result
from ..protocol.compat_adapter import prepare_kfc_model_set
from ..protocol.decision_parser import parse_response_decision
from ..services import (
    MultimodalService,
    ProactiveService,
    SummaryService,
    TimeoutService,
)
from .turn_controller import commit_turn_decision, prepare_turn_input

if TYPE_CHECKING:
    from ..chatter import KokoroFlowChatter


logger = get_logger("kfc_chatter")


async def execute_orchestrator(
    chatter: KokoroFlowChatter,
) -> AsyncGenerator[Wait | Success | Failure | Stop, None]:
    """执行 KFC 对话主循环。"""
    from src.app.plugin_system.api.stream_api import activate_stream

    self = chatter

    chat_stream = await activate_stream(self.stream_id)
    if chat_stream is None:
        logger.error(f"无法激活聊天流: {self.stream_id}")
        yield Failure("聊天流激活失败")
        return
    config = self._get_config()

    if not config.general.enabled:
        logger.debug("KFC 插件已禁用，跳过 execute")
        yield Stop(0)
        return

    session = await self._get_session()
    timeout_service = TimeoutService(config)

    vlm_registered = False
    if config.general.native_multimodal:
        self._register_vlm_skip()
        vlm_registered = True

    try:
        model_set = None
        temperature = config.general.temperature
        max_tokens = config.general.max_tokens
        if config.general.models:
            parts = [
                get_model_set_by_name(
                    model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                for model_name in config.general.models
            ]
            valid_parts = [part for part in parts if part]
            if valid_parts:
                model_set = valid_parts[0]
                for part in valid_parts[1:]:
                    model_set = model_set + part
            if not model_set:
                logger.warning(
                    f"models 中的模型均未注册: {config.general.models}，"
                    f"回退到任务模型 '{config.general.model_task}'"
                )

        if not model_set:
            model_set = get_model_set_by_task(config.general.model_task)

        if not model_set:
            logger.error("无法获取模型配置")
            yield Failure("模型配置错误：未找到有效的模型配置")
            return

        model_set = prepare_kfc_model_set(model_set)

        (
            response,
            image_budget,
            usable_map,
            prompt_builder,
            has_history,
        ) = await self._build_initial_context(
            chat_stream,
            config,
            session,
            model_set,
        )

        history_images_injected = False
        has_pending_tool_results = False
        is_final_timeout = False
        pre_send_user_text = ""
        last_user_ts = 0.0
        chain_user_pre_saved = False
        extra_payload: LLMPayload | None = None

        while True:
            turn_input = await prepare_turn_input(
                self,
                response,
                chat_stream,
                config,
                session,
                prompt_builder,
                timeout_service,
                image_budget,
                has_history,
                history_images_injected,
                has_pending_tool_results,
            )
            response = turn_input.response
            unread_msgs = turn_input.unread_msgs
            extra_payload = turn_input.extra_payload
            history_images_injected = turn_input.history_images_injected
            has_pending_tool_results = turn_input.has_pending_tool_results
            is_final_timeout = turn_input.is_final_timeout

            if turn_input.next_signal is not None:
                yield turn_input.next_signal
            if turn_input.continue_loop:
                continue

            if unread_msgs:
                last_user_ts = min(
                    (self._extract_timestamp(message) for message in unread_msgs),
                    default=time.time(),
                )

            new_user_text = ""
            for payload in reversed(response.payloads):
                if payload.role == ROLE.USER:
                    new_user_text = "".join(
                        chunk.text  # type: ignore[attr-defined]
                        for chunk in payload.content
                        if isinstance(chunk, Text)
                    )
                    break
            if new_user_text != pre_send_user_text:
                pre_send_user_text = new_user_text
                chain_user_pre_saved = False

            if pre_send_user_text and not chain_user_pre_saved:
                session.update_chain(
                    [{"role": "user", "text": pre_send_user_text, "ts": last_user_ts}],
                    config.prompt.max_context_payloads,
                )
                await self._save_session(session)
                chain_user_pre_saved = True

            extra_payload_added = False
            if extra_payload is not None:
                response.payloads.append(extra_payload)
                extra_payload_added = True
            if config.debug.show_prompt:
                self._log_prompt(response)

            if unread_msgs:
                known_ids: frozenset[str] = frozenset(
                    message_id
                    for message in unread_msgs
                    if (message_id := getattr(message, "message_id", None)) is not None
                )
            else:
                _, current_snapshot = await self.fetch_unreads(
                    time_format="%Y-%m-%d %H:%M:%S"
                )
                known_ids = frozenset(
                    message_id
                    for message in current_snapshot
                    if (message_id := getattr(message, "message_id", None)) is not None
                )

            try:
                if config.buffer.interrupt_enabled:
                    new_response, interrupt_msgs = await self._send_interruptable(
                        response,
                        config,
                        known_ids,
                    )
                    if interrupt_msgs:
                        if extra_payload_added and extra_payload is not None:
                            response.payloads = [
                                payload
                                for payload in response.payloads
                                if payload is not extra_payload
                            ]
                        extra_payload = None
                        await self.flush_unreads(unread_msgs or [])
                        session.add_interrupt_event(interrupt_msgs)
                        await self._save_session(session)
                        continue
                    assert new_response is not None
                    response = new_response
                else:
                    response = await self._send_with_perceive_loop(
                        response,
                        config.general.max_compat_retries,
                    )
                await self.flush_unreads(unread_msgs if unread_msgs else [])
            except Exception as exc:
                logger.error(f"LLM 请求失败: {exc}", exc_info=True)
                if extra_payload_added and extra_payload is not None:
                    response.payloads = [
                        payload
                        for payload in response.payloads
                        if payload is not extra_payload
                    ]
                extra_payload = None
                await self._save_session(session)
                yield Failure("LLM 请求失败", exc)
                break

            if extra_payload_added and extra_payload is not None:
                response.payloads = [
                    payload for payload in response.payloads if payload is not extra_payload
                ]
            extra_payload = None

            call_list = getattr(response, "call_list", None) or []
            if call_list:
                logger.info(f"本轮调用列表：{[call.name for call in call_list]}")
            elif getattr(response, "message", ""):
                logger.debug("[KFC] 本轮无 tool call，等待标准化器判定是否需要重试")

            trigger_msg = unread_msgs[-1] if unread_msgs else None
            if trigger_msg is None:
                trigger_msg = await self._get_virtual_trigger_message()
            decision = await parse_response_decision(
                response,
                usable_map,
                trigger_msg,
                config,
                execute_reply_fn=self._execute_reply,
                run_tool_call_fn=self.run_tool_call,
                pre_execute_hook=lambda result: log_kfc_result(result, config),
            )

            if decision.proactive_schedule is not None:
                try:
                    ProactiveService.apply_schedule(
                        session,
                        decision.proactive_schedule,
                    )
                except Exception as exc:
                    logger.warning(f"[KFC] schedule_proactive 参数解析失败: {exc}")

            turn_control = await commit_turn_decision(
                self,
                decision,
                response,
                session,
                config,
                prompt_builder,
                chat_stream,
                pre_send_user_text,
                last_user_ts,
                chain_user_pre_saved,
                is_final_timeout,
            )
            is_final_timeout = turn_control.is_final_timeout

            if turn_control.has_pending_tool_results:
                has_pending_tool_results = True

            if turn_control.next_signal is not None:
                yield turn_control.next_signal
            if turn_control.return_after_yield:
                return
            if turn_control.continue_loop:
                continue

    finally:
        if vlm_registered:
            self._unregister_vlm_skip()