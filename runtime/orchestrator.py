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
from src.kernel.concurrency import get_watchdog
from src.app.plugin_system.types import ChatStream, LLMPayload, ROLE, Text

from ..debug.log_formatter import log_kfc_result
from ..domain.chain_entry import ChainEntry
from ..protocol.compat_adapter import prepare_kfc_model_set
from ..protocol.decision_parser import parse_response_decision
from ..protocol.response_normalizer import normalize_response
from .phase_machine import (
    ConversationPhase,
    phase_for_model_result,
    phase_for_turn_start,
)
from .request_view import build_request_view
from ..services import (
    ProactiveService,
    TimeoutService,
)
from .turn_controller import commit_turn_decision, prepare_turn_input

if TYPE_CHECKING:
    from ..chatter import KokoroFlowChatter


def _heal_orphan_tool_results(response: Any, *, where: str) -> int:
    """扫描 response.payloads，丢弃孤立的 TOOL_RESULT。

    "孤立" 的判定：一个 TOOL_RESULT 之前必须紧跟 ASSISTANT(含 tool_calls)
    或另一个连续的 TOOL_RESULT；否则视为非法链路状态，就地移除并打 ERROR 日志。

    Args:
        response: 拥有 payloads 列表的响应对象。
        where: 调用位置标识（用于日志），例如 "loop-top"。

    Returns:
        int: 被丢弃的孤立 TOOL_RESULT 数量。
    """
    payloads = getattr(response, "payloads", None)
    if not isinstance(payloads, list) or not payloads:
        return 0

    pinned_roles = {ROLE.SYSTEM, ROLE.TOOL}
    healed = 0
    idx = 0
    while idx < len(payloads):
        payload = payloads[idx]
        if payload.role != ROLE.TOOL_RESULT or payload.role in pinned_roles:
            idx += 1
            continue

        prev_idx = idx - 1
        while prev_idx >= 0 and payloads[prev_idx].role in pinned_roles:
            prev_idx -= 1

        prev_payload = payloads[prev_idx] if prev_idx >= 0 else None
        prev_role = prev_payload.role if prev_payload is not None else None

        valid_prev = prev_role == ROLE.TOOL_RESULT or (
            prev_role == ROLE.ASSISTANT
            and prev_payload is not None
            and _assistant_has_tool_calls(prev_payload)
        )
        if valid_prev:
            idx += 1
            continue

        snapshot_start = max(0, idx - 5)
        snapshot_end = min(len(payloads), idx + 6)
        snapshot = [
            f"[{s_idx}] {payloads[s_idx].role.value}: {_preview_payload(payloads[s_idx])}"
            for s_idx in range(snapshot_start, snapshot_end)
        ]
        logger.error(
            f"孤立 TOOL_RESULT 自愈（{where}）：丢弃 idx={idx}，"
            f"prev_role={prev_role.value if prev_role else None}\n"
            + "\n".join(snapshot)
        )
        payloads.pop(idx)
        healed += 1

    return healed


def _assistant_has_tool_calls(payload: LLMPayload) -> bool:
    """判断 ASSISTANT payload 是否包含 tool_calls。"""
    content = payload.content
    if not isinstance(content, list):
        return False
    return any(type(item).__name__ == "ToolCall" for item in content)


def _preview_payload(payload: LLMPayload) -> str:
    """将 payload 内容压成短预览字符串（最多 80 字符）。"""
    content = payload.content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            type_name = type(item).__name__
            text_attr = getattr(item, "text", None)
            if isinstance(text_attr, str):
                parts.append(f"{type_name}({text_attr[:30]!r})")
            else:
                parts.append(f"{type_name}(name={getattr(item, 'name', None)!r})")
        return " | ".join(parts)[:80]
    text_attr = getattr(content, "text", None)
    return (repr(text_attr)[:80] if isinstance(text_attr, str) else repr(content)[:80])


logger = get_logger("kfc_chatter")

# LLM 返回纯文本（无 tool call）时的最大重试次数
_MAX_PLAIN_TEXT_RETRIES = 1

# 重试时注入的提醒文本
_PLAIN_TEXT_RETRY_REMINDER = (
    "（系统提示：你刚才返回了纯文本而非工具调用。"
    "请务必通过 kfc_reply 或 do_nothing 工具调用来完成响应，不要直接输出文字。）"
)


# 摘要标记前缀（用于在动态 USER payload 中定位摘要段落）
_SUMMARY_MARKER_PREFIX = "【你对"
_SUMMARY_MARKER_SUFFIX = "的近期记忆】"
_SECTION_SEPARATOR = "\n\n---\n\n"


def _hot_update_summary(
    response: Any,
    chat_stream: ChatStream,
    session: Any,
    prompt_builder: Any,
) -> None:
    """在 response 的动态 USER payload 中热替换摘要段落。

    动态 USER payload 的结构为 channel_info + summary + history，
    各部分以 ``\\n\\n---\\n\\n`` 分隔。本函数通过定位摘要标记前缀
    ``【你对...的近期记忆】`` 来找到并替换摘要段落。
    如果原始 payload 中不存在摘要（首次生成），则在适当位置插入。
    """
    from ..context.sources.history_source import build_history_summary_payload

    new_summary = session.history_summary or ""
    if not new_summary:
        return

    # 构建新的摘要文本
    summary_payload = build_history_summary_payload(chat_stream, new_summary)
    if summary_payload is None:
        return
    new_summary_text = ""
    for item in summary_payload.content:
        if hasattr(item, "text"):
            new_summary_text += item.text  # type: ignore[attr-defined]
    if not new_summary_text:
        return

    # 定位第一个 USER payload（动态上下文 payload）
    payloads = getattr(response, "payloads", [])
    dynamic_idx = -1
    for idx, payload in enumerate(payloads):
        if payload.role == ROLE.USER:
            dynamic_idx = idx
            break

    if dynamic_idx < 0:
        return

    dynamic_payload = payloads[dynamic_idx]
    # 提取当前文本
    if isinstance(dynamic_payload.content, list):
        old_text = ""
        for item in dynamic_payload.content:
            if hasattr(item, "text"):
                old_text += item.text  # type: ignore[attr-defined]
    elif hasattr(dynamic_payload.content, "text"):
        old_text = dynamic_payload.content.text  # type: ignore[attr-defined]
    else:
        return

    # 按段落分割并替换/插入摘要
    sections = old_text.split(_SECTION_SEPARATOR)
    summary_found = False
    for i, section in enumerate(sections):
        if _SUMMARY_MARKER_PREFIX in section and _SUMMARY_MARKER_SUFFIX in section:
            sections[i] = new_summary_text
            summary_found = True
            break

    if not summary_found:
        # 摘要不存在（首次生成），在第一个段落（通道信息）之后插入
        if len(sections) >= 2:
            sections.insert(1, new_summary_text)
        else:
            sections.append(new_summary_text)

    new_text = _SECTION_SEPARATOR.join(sections)

    # 替换 payload 内容
    if isinstance(dynamic_payload.content, list):
        dynamic_payload.content[:] = [Text(new_text)]
    else:
        dynamic_payload.content = Text(new_text)


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

    if config.general.native_multimodal:
        self._register_vlm_skip()

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
            usable_map,
            prompt_builder,
            has_history,
        ) = await self._build_initial_context(
            chat_stream,
            config,
            session,
            model_set,
        )

        has_pending_tool_results = False
        is_final_timeout = False
        pre_send_user_text = ""
        last_user_ts = 0.0
        chain_user_pre_saved = False
        extra_payload: LLMPayload | None = None
        phase = ConversationPhase.WAIT_INPUT
        plain_text_retry_count = 0
        follow_up_count = 0
        # 记录当前"烧入"初始上下文的摘要，用于检测后台压缩任务的更新
        _baked_summary = session.history_summary or ""

        while True:
            # ── 摘要热更新 ──
            # 后台压缩任务可能已更新 session.history_summary，
            # 如果发生变化，立即替换 response 中的动态上下文 payload。
            current_summary = session.history_summary or ""
            if current_summary != _baked_summary:
                _hot_update_summary(
                    response, chat_stream, session, prompt_builder,
                )
                _baked_summary = current_summary
                logger.info("[KFC] 近期记忆摘要已热更新到 LLM 上下文")

            _heal_orphan_tool_results(response, where="loop-top")
            phase = phase_for_turn_start(
                response,
                has_pending_tool_results=has_pending_tool_results,
            )
            if phase is ConversationPhase.FOLLOW_UP:
                logger.debug("[KFC] role-phase=FOLLOW_UP，跳过新增 USER 输入并续轮")
            turn_input = await prepare_turn_input(
                self,
                response,
                chat_stream,
                config,
                session,
                prompt_builder,
                timeout_service,
                has_pending_tool_results,
            )
            response = turn_input.response
            unread_msgs = turn_input.unread_msgs
            extra_payload = turn_input.extra_payload
            # 新消息到来时（has_pending_tool_results 由 True 变 False），重置续轮计数
            if has_pending_tool_results and not turn_input.has_pending_tool_results:
                follow_up_count = 0
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

            # wrapped_user_text 是 chain_text（仅含原始消息内容），
            # 不含末尾强调指令/平台信息/system_reminder，避免这些临时提示词被持久化进链。
            if turn_input.wrapped_user_text:
                new_user_text = turn_input.wrapped_user_text
                if new_user_text != pre_send_user_text:
                    pre_send_user_text = new_user_text
                    chain_user_pre_saved = False

            if pre_send_user_text and not chain_user_pre_saved:
                session.update_chain(
                    [ChainEntry.user(pre_send_user_text, ts=last_user_ts).to_dict()],
                    config.prompt.max_context_payloads,
                )
                await self._save_session(session)
                chain_user_pre_saved = True

            transient_payloads: list[LLMPayload] = []
            if extra_payload is not None:
                # 框架已允许 TOOL_RESULT → USER，extra_payload（USER 类型）可直接追加。
                transient_payloads.append(extra_payload)
            send_target = build_request_view(response, transient_payloads)
            if config.debug.show_prompt:
                self._log_prompt(send_target, chain_payloads=session.chain_payloads)

            if unread_msgs:
                known_ids: frozenset[str] = frozenset(
                    message.message_id
                    for message in unread_msgs
                    if message.message_id
                )
            else:
                _, current_snapshot = await self.fetch_unreads(
                    time_format="%Y-%m-%d %H:%M:%S"
                )
                known_ids = frozenset(
                    message.message_id
                    for message in current_snapshot
                    if message.message_id
                )

            try:
                phase = ConversationPhase.MODEL_TURN
                if config.buffer.interrupt_enabled and not transient_payloads:
                    new_response, interrupt_msgs = await self._send_interruptable(
                        response,
                        config,
                        known_ids,
                    )
                    if interrupt_msgs:
                        extra_payload = None
                        await self.flush_unreads(unread_msgs or [])
                        session.add_interrupt_event(interrupt_msgs)
                        await self._save_session(session)
                        continue
                    assert new_response is not None
                    response = new_response
                else:
                    watchdog = get_watchdog()
                    watchdog.feed_dog(self.stream_id)
                    response = await send_target.send(auto_append_response=True, stream=False)
                    watchdog.feed_dog(self.stream_id)
                    normalize_response(response)
                await self.flush_unreads(unread_msgs if unread_msgs else [])
            except Exception as exc:
                logger.error(f"LLM 请求失败: {exc}", exc_info=True)
                extra_payload = None
                # 失败路径必须与成功路径保持同样的 unread 消费契约：
                # 框架 LLM 层已在内部跑完 policy 重试与多模型 fallback，
                # 异常穿透到这里说明这批消息当下确实无法处理。若不消费 unread，
                # 框架下一 Tick 仍会拿同一批未读再次拉起 execute()，叠加
                # KFC 的"先持久化、后发送"时序，sessions/xxx.json 会被
                # 同一条触发消息反复 append 到 mental_log / chain_payloads。
                # 这里把 user 条目对应的 unread 搬入 history（已通过
                # update_chain 写入持久化链，仍可在下次上下文中被还原），
                # 把决定权交还给框架的正常 Tick 调度。
                if unread_msgs:
                    try:
                        await self.flush_unreads(unread_msgs)
                    except Exception as flush_exc:
                        logger.warning(
                            f"LLM 失败后 flush_unreads 失败（不影响主流程）: {flush_exc}"
                        )
                await self._save_session(session)
                yield Failure("LLM 请求失败", exc)
                break

            extra_payload = None
            phase = phase_for_model_result(response)

            _heal_orphan_tool_results(response, where="post-send")

            call_list = response.call_list or []

            # ── 纯文本重试机制 ──
            # 当 LLM 返回纯文本（无 tool call）时，注入提醒并重试一次
            if not call_list and plain_text_retry_count < _MAX_PLAIN_TEXT_RETRIES:
                raw_message = (response.message or "").strip()
                if raw_message:
                    plain_text_retry_count += 1
                    logger.info(
                        f"[KFC] LLM 返回纯文本（第 {plain_text_retry_count} 次），"
                        f"注入提醒后重试: {raw_message[:80]}"
                    )
                    # 框架已允许 TOOL_RESULT → USER，纯文本重试时直接追加 USER 提醒。
                    response.add_payload(
                        LLMPayload(ROLE.USER, Text(_PLAIN_TEXT_RETRY_REMINDER))
                    )
                    has_pending_tool_results = True
                    continue

            # 成功获得 tool call 时重置重试计数
            if call_list:
                plain_text_retry_count = 0
                logger.info(f"本轮调用列表：{[call.name for call in call_list]}")
            elif response.message:
                logger.debug("[KFC] 本轮无 tool call，进入决策判定")

            trigger_msg = unread_msgs[-1] if unread_msgs else None
            if trigger_msg is None:
                trigger_msg = await self._get_virtual_trigger_message()
            phase = ConversationPhase.TOOL_EXEC if call_list else ConversationPhase.COMMIT
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

            phase = ConversationPhase.COMMIT
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
                # 只有工具失败时才计入重试次数，正常的工具链不受影响
                if decision.has_failed_tool:
                    follow_up_count += 1
                    max_retries = config.general.max_follow_up_retries
                    if max_retries > 0 and follow_up_count > max_retries:
                        logger.warning(
                            f"[KFC] 工具失败重试次数已达上限 {max_retries}，"
                            "强制停止续轮（防止工具调用格式错误导致无限重试）"
                        )
                        follow_up_count = 0
                        has_pending_tool_results = False
                        yield Stop(0)
                        return
                has_pending_tool_results = True

            # assistant entry 已写入链，清空 pre_send_user_text 防止下一轮续轮重复持久化
            if turn_control.chain_assistant_saved:
                chain_user_pre_saved = True
                pre_send_user_text = ""

            if turn_control.next_signal is not None:
                yield turn_control.next_signal
            if turn_control.return_after_yield:
                return
            if turn_control.continue_loop:
                continue

    finally:
        if config.general.native_multimodal:
            self._unregister_vlm_skip()