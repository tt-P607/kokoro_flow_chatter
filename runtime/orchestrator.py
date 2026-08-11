"""KFC 对话主循环编排。

本模块只负责"按什么顺序做什么"，具体工作全部委托给同层的专职模块：

- ``model_setup``：解析模型集
- ``context_builder``：构建初始请求
- ``turn_controller``：准备回合输入、提交回合决策
- ``payload_hygiene``：发送前清理上下文链
- ``summary_sync``：热更新后台生成的记忆摘要
- ``input_status``：上报「正在输入」
- ``interrupt_controller``：可打断的 LLM 调用

一次 ``execute()`` 内的循环会持续到模型收口（Stop）或需要等待新消息
（Wait），期间维持同一条 ``response`` 链以累积上下文。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.chat_api import restore_stream_to_default
from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.stream_api import activate_stream
from src.app.plugin_system.base import Failure, Stop, Success, Wait
from src.app.plugin_system.types import LLMPayload, ROLE, Text
from src.kernel.concurrency import get_watchdog

from ..debug.log_formatter import log_kfc_result
from ..domain.chain_entry import ChainEntry
from ..execution import run_decision
from ..protocol.response_normalizer import normalize_response
from ..services import ProactiveService, TimeoutService
from ..snapshot import capture_snapshot
from .context_builder import build_initial_request
from .input_status import InputStatusReporter
from .model_setup import resolve_model_set
from .payload_hygiene import heal_orphan_tool_results, strip_stale_reminder_prefixes
from .request_view import build_request_view
from .summary_sync import SummarySynchronizer
from .turn_controller import commit_turn_decision, prepare_turn_input

if TYPE_CHECKING:
    from ..chatter import KokoroFlowChatter

logger = get_logger("kfc_orchestrator")

_PLAIN_TEXT_RETRY_REMINDER = (
    "（系统提示：你刚才返回了纯文本而非工具调用。"
    "请务必通过 action-kfc_reply 或 action-do_nothing 工具调用来完成响应，"
    "不要直接输出文字。）"
)
"""模型只输出正文、未调用任何工具时注入的纠正提示。"""

_INTERRUPT_COOLDOWN_GROWTH = 0.5
"""连续打断时冷却窗口的递增系数：第 N 次冷却为基准值的 ``1 + (N-1) * 0.5`` 倍。"""

_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


async def execute_orchestrator(
    chatter: KokoroFlowChatter,
) -> AsyncGenerator[Wait | Success | Failure | Stop, None]:
    """执行 KFC 对话主循环。

    Args:
        chatter: 当前 chatter 实例。

    Yields:
        Wait | Success | Failure | Stop: 交还给框架的循环控制信号。
    """
    chat_stream = await activate_stream(chatter.stream_id)
    if chat_stream is None:
        logger.error(f"无法激活聊天流: {chatter.stream_id}")
        yield Failure("聊天流激活失败")
        return

    config = chatter.get_config()
    if not config.general.enabled:
        logger.info("KFC 插件已禁用，解除 chatter 绑定以允许框架重新选择")
        restore_stream_to_default(chatter.stream_id)
        yield Success("KFC 插件已禁用")
        return

    model_set = resolve_model_set(config)
    if not model_set:
        logger.error("未找到有效的模型配置")
        yield Failure("模型配置错误：未找到有效的模型配置")
        return

    session = await chatter.load_session()
    timeout_service = TimeoutService(config)

    if config.general.native_multimodal:
        chatter.register_vlm_skip()

    try:
        response, usable_map = await build_initial_request(
            chatter, chat_stream, config, session, model_set
        )
        state = _LoopState(summary=SummarySynchronizer(session.history_summary))

        while True:
            state.summary.sync_if_changed(response, chat_stream, session.history_summary)
            heal_orphan_tool_results(response, where="loop-top")

            turn_input = await prepare_turn_input(
                chatter,
                response,
                chat_stream,
                config,
                session,
                timeout_service,
                state.has_pending_tool_results,
            )
            response = turn_input.response
            # 新消息到来意味着上一串工具续轮已结束，重置失败计数
            if state.has_pending_tool_results and not turn_input.has_pending_tool_results:
                state.follow_up_count = 0
            state.has_pending_tool_results = turn_input.has_pending_tool_results
            state.is_final_timeout = turn_input.is_final_timeout

            if turn_input.next_signal is not None:
                yield turn_input.next_signal
            if turn_input.continue_loop:
                continue

            unread_msgs = turn_input.unread_msgs
            if unread_msgs:
                state.last_user_ts = min(
                    chatter.extract_timestamp(message) for message in unread_msgs
                )

            _stage_user_chain_entry(state, turn_input.chain_text)
            if state.pending_user_text and not state.chain_user_saved:
                session.update_chain(
                    [
                        ChainEntry.user(
                            state.pending_user_text, ts=state.last_user_ts
                        ).to_dict()
                    ],
                    config.prompt.max_context_payloads,
                )
                await chatter.save_session(session)
                state.chain_user_saved = True

            strip_stale_reminder_prefixes(response)
            transient_payloads = (
                [turn_input.extra_payload] if turn_input.extra_payload else []
            )
            send_target = build_request_view(response, transient_payloads)
            if config.debug.show_prompt:
                chatter.log_prompt(send_target, session.chain_payloads)

            known_ids = await _resolve_known_message_ids(chatter, unread_msgs)
            reporter = InputStatusReporter(chatter.stream_id, session.user_id)
            should_report = config.general.enable_input_status

            try:
                if should_report:
                    await reporter.start()

                sent_response, interrupt_msgs = await _send_llm_request(
                    chatter, send_target, config, known_ids, state
                )
                if interrupt_msgs:
                    await chatter.flush_unreads(unread_msgs)
                    session.add_interrupt_event(interrupt_msgs)
                    await chatter.save_session(session)
                    await _wait_interrupt_cooldown(config, state)
                    continue

                response = sent_response
                await chatter.flush_unreads(unread_msgs)
            except Exception as error:
                logger.error(f"LLM 请求失败: {error}", exc_info=True)
                # 失败路径必须与成功路径保持相同的 unread 消费契约：框架 LLM
                # 层已跑完重试与多模型 fallback，异常穿透至此说明这批消息当下
                # 确实无法处理。不消费就会在下一 Tick 拿到同一批未读重新拉起
                # execute()，叠加"先持久化后发送"的时序，同一条消息会被反复
                # 写入活动流与对话链。
                await chatter.flush_unreads(unread_msgs)
                await chatter.save_session(session)
                yield Failure("LLM 请求失败", error)
                return
            finally:
                if should_report:
                    await reporter.stop()

            heal_orphan_tool_results(response, where="post-send")

            if not response.call_list:
                # 每次纯文本纠正重试都会重新走完整的多模型 fallback，因此
                # 上限按"每个模型各 max_follow_up_retries 次"放大，避免多个
                # 模型共享同一份总重试额度而提前收口。
                model_count = len(model_set) if isinstance(model_set, list) else 1
                max_retries = config.general.max_follow_up_retries * model_count
                if state.plain_text_retry_count < max_retries:
                    _log_missing_tool_call(response, state)
                    state.plain_text_retry_count += 1
                    response.add_payload(
                        LLMPayload(ROLE.USER, Text(_PLAIN_TEXT_RETRY_REMINDER))
                    )
                    state.has_pending_tool_results = True
                    continue
                logger.warning(
                    f"经过 {state.plain_text_retry_count} 次重试仍未取得有效工具调用，"
                    "本轮强制收口"
                )
            else:
                state.plain_text_retry_count = 0
                logger.info(f"本轮调用列表：{[call.name for call in response.call_list]}")

            trigger_msg = unread_msgs[-1] if unread_msgs else None
            if trigger_msg is None:
                trigger_msg = await chatter.build_virtual_trigger_message()

            decision = await run_decision(
                response,
                usable_map,
                trigger_msg,
                config,
                execute_reply_fn=chatter.send_reply,
                run_tool_call_fn=chatter.run_tool_call,
                pre_execute_hook=lambda result: log_kfc_result(result, config),
            )
            if decision.proactive_schedule is not None:
                ProactiveService.apply_schedule(session, decision.proactive_schedule)

            turn_control = await commit_turn_decision(
                chatter,
                decision,
                response,
                session,
                config,
                chat_stream,
                pending_user_text=state.pending_user_text,
                last_user_ts=state.last_user_ts,
                chain_user_saved=state.chain_user_saved,
                is_final_timeout=state.is_final_timeout,
            )
            state.is_final_timeout = turn_control.is_final_timeout

            if turn_control.has_pending_tool_results:
                if decision.has_failed_tool and _exceeded_retry_limit(config, state):
                    yield Stop(0)
                    return
                state.has_pending_tool_results = True

            # 回合闭合点：无待消化工具结果时，主链已含本轮完整 bot 输出
            # （推理 + 正文 + 工具调用 + 回执）且工具段必然闭合，捕获无损快照。
            if not turn_control.has_pending_tool_results:
                snapshot = capture_snapshot(
                    response.payloads, config.prompt.max_context_payloads
                )
                if snapshot is not None:
                    session.context_snapshot = snapshot
                    await chatter.save_session(session)

            if turn_control.chain_assistant_saved:
                # assistant 已入链，清空暂存避免续轮时重复持久化 user 条目
                state.chain_user_saved = True
                state.pending_user_text = ""

            if turn_control.next_signal is not None:
                yield turn_control.next_signal
            if turn_control.return_after_yield:
                return
    finally:
        if config.general.native_multimodal:
            chatter.unregister_vlm_skip()


class _LoopState:
    """主循环的可变状态。

    集中承载跨轮次传递的计数与暂存文本，避免在循环体内散落大量
    同生命周期的局部变量。
    """

    __slots__ = (
        "chain_user_saved",
        "consecutive_interrupt_count",
        "follow_up_count",
        "has_pending_tool_results",
        "is_final_timeout",
        "last_user_ts",
        "pending_user_text",
        "plain_text_retry_count",
        "summary",
    )

    def __init__(self, summary: SummarySynchronizer) -> None:
        """初始化循环状态。"""
        self.summary = summary
        self.has_pending_tool_results = False
        self.is_final_timeout = False
        self.pending_user_text = ""
        self.last_user_ts = 0.0
        self.chain_user_saved = False
        self.plain_text_retry_count = 0
        self.follow_up_count = 0
        self.consecutive_interrupt_count = 0


def _stage_user_chain_entry(state: _LoopState, chain_text: str) -> None:
    """暂存本轮待入链的用户文本。

    ``chain_text`` 只含原始消息内容，不含末尾强调指令与临时注入，
    避免这些每轮变化的提示词被固化进持久化对话链。
    """
    if not chain_text or chain_text == state.pending_user_text:
        return
    state.pending_user_text = chain_text
    state.chain_user_saved = False


async def _resolve_known_message_ids(
    chatter: KokoroFlowChatter,
    unread_msgs: list[Any],
) -> frozenset[str]:
    """确定打断检测的基线消息 ID 集合。

    本轮已纳入上下文的消息不应触发打断；无未读时需要重新快照当前
    未读队列，否则打断检测会把既有消息误判为新消息。
    """
    if unread_msgs:
        return frozenset(
            message.message_id for message in unread_msgs if message.message_id
        )
    _, snapshot = await chatter.fetch_unreads(time_format=_TIME_FORMAT)
    return frozenset(message.message_id for message in snapshot if message.message_id)


async def _send_llm_request(
    chatter: KokoroFlowChatter,
    send_target: Any,
    config: Any,
    known_ids: frozenset[str],
    state: _LoopState,
) -> tuple[Any, list[Any]]:
    """发送 LLM 请求，按配置决定是否允许打断。

    Returns:
        tuple: ``(响应, 打断消息列表)``；被打断时响应为 ``None``。
    """
    max_interrupts = config.buffer.max_consecutive_interrupts
    interrupt_allowed = (
        config.buffer.interrupt_enabled
        and state.consecutive_interrupt_count < max_interrupts
    )

    if interrupt_allowed:
        response, interrupt_msgs = await chatter.send_interruptable(
            send_target, config, known_ids
        )
        if interrupt_msgs:
            state.consecutive_interrupt_count += 1
            return None, interrupt_msgs
        state.consecutive_interrupt_count = 0
        return response, []

    if state.consecutive_interrupt_count >= max_interrupts:
        logger.warning(
            f"连续打断已达上限 {max_interrupts}，本次不再打断，"
            "等待 LLM 正常完成后统一处理"
        )

    watchdog = get_watchdog()
    watchdog.feed_dog(chatter.stream_id)
    response = await send_target.send(auto_append_response=True, stream=False)
    watchdog.feed_dog(chatter.stream_id)
    normalize_response(response)
    state.consecutive_interrupt_count = 0
    return response, []


async def _wait_interrupt_cooldown(config: Any, state: _LoopState) -> None:
    """打断后等待冷却窗口，收集可能连发的后续消息。

    连续打断时冷却时间递增，避免高频消息把 LLM 调用拖入无限重启。
    """
    base_cooldown = config.buffer.interrupt_cooldown
    growth = 1.0 + (state.consecutive_interrupt_count - 1) * _INTERRUPT_COOLDOWN_GROWTH
    cooldown = base_cooldown * growth
    if cooldown <= 0:
        return
    logger.debug(
        f"打断后冷却 {cooldown:.1f}s"
        f"（连续打断 {state.consecutive_interrupt_count}/"
        f"{config.buffer.max_consecutive_interrupts}）"
    )
    await asyncio.sleep(cooldown)


def _log_missing_tool_call(response: Any, state: _LoopState) -> None:
    """记录模型未返回工具调用的情形。"""
    attempt = state.plain_text_retry_count + 1
    raw_message = (response.message or "").strip()
    if raw_message:
        logger.info(f"LLM 返回纯文本（第 {attempt} 次），注入提醒后重试: {raw_message[:80]}")
    else:
        logger.warning(f"LLM 返回空响应（第 {attempt} 次），注入提醒后重试")


def _exceeded_retry_limit(config: Any, state: _LoopState) -> bool:
    """累计工具失败次数并判断是否超出续轮上限。

    只有工具执行失败才计数——正常的多轮工具链不应受此限制。
    """
    state.follow_up_count += 1
    max_retries = config.general.max_follow_up_retries
    if max_retries <= 0 or state.follow_up_count <= max_retries:
        return False

    logger.warning(
        f"工具失败重试次数已达上限 {max_retries}，强制停止续轮"
        "（防止工具调用格式错误导致无限重试）"
    )
    state.follow_up_count = 0
    state.has_pending_tool_results = False
    return True
