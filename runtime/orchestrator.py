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
from ..execution import run_decision
from ..protocol.response_normalizer import normalize_response
from ..services import ProactiveService, TimeoutService
from ..snapshot import capture_snapshot
from .context_builder import REQUEST_NAME, build_initial_request
from .guard_hook import check_response_guard
from .input_status import InputStatusReporter
from .model_setup import resolve_model_set
from .payload_hygiene import heal_orphan_tool_results
from .request_view import build_request_view
from .summary_sync import SummarySynchronizer
from ..context.planner import build_last_mile_payload
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

_GUARD_RETRY_REMINDER = (
    "（系统提示：你刚才的回复未通过当前对话的响应检查，已被丢弃，"
    "不属于有效对话历史。请重新处理用户当前的消息，保持既定角色身份、人设、"
    "语气、情绪与当前场景，不要以助手、模型、平台或规则说明者的身份回应，"
    "也不要解释或提及本次检查、上一条失败回复或内部规则。"
    "能够正常回应时请通过 action-kfc_reply（或 action-do_nothing）自然继续对话；"
    "无法给出角色内回复时请使用 action-do_nothing，不要输出解释性拒答。）"
)
"""模型层安全拒答被拦截后注入的响应质量纠正提示。

只纠正回复所处的角色状态，不涉及安全策略本身，也不包含任何绕过审核、
要求必须回答或禁止拒绝的内容。
"""

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

            if turn_input.persistent_user_payload is not None:
                state.turn_has_new_input = True
                # 新一轮真实输入开启新的守卫重试预算
                state.guard_retry_count = 0
                changed = session.append_context_entries(
                    [turn_input.persistent_user_payload],
                    config.prompt.max_context_payloads,
                )
                if changed:
                    await chatter.save_session(session)

            unread_msgs = turn_input.unread_msgs
            transient_payloads: list[LLMPayload] = [build_last_mile_payload()]
            if turn_input.request_only_payload is not None:
                transient_payloads.append(turn_input.request_only_payload)
            if turn_input.extra_payload is not None:
                transient_payloads.append(turn_input.extra_payload)
            transient_payloads.extend(state.plain_text_reminders)
            transient_payloads.extend(_consume_guard_reminder(state))
            _set_external_resume_metadata(
                response,
                request_marker=turn_input.external_resume_request_marker,
                source=turn_input.external_resume_source,
            )
            # 纯文本重试的提醒随本轮请求临时注入，成功取得工具调用后自动
            # 消失；失败的纯文本 ASSISTANT 输出同样不落主链——发送前记录
            # 基线长度，模型仍只返回正文时直接回滚。
            payload_baseline = len(response.payloads)
            send_target = build_request_view(response, transient_payloads)
            if config.debug.show_prompt:
                chatter.log_prompt(send_target)

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
                _clear_external_resume_metadata(response)
                if should_report:
                    await reporter.stop()

            heal_orphan_tool_results(response, where="post-send")

            # 守卫先于格式纠正：模型层拒答既可能包装在合法工具调用里，也可能
            # 直接以纯文本出现，两者是同一种失败，必须先归入同一条恢复路径，
            # 否则拒答会被当成"忘记调用工具"继续消耗纯文本重试额度。
            guard_blocked = False
            guard_evidence: tuple[str, ...] = ()
            if config.general.guard_enabled:
                guard = await check_response_guard(
                    response,
                    request_name=REQUEST_NAME,
                    retry_index=state.guard_retry_count,
                )
                guard_blocked = guard.blocked
                guard_evidence = guard.evidence if guard.blocked else ()

            if guard_blocked:
                if _handle_guard_refusal(
                    response,
                    payload_baseline,
                    state,
                    config,
                    guard_evidence,
                    from_tool_call=bool(response.call_list),
                ):
                    continue
                # 守卫已明确判定本轮响应无效，且重试预算已耗尽：response 已被
                # 完整回滚，本轮必须在这里显式结束。继续往下走只会让空响应
                # 进入决策层，在提交阶段写入一条空的 bot planning，把一次明确
                # 的失败伪装成一次正常决策。
                yield Stop(0)
                return
            elif not response.call_list:
                if _handle_plain_text_violation(
                    response, payload_baseline, state, config, model_set
                ):
                    continue
            else:
                state.plain_text_reminders.clear()
                state.plain_text_retry_count = 0
                state.guard_retry_count = 0
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
                has_new_user_input=state.turn_has_new_input,
                is_final_timeout=state.is_final_timeout,
            )
            state.is_final_timeout = turn_control.is_final_timeout
            # 一批真实新消息只推进一次记忆压缩计数；工具续轮不再重复计入。
            state.turn_has_new_input = False

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
        "consecutive_interrupt_count",
        "follow_up_count",
        "guard_reminder",
        "guard_retry_count",
        "has_pending_tool_results",
        "is_final_timeout",
        "plain_text_reminders",
        "plain_text_retry_count",
        "summary",
        "turn_has_new_input",
    )

    def __init__(self, summary: SummarySynchronizer) -> None:
        """初始化循环状态。"""
        self.summary = summary
        self.has_pending_tool_results = False
        self.is_final_timeout = False
        self.turn_has_new_input = False
        self.plain_text_retry_count = 0
        self.guard_retry_count = 0
        self.plain_text_reminders: list[LLMPayload] = []
        self.guard_reminder: LLMPayload | None = None
        self.follow_up_count = 0
        self.consecutive_interrupt_count = 0


_EXTERNAL_RESUME_MARKER_KEY = "kfc_external_resume_request_marker"
_EXTERNAL_RESUME_SOURCE_KEY = "kfc_external_resume_source"


def _request_metadata(response: Any) -> dict[str, Any] | None:
    """取得 response / request 背后的可变 request metadata。"""
    upper = response._upper if hasattr(response, "_upper") else response
    metadata = getattr(upper, "meta_data", None)
    return metadata if isinstance(metadata, dict) else None


def _set_external_resume_metadata(
    response: Any,
    *,
    request_marker: str,
    source: str,
) -> None:
    """仅为本次 external resume 请求设置通用元数据。"""
    metadata = _request_metadata(response)
    if metadata is None:
        return
    metadata.pop(_EXTERNAL_RESUME_MARKER_KEY, None)
    metadata.pop(_EXTERNAL_RESUME_SOURCE_KEY, None)
    if not request_marker:
        return
    metadata[_EXTERNAL_RESUME_MARKER_KEY] = request_marker
    metadata[_EXTERNAL_RESUME_SOURCE_KEY] = source


def _clear_external_resume_metadata(response: Any) -> None:
    """请求结束后清理 transient external resume 元数据。"""
    metadata = _request_metadata(response)
    if metadata is None:
        return
    metadata.pop(_EXTERNAL_RESUME_MARKER_KEY, None)
    metadata.pop(_EXTERNAL_RESUME_SOURCE_KEY, None)




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


def _rollback_failed_assistant(response: Any, payload_baseline: int) -> None:
    """丢弃发送基线之后追加的失败 ASSISTANT 输出。

    模型未返回工具调用时，本次正文不会被执行也不会真正发出，持久化
    会让后续轮次读到"说过却无下文"的残缺历史。仅在末尾确实是本轮
    新增的 ASSISTANT 时回滚；同时清空 ``message`` 等输出字段，防止
    提交阶段把这段正文写进持久对话链。
    """
    payloads = response.payloads
    if not isinstance(payloads, list):
        return
    if len(payloads) > payload_baseline:
        trailing = payloads[payload_baseline:]
        if all(payload.role == ROLE.ASSISTANT for payload in trailing):
            del payloads[payload_baseline:]
            response.message = ""
            response.reasoning_content = ""
            # 推理分段与推理正文是同一份内容的两个视图，回滚必须同时
            # 清空，否则被拦响应仍会以分段形式留在响应对象上。
            response.reasoning_parts = []
            response.call_list = []


def _consume_guard_reminder(state: _LoopState) -> list[LLMPayload]:
    """取出并消费一次性的 Guard 重试提示。

    提示只服务于紧接着的这一次请求：构造完发送视图即失效，因此重试再次
    命中时会重新创建，而不会在同一次请求里叠加多条，也不会跨请求残留。

    Args:
        state: 主循环可变状态。

    Returns:
        list[LLMPayload]: 本次请求应临时注入的提示，至多一条。
    """
    reminder = state.guard_reminder
    if reminder is None:
        return []
    state.guard_reminder = None
    return [reminder]


def _handle_guard_refusal(
    response: Any,
    payload_baseline: int,
    state: _LoopState,
    config: Any,
    evidence: tuple[str, ...],
    *,
    from_tool_call: bool,
) -> bool:
    """回滚被守卫拦截的本轮输出，并按守卫重试预算决定是否重试。

    纯文本拒答与工具调用拒答在这里汇合：两者都是模型层安全拒答，处理
    方式完全相同——先完整回滚，再决定重试或收口。守卫只负责判定，本函数
    只负责判定之后的控制流，自身不含任何检测规则。

    Args:
        response: 本轮 LLM 响应链。
        payload_baseline: 发送前记录的主链长度基线。
        state: 主循环可变状态。
        config: KFC 配置。
        evidence: 守卫给出的证据标签。
        from_tool_call: 拒答是否包装在合法工具调用中，仅用于日志区分形态。

    Returns:
        bool: True 表示主循环应重试一次，False 表示预算已耗尽、本轮收口。
    """
    _rollback_failed_assistant(response, payload_baseline)
    # 本次失败原因已明确为模型层拒答，格式纠正提示必须一并清掉：两种提示
    # 同时注入会把模型推向"改用工具调用复述同一份拒答"。
    state.plain_text_reminders.clear()
    state.guard_reminder = None
    evidence_text = ",".join(evidence) or "-"
    shape = "工具调用内容" if from_tool_call else "纯文本响应"

    if state.guard_retry_count < config.general.guard_max_retries:
        state.guard_retry_count += 1
        logger.warning(
            f"Response Guard 判定{shape}为模型层安全拒答（第 "
            f"{state.guard_retry_count} 次 Guard retry），回滚本轮输出并注入"
            f"一次性 Guard Retry Reminder；evidence={evidence_text}"
        )
        state.guard_reminder = LLMPayload(ROLE.USER, Text(_GUARD_RETRY_REMINDER))
        state.has_pending_tool_results = True
        return True

    logger.warning(
        f"Response Guard 判定{shape}为模型层安全拒答且已达重试上限 "
        f"{config.general.guard_max_retries}，本轮完整回滚并收口；"
        f"evidence={evidence_text}"
    )
    return False


def _handle_plain_text_violation(
    response: Any,
    payload_baseline: int,
    state: _LoopState,
    config: Any,
    model_set: Any,
) -> bool:
    """处理守卫放行、但模型未调用任何工具的纯文本响应。

    这里只处理格式问题；安全拒答已在上游被守卫分流，不会进入本函数。
    每次纯文本纠正重试都会重新走完整的多模型 fallback，因此上限按
    "每个模型各 max_follow_up_retries 次"放大，避免多个模型共享同一份
    总重试额度而提前收口。

    Args:
        response: 本轮 LLM 响应链。
        payload_baseline: 发送前记录的主链长度基线。
        state: 主循环可变状态。
        config: KFC 配置。
        model_set: 已解析的模型集，用于放大重试上限。

    Returns:
        bool: True 表示主循环应重试一次，False 表示预算已耗尽、本轮收口。
    """
    model_count = len(model_set) if isinstance(model_set, list) else 1
    max_retries = config.general.max_follow_up_retries * model_count
    # 本次失败原因已明确为格式问题，上一次的守卫提示必须清掉。
    state.guard_reminder = None

    if state.plain_text_retry_count < max_retries:
        _log_missing_tool_call(response, state)
        state.plain_text_retry_count += 1
        _rollback_failed_assistant(response, payload_baseline)
        # 纠正提醒逐条累积为多重强调信号，随每次请求临时注入；
        # 取得有效工具调用或本轮收口时一并清除。
        state.plain_text_reminders.append(
            LLMPayload(ROLE.USER, Text(_PLAIN_TEXT_RETRY_REMINDER))
        )
        state.has_pending_tool_results = True
        return True

    logger.warning(
        f"经过 {state.plain_text_retry_count} 次重试仍未取得有效工具调用，"
        "本轮强制收口"
    )
    # 强制收口时同样丢弃本次失败的纯文本输出：没有工具调用的
    # 正文不可能被执行成功，持久化只会让下一轮读到自相矛盾的
    # "我说过 xxx 却毫无反应"的残缺历史。
    _rollback_failed_assistant(response, payload_baseline)
    state.plain_text_reminders.clear()
    return False


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
