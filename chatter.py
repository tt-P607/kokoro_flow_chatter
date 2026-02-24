"""KokoroFlowChatter 核心聊天器。

实现完整的心理活动流对话循环：
1. 构建 LLM 上下文（系统提示 + 活动流 + 未读消息）
2. 维护 LLMResponse 链（response = request → loop）
3. 通过原生 Tool Calling 执行动作
4. 管理等待状态
5. 超时后重新注入上下文继续对话

严格遵循 DefaultChatter._execute_enhanced() 的 response 链模式。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, AsyncGenerator

from src.app.plugin_system.api.llm_api import (
    create_llm_request,
    get_model_set_by_task,
)
from src.app.plugin_system.api.log_api import get_logger
from src.core.components.base import (
    BaseChatter,
    Failure,
    Stop,
    Success,
    Wait,
)
from src.core.components.types import ChatType
from src.kernel.concurrency import get_watchdog
from src.kernel.llm import Content, LLMContextManager, LLMPayload, ROLE, Text, ToolResult

from .debug.log_formatter import format_prompt_for_log, log_kfc_result
from .models import KFC_REPLY, DO_NOTHING, ToolCallResult
from .parser import parse_tool_calls
from .prompts.templates import KFC_PERCEIVE_FOLLOWUP_PROMPT

if TYPE_CHECKING:
    from src.core.models.stream import ChatStream
    from src.kernel.llm import ToolRegistry

    from .config import KFCConfig
    from .prompts.builder import KFCPromptBuilder
    from .session import KFCSession, KFCSessionStore

logger = get_logger("kfc_chatter")



class KokoroFlowChatter(BaseChatter):
    """KokoroFlowChatter 核心聊天器。

    基于心理活动流的对话模型：
    - 维护 LLMResponse 链贯穿整个 execute() 生命周期
    - 通过原生 Tool Calling 注入工具并解析响应
    - 活动流为持久化审计日志，LLM 上下文通过 response 链自动积累
    """

    chatter_name: str = "kokoro_flow_chatter"
    chatter_description: str = (
        "心理活动流聊天器，模拟真实人类的连续心理活动和对话节奏"
    )

    associated_platforms: list[str] = []
    chat_type: ChatType = ChatType.PRIVATE
    dependencies: list[str] = []

    # ── 配置与会话辅助 ──────────────────────────────────────

    def _get_config(self) -> KFCConfig:
        """获取 KFC 配置。"""
        from .config import KFCConfig

        plugin_config = getattr(self.plugin, "config", None)
        if plugin_config and isinstance(plugin_config, KFCConfig):
            return plugin_config
        return KFCConfig()

    async def _get_session(self) -> KFCSession:
        """获取当前 stream 的 Session（持有 per-stream 锁）。"""
        session_store = self._get_session_store()
        async with session_store.lock(self.stream_id):
            return await session_store.get_or_create(self.stream_id)

    def _get_session_store(self) -> KFCSessionStore:
        """获取 Session Store（由 plugin.__init__ 初始化）。"""
        return self.plugin._session_store  # type: ignore[attr-defined]

    # ── 核心对话循环 ──────────────────────────────────────────

    async def execute(self) -> AsyncGenerator[Wait | Success | Failure | Stop, None]:
        """执行聊天器的对话循环。

        核心流程：
        1. 初始化 LLM 请求（系统提示 + 原生工具注册）
        2. response = request（建立 response 链）
        3. 循环：读取消息 → 构建 payload → LLM 调用 → 解析结果 → 执行动作
        4. 等待超时后注入超时 payload 继续 response 链

        Yields:
            Wait | Success | Failure | Stop: 执行结果
        """
        from src.core.managers.stream_manager import get_stream_manager

        # ── 初始化 ──
        stream_manager = get_stream_manager()
        chat_stream = await stream_manager.activate_stream(self.stream_id)
        config = self._get_config()

        # 检查插件是否启用
        if not config.general.enabled:
            logger.debug("KFC 插件已禁用，跳过 execute")
            yield Stop(0)
            return

        session = await self._get_session()

        # ── 注册 VLM 跳过（原生多模态模式） ──
        vlm_registered = False
        if config.general.native_multimodal:
            self._register_vlm_skip()
            vlm_registered = True

        try:  # 确保退出时清理 VLM 跳过注册
            # ── 构建 LLM 请求 ──
            model_set = get_model_set_by_task(config.general.model_task)
            if not model_set:
                logger.error("无法获取模型配置")
                yield Failure("模型配置错误：未找到 model_task 配置")
                return

            context_manager = LLMContextManager(
                max_payloads=config.prompt.max_context_payloads
            )
            request = create_llm_request(
                model_set,
                "kokoro_flow_chatter",
                context_manager=context_manager,
            )

            # 系统提示词
            from .prompts.builder import KFCPromptBuilder

            prompt_builder = KFCPromptBuilder()
            system_prompt = prompt_builder.build_system_prompt(chat_stream)
            request.add_payload(LLMPayload(ROLE.SYSTEM, Text(system_prompt)))

            # 历史消息 + 内心独白融合叙事
            history_text = prompt_builder.build_fused_narrative(
                chat_stream, session.mental_log
            )
            # 图片预算：历史图片 + 当前轮图片共用同一配额
            image_budget: ImageBudget | None = None
            if config.general.native_multimodal:
                from .multimodal import ImageBudget

                image_budget = ImageBudget(config.general.max_images_per_payload)

            if history_text:
                history_content: Content | list[Content] = Text(history_text)
                if image_budget and not image_budget.is_exhausted():
                    from .multimodal import (
                        build_multimodal_content,
                        extract_media_from_messages,
                    )

                    history_msgs = getattr(
                        getattr(chat_stream, "context", None), "history_messages", []
                    )
                    if history_msgs:
                        history_media = extract_media_from_messages(
                            history_msgs[-20:],
                            max_items=image_budget.remaining,
                        )
                        if history_media:
                            image_budget.consume(len(history_media))
                            history_content = build_multimodal_content(
                                history_text, history_media
                            )
                            logger.debug(
                                f" 历史多模态: 提取到 {len(history_media)} 张图片/表情包"
                                f" (剩余配额 {image_budget.remaining})"
                            )
                request.add_payload(LLMPayload(ROLE.USER, history_content))

            # ── 注册工具（原生 Tool Calling） ──
            usable_map = await self.inject_usables(request)

            # ── response = request（建立 response 链）──
            response = request

            # ── 对话循环 ──
            while True:
                # 读取未读消息
                formatted_text, unread_msgs = await self.fetch_unreads()

                if formatted_text and unread_msgs:
                    # 记录用户消息到活动流
                    for msg in unread_msgs:
                        sender_id = getattr(msg, "sender_id", "")
                        session.add_user_message(
                            content=getattr(msg, "processed_plain_text", "") or str(
                                getattr(msg, "content", "")
                            ),
                            user_name=getattr(msg, "sender_name", "用户"),
                            user_id=sender_id,
                            timestamp=self._extract_timestamp(msg),
                        )
                        # 更新 session 中的用户信息（每次都取最新的）
                        if sender_id:
                            session.user_id = sender_id
                        if chat_stream.platform:
                            session.platform = chat_stream.platform

                    # 检查等待期间收到回复的时效
                    if session.is_waiting():
                        self._record_reply_timing(session)
                        session.clear_waiting()

                    # 多模态：提取未读消息中的图片数据（共享图片预算）
                    media_items = self._extract_media(
                        unread_msgs, config, image_budget
                    )

                    # 构建 user payload（委托给 PromptBuilder）
                    mental_summary = session.mental_log.format_as_summary(
                        max_entries=config.prompt.max_log_entries
                    )
                    user_payload = prompt_builder.build_user_payload(
                        formatted_unreads=formatted_text,
                        mental_log_summary=mental_summary,
                        media_items=media_items,
                    )
                    response.add_payload(user_payload)

                elif session.is_waiting():
                    # 正在等待中，检查是否有超时
                    from .thinker.timeout_handler import TimeoutHandler

                    timeout_handler = TimeoutHandler(config)
                    if timeout_handler.check_timeout(session):
                        timeout_ctx = timeout_handler.handle_timeout(session)

                        if timeout_handler.should_give_up(session):
                            logger.info("连续超时次数过多，结束对话")
                            await self._save_session(session)
                            yield Stop(0)
                            return

                        # 构建超时 payload（委托给 PromptBuilder）
                        timeout_payload = KFCPromptBuilder.build_timeout_payload(
                            elapsed_seconds=timeout_ctx["elapsed_seconds"],  # type: ignore[arg-type]
                            expected_reaction=timeout_ctx["expected_reaction"],  # type: ignore[arg-type]
                            consecutive_timeouts=timeout_ctx["consecutive_timeouts"],  # type: ignore[arg-type]
                            last_bot_message=timeout_ctx.get("last_bot_message", ""),  # type: ignore[arg-type]
                            pending_thoughts=timeout_ctx.get("pending_thoughts"),  # type: ignore[arg-type]
                        )
                        response.add_payload(timeout_payload)
                    else:
                        yield Wait(0)
                        continue
                else:
                    yield Wait(0)
                    continue

                # ── 调用 LLM（两阶段感知-决策循环） ──
                if config.debug.show_prompt:
                    self._log_prompt(response)

                try:
                    response = await self._send_with_perceive_loop(
                        response, config.general.max_compat_retries
                    )
                    await self.flush_unreads(unread_msgs if unread_msgs else [])
                except Exception as e:
                    logger.error(f"LLM 请求失败: {e}", exc_info=True)
                    yield Failure("LLM 请求失败", e)
                    continue

                # ── 解析 + 执行 ──
                trigger_msg = unread_msgs[-1] if unread_msgs else None
                result = await parse_tool_calls(
                    response, usable_map, trigger_msg, config,
                    execute_reply_fn=self._execute_reply,
                    run_tool_call_fn=self.run_tool_call,
                )

                # 日志与活动流记录
                log_kfc_result(result, config)
                session.add_bot_planning(
                    thought=result.thought,
                    actions=result.actions,
                    expected_reaction=result.expected_reaction,
                    max_wait_seconds=result.max_wait_seconds,
                )

                # ── 控制流决策 ──
                if not result.has_meaningful_action:
                    if response.message and response.message.strip():
                        logger.warning(
                            f"LLM 返回无法解析的内容: {response.message[:100]}"
                        )
                    yield Stop(0)
                    await self._save_session(session)
                    return

                if result.has_do_nothing and not result.has_reply:
                    logger.debug("do_nothing，跳过本轮")
                    yield Stop(0)
                    await self._save_session(session)
                    return

                # ── 等待控制 ──
                wait_seconds = config.wait.apply_rules(
                    result.max_wait_seconds,
                    session.consecutive_timeout_count,
                )

                if wait_seconds > 0:
                    from .models import WaitingConfig

                    waiting_config = WaitingConfig(
                        expected_reaction=result.expected_reaction,
                        max_wait_seconds=wait_seconds,
                        started_at=time.time(),
                    )
                    session.set_waiting(waiting_config)
                    session.pending_thoughts.clear()
                    await self._save_session(session)
                    yield Wait(0)
                    continue

                # max_wait_seconds <= 0 → 话题结束
                session.clear_waiting()
                await self._save_session(session)
                yield Stop(0)
                return

        finally:
            if vlm_registered:
                self._unregister_vlm_skip()

    # ── 两阶段感知-决策循环 ──────────────────────────────────

    async def _send_with_perceive_loop(
        self,
        response: Any,
        max_retries: int,
    ) -> Any:
        """发送 LLM 请求，实现两阶段"感知→决策"循环。

        当模型收到图片后"破防"——输出纯自然语言感言而非 JSON 工具调用时，
        不将其视为错误，而是利用 auto_append_response=True 让这段感言
        自动追加到上下文中（记忆固化），然后注入轻量提示再次发送。
        第二次发送时模型上下文已包含自己的观察结论，无新图片干扰，
        能够正常输出结构化的工具调用。

        流程:
            1. send(auto_append_response=True) → 模型可能输出纯文本
            2. 检查 call_list 是否为空
            3. 若为空且有文本内容 → 感知阶段完成，注入跟进提示
            4. 再次 send() → 模型基于已有记忆输出工具调用

        Args:
            response: LLM 请求/响应链对象（LLMRequest 或 LLMResponse）
            max_retries: 最大感知-决策循环次数（0 表示不做二次发送）

        Returns:
            已消费（await）的 LLMResponse 对象
        """
        watchdog = get_watchdog()

        for attempt in range(max_retries + 1):
            # 喂狗：LLM 请求前刷新心跳，防止长时间阻塞触发 WatchDog 重启
            watchdog.feed_dog(self.stream_id)

            # auto_append_response=True：模型输出自动追加到上下文
            new_response = await response.send(
                auto_append_response=True, stream=False
            )
            await new_response

            # LLM 请求完成后再次喂狗
            watchdog.feed_dog(self.stream_id)

            # 模型成功输出了工具调用 → 直接返回
            if new_response.call_list:
                return new_response

            # 模型输出了纯文本但没有工具调用（"破防"）
            if attempt < max_retries:
                perceive_text = (new_response.message or "").strip()
                logger.info(
                    f"模型感知阶段输出纯文本，进入决策阶段 "
                    f"(第 {attempt + 1} 轮): "
                    f"{perceive_text[:80]}{'...' if len(perceive_text) > 80 else ''}"
                )
                # 感言已通过 auto_append 进入上下文，
                # 注入轻量提示引导模型进入决策阶段
                new_response.add_payload(
                    LLMPayload(ROLE.USER, Text(KFC_PERCEIVE_FOLLOWUP_PROMPT))
                )
                response = new_response
                continue

            # 重试次数耗尽，返回最后一次响应（由调用方处理空 call_list）
            return new_response

    # ── 动作执行 ────────────────────────────────────────────

    async def _execute_reply(
        self,
        content: str,
        config: KFCConfig,
        trigger_msg: Any | None = None,
    ) -> None:
        """通过框架标准路径发送回复。

        Args:
            content: 回复文本内容
            config: KFC 配置
            trigger_msg: 触发消息，为 None 时构造虚拟消息
        """
        from .actions.reply import KFCReplyAction

        if trigger_msg is None:
            trigger_msg = await self._get_virtual_trigger_message()
            if trigger_msg is None:
                logger.warning("无触发消息，无法发送回复")
                return

        try:
            await self.exec_llm_usable(KFCReplyAction, trigger_msg, content=content)
        except Exception as e:
            logger.error(f"通过框架执行 KFCReplyAction 失败: {e}", exc_info=True)

    # ── 辅助方法 ────────────────────────────────────────────

    def _register_vlm_skip(self) -> None:
        """为当前聊天流注册 VLM 跳过。

        在 native_multimodal 模式下，KFC 直接将原始图片数据打包进
        LLM payload，由主模型理解图片内容。框架的 VLM 管线会将图片
        转述为文本描述，这对 KFC 是冗余操作。

        此方法在 execute() 开头调用，确保后续到达的消息不再触发 VLM。
        调用是幂等的——多次注册同一 stream_id 不会产生副作用。
        """
        try:
            from src.core.managers import get_media_manager

            get_media_manager().skip_vlm_for_stream(self.stream_id)
        except Exception as e:
            logger.debug(f"注册 VLM 跳过失败（不影响功能）: {e}")

    def _unregister_vlm_skip(self) -> None:
        """注销当前聊天流的 VLM 跳过。

        在 execute() 结束时调用（通过 try/finally），
        恢复框架对该 stream 的 VLM 识别能力。
        """
        try:
            from src.core.managers import get_media_manager

            get_media_manager().unskip_vlm_for_stream(self.stream_id)
        except Exception as e:
            logger.debug(f"注销 VLM 跳过失败: {e}")

    def _extract_media(
        self,
        unread_msgs: list[Any],
        config: KFCConfig,
        image_budget: Any | None = None,
    ) -> list[Any] | None:
        """从未读消息中提取多模态图片数据。

        Args:
            unread_msgs: 未读消息列表
            config: KFC 配置
            image_budget: 图片预算追踪器，为 None 时使用 max_images_per_payload

        Returns:
            list | None: 图片列表，未启用或无图片时返回 None
        """
        if not config.general.native_multimodal:
            return None

        # 确定本次提取的配额
        if image_budget is not None:
            if image_budget.is_exhausted():
                logger.debug(" 原生多模态: 图片配额已用尽，跳过提取")
                return None
            max_items = image_budget.remaining
        else:
            max_items = config.general.max_images_per_payload

        from .multimodal import extract_media_from_messages

        raw_items = extract_media_from_messages(
            unread_msgs,
            max_items=max_items,
        )
        if raw_items:
            if image_budget is not None:
                image_budget.consume(len(raw_items))
            logger.debug(
                f" 原生多模态: 提取到 {len(raw_items)} 张图片"
                f" (配额剩余 {image_budget.remaining if image_budget else 'N/A'})"
            )
            return raw_items

        logger.debug(" 原生多模态: 未读消息中无图片")
        return None

    async def _get_virtual_trigger_message(self) -> Any:
        """构造虚拟触发消息，用于超时主动发言等无真实触发消息的场景。"""
        from src.core.managers.stream_manager import get_stream_manager

        sm = get_stream_manager()
        chat_stream = sm._streams.get(self.stream_id)  # HACK: 需要框架公开 API (stream_manager.get_stream)
        if not chat_stream:
            return None

        context = getattr(chat_stream, "context", None)
        if context and hasattr(context, "history_messages") and context.history_messages:
            return context.history_messages[-1]

        from src.core.models.message import Message

        return Message(
            message_id="virtual_timeout_trigger",
            platform=chat_stream.platform or "unknown",
            stream_id=self.stream_id,
            sender_id="system",
            sender_name="system",
            content="[超时触发]",
            processed_plain_text="[超时触发]",
        )

    async def _save_session(self, session: KFCSession) -> None:
        """保存 Session（持有 per-stream 锁）。"""
        store = self._get_session_store()
        async with store.lock(session.stream_id):
            await store.save(session)

    @staticmethod
    def _extract_timestamp(msg: Any) -> float:
        """从消息对象提取时间戳。

        框架 Message.time 定义为 float | int，此处做最小防御。
        """
        raw_time = getattr(msg, "time", None)
        if isinstance(raw_time, (int, float)):
            return float(raw_time)
        return time.time()

    @staticmethod
    def _record_reply_timing(session: KFCSession) -> None:
        """记录回复时效到活动流。"""
        from .mental_log import MentalLogEntry
        from .models import KFCEventType

        elapsed = session.waiting_config.get_elapsed_seconds()
        max_wait = session.waiting_config.max_wait_seconds

        if elapsed <= max_wait:
            event_type = KFCEventType.REPLY_IN_TIME
        else:
            event_type = KFCEventType.REPLY_LATE

        entry = MentalLogEntry(
            event_type=event_type,
            timestamp=time.time(),
            elapsed_seconds=elapsed,
        )
        session.mental_log.add(entry)

    # ── 调试日志方法 ────────────────────────────────────────

    def _log_prompt(self, response: Any) -> None:
        """输出发送给 LLM 的完整提示词（面板格式）。"""
        prompt_text = format_prompt_for_log(response)
        logger.print_panel(
            prompt_text,
            title=f"KFC 提示词 (stream={self.stream_id[:8]})",
            border_style="cyan",
        )
