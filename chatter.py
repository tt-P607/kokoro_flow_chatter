"""KokoroFlowChatter 核心聊天器。

实现完整的心理活动流对话循环：
1. 构建 LLM 上下文（系统提示 + 活动流 + 未读消息）
2. 维护 LLMResponse 链（response = request → loop）
3. 通过策略层解析响应和构建 payload
4. 执行动作并管理等待状态
5. 超时后重新注入上下文继续对话

严格遵循 DefaultChatter._execute_enhanced() 的 response 链模式。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, AsyncGenerator

from src.app.plugin_system.api.llm_api import (
    create_llm_request,
    create_tool_registry,
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
from src.core.config import get_core_config
from src.kernel.llm import Content, LLMContextManager, LLMPayload, ROLE, Text, ToolResult

if TYPE_CHECKING:
    from src.core.components.base.plugin import BasePlugin
    from src.core.models.stream import ChatStream

    from .config import KFCConfig
    from .models import StrategyResult
    from .session import KFCSession, KFCSessionStore
    from .strategies.base import ChatStrategy

logger = get_logger("kfc_chatter")

# 控制流标记
_KFC_REPLY = "kfc_reply"


class KokoroFlowChatter(BaseChatter):
    """KokoroFlowChatter 核心聊天器。

    基于心理活动流的对话模型：
    - 维护 LLMResponse 链贯穿整个 execute() 生命周期
    - 通过策略层构建 payload 和解析响应
    - 活动流为持久化审计日志，LLM 上下文通过 response 链自动积累
    - 支持 unified / split 两种执行模式
    """

    chatter_name: str = "kokoro_flow_chatter"
    chatter_description: str = (
        "心理活动流聊天器，模拟真实人类的连续心理活动和对话节奏"
    )

    associated_platforms: list[str] = []
    chat_type: ChatType = ChatType.PRIVATE
    dependencies: list[str] = []

    def _get_config(self) -> KFCConfig:
        """获取 KFC 配置。"""
        from .config import KFCConfig

        plugin_config = getattr(self.plugin, "config", None)
        if plugin_config and isinstance(plugin_config, KFCConfig):
            return plugin_config
        return KFCConfig()

    def _get_strategy(self) -> ChatStrategy:
        """根据配置获取策略实例。"""
        from .strategies import SplitStrategy, UnifiedStrategy

        config = self._get_config()
        if config.general.mode == "split":
            return SplitStrategy()
        return UnifiedStrategy()

    async def _get_session(self) -> KFCSession:
        """获取当前 stream 的 Session（持有 per-stream 锁）。"""
        session_store = self._get_session_store()
        async with session_store.lock(self.stream_id):
            return await session_store.get_or_create(self.stream_id)

    def _get_session_store(self) -> KFCSessionStore:
        """获取 Session Store。"""
        plugin = self.plugin
        store = getattr(plugin, "_session_store", None)
        if store is not None:
            return store

        from .session import KFCSessionStore

        new_store = KFCSessionStore()
        plugin._session_store = new_store  # type: ignore[attr-defined]
        return new_store

    async def execute(self) -> AsyncGenerator[Wait | Success | Failure | Stop, None]:
        """执行聊天器的对话循环。

        核心流程：
        1. 初始化 LLM 请求（系统提示 + 工具注册）
        2. response = request（建立 response 链）
        3. 循环：读取未读 → 策略构建 payload → LLM 调用 → 解析 → 执行动作
        4. 等待超时后注入超时 payload 继续 response 链

        Yields:
            Wait | Success | Failure | Stop: 执行结果
        """
        from src.core.managers.stream_manager import get_stream_manager

        # ── 初始化 ──
        stream_manager = get_stream_manager()
        chat_stream = await stream_manager.activate_stream(self.stream_id)
        config = self._get_config()
        strategy = self._get_strategy()
        session = await self._get_session()

        # ── 构建 LLM 请求 ──
        model_set = get_model_set_by_task(config.general.model_task)
        if not model_set:
            logger.error("无法获取模型配置")
            yield Failure("模型配置错误：未找到 model_task 配置")
            return

        # split 模式：决策步使用轻量级 sub_actor，回复步使用主模型
        is_unified = config.general.mode != "split"
        decision_model_set = None
        if not is_unified:
            decision_model_set = get_model_set_by_task("sub_actor")
            if not decision_model_set:
                logger.warning("sub_actor 模型不可用，split 决策步降级为主模型")
                decision_model_set = model_set

        context_manager = LLMContextManager(
            max_payloads=config.prompt.max_context_payloads
        )
        request = create_llm_request(
            model_set,
            "kokoro_flow_chatter",
            context_manager=context_manager,
        )

        # ── 注册第三方工具（过滤 KFC 核心动作 + 其他 chatter 专属动作） ──
        usables = await self.get_llm_usables()
        usables = await self.modify_llm_usables(usables)

        _KFC_TOOL_NAMES = {_KFC_REPLY}
        if is_unified:
            pre_filter_count = len(usables)
            filtered_usables = []
            for u in usables:
                # 排除 KFC 核心动作
                if getattr(u, "action_name", None) in _KFC_TOOL_NAMES:
                    continue
                # 排除其他 chatter 专属的 Action（chatter_allow 中没有本 chatter）
                chatter_allow = getattr(u, "chatter_allow", None)
                if chatter_allow and self.chatter_name not in chatter_allow:
                    continue
                filtered_usables.append(u)
            usables = filtered_usables
            if config.debug.show_prompt:
                kept_names = [
                    getattr(u, "action_name", None) or getattr(u, "tool_name", None) or type(u).__name__
                    for u in usables
                ]
                logger.debug(
                    f"[KFC] 工具过滤: {pre_filter_count} → {len(usables)} | "
                    f"保留: {kept_names}"
                )

        usable_map = create_tool_registry(usables)

        # 不使用原生 Tool Calling（避免 tool_choice 冲突 + thought 泄露）
        # 改为将第三方工具描述注入系统提示词的 JSON 动作类型中
        tool_schemas: list[dict] = []
        if usable_map.get_all():
            tool_schemas = [t.to_schema() for t in usable_map.get_all()]
            if config.debug.show_prompt:
                tool_schema_names = [
                    s.get("function", s).get("name", "?") for s in tool_schemas
                ]
                logger.debug(
                    f"[KFC] 注入 {len(tool_schema_names)} 个工具到 JSON 动作类型: "
                    f"{tool_schema_names}"
                )

        # 系统提示词（含动态工具描述）
        from .prompts.builder import KFCPromptBuilder

        prompt_builder = KFCPromptBuilder(log_format=config.prompt.log_format)
        system_prompt = prompt_builder.build_system_prompt(
            chat_stream, tool_schemas=tool_schemas
        )
        request.add_payload(LLMPayload(ROLE.SYSTEM, Text(system_prompt)))

        # 历史消息（来自 stream context）
        history_text = self._build_history_text(chat_stream)
        if history_text:
            history_content: Content | list[Content] = Text(history_text)
            # 多模态模式：提取历史中的表情包/图片一并打包
            if is_unified and config.general.native_multimodal:
                from .multimodal import extract_media_from_messages, build_multimodal_content
                history_msgs = getattr(
                    getattr(chat_stream, "context", None), "history_messages", []
                )
                if history_msgs:
                    history_media = extract_media_from_messages(
                        history_msgs[-20:],
                        max_items=config.general.max_images_per_payload,
                    )
                    if history_media:
                        history_content = build_multimodal_content(
                            history_text, history_media
                        )
                        logger.debug(
                            f" 历史多模态: 提取到 {len(history_media)} 张图片/表情包"
                        )
            request.add_payload(LLMPayload(ROLE.USER, history_content))

        # ── response = request（建立 response 链）──
        response = request

        # 原生多模态模式：仅 unified 模式生效
        # split 模式的决策步(sub_actor)通常不支持多模态输入，因此不跳过 VLM 识别
        if is_unified and config.general.native_multimodal:
            try:
                from src.core.managers.media_manager import get_media_manager
                get_media_manager().skip_vlm_for_stream(self.stream_id)
            except Exception:
                logger.warning("注册跳过 VLM 识别失败，后续消息仍会触发 VLM")

        # ── 对话循环 ──
        mental_summary = ""  # 活动流摘要（unified 和 split 共用）
        media_items = None  # 多模态图片（split 模式跨段传递）
        timeout_payload = None  # 超时 payload（split 模式跨段传递）

        while True:
            # 重置 split 模式跨段变量
            timeout_payload = None
            media_items = None
            mental_summary = ""

            # 读取未读消息
            formatted_text, unread_msgs = await self.fetch_unreads()

            if formatted_text and unread_msgs:
                # 记录用户消息到活动流
                for msg in unread_msgs:
                    session.add_user_message(
                        content=getattr(msg, "processed_plain_text", "") or str(
                            getattr(msg, "content", "")
                        ),
                        user_name=getattr(msg, "sender_name", "用户"),
                        user_id=getattr(msg, "sender_id", ""),
                        timestamp=self._extract_timestamp(msg),
                    )

                # 检查等待期间收到回复的时效
                if session.is_waiting():
                    self._record_reply_timing(session)
                    session.clear_waiting()

                # 多模态：提取未读消息中的图片数据（仅 unified 模式生效）
                # split 模式的决策步使用轻量级 sub_actor，通常不支持多模态，
                # 回复步虽使用主模型但不传入图片以保持一致性
                media_items = None
                if is_unified and config.general.native_multimodal:
                    from .multimodal import extract_media_from_messages
                    raw_items = extract_media_from_messages(
                        unread_msgs,
                        max_items=config.general.max_images_per_payload,
                    )
                    media_items = raw_items or None
                    logger.debug(
                        f" 原生多模态: 提取到 {len(raw_items)} 张图片"
                        + (f" (截断至 {config.general.max_images_per_payload})"
                           if raw_items else "，未读消息中无图片")
                    )

                # 策略构建 user payload
                mental_summary = session.mental_log.format_as_summary(
                    max_entries=config.prompt.max_log_entries
                )

                if is_unified:
                    # unified 模式：决策+回复一体，payload 直接加入主链
                    user_payload = strategy.build_user_payload(
                        formatted_unreads=formatted_text,
                        mental_log_summary=mental_summary,
                        media_items=media_items,
                    )
                    response.add_payload(user_payload)
                # split 模式：payload 在 LLM 调用段处理，此处暂存上下文
                # mental_summary / formatted_text / media_items 在下方 split 段使用

            elif session.is_waiting():
                # 正在等待中，检查是否有超时或连续思考
                from .thinker.timeout_handler import TimeoutHandler

                timeout_handler = TimeoutHandler(config)
                if timeout_handler.check_timeout(session):
                    # 超时处理
                    timeout_ctx = timeout_handler.handle_timeout(session)

                    if timeout_handler.should_give_up(session):
                        logger.info("连续超时次数过多，结束对话")
                        await self._save_session(session)
                        self._unskip_vlm(config)
                        yield Stop(0)
                        return

                    # 注入超时 payload
                    timeout_payload = strategy.generate_timeout_decision(
                        elapsed_seconds=timeout_ctx["elapsed_seconds"],  # type: ignore[arg-type]
                        expected_reaction=timeout_ctx["expected_reaction"],  # type: ignore[arg-type]
                        consecutive_timeouts=timeout_ctx["consecutive_timeouts"],  # type: ignore[arg-type]
                        pending_thoughts=timeout_ctx.get("pending_thoughts"),  # type: ignore[arg-type]
                    )

                    if is_unified:
                        # unified 模式：直接加入主链
                        response.add_payload(timeout_payload)
                    # split 模式：timeout_payload 在 LLM 调用段使用
                else:
                    # 未超时，继续等待。使用 Wait(0) 让框架在下一个 tick 立即唤醒，
                    # 以便及时响应新到达的消息（框架的 Wait(N) 会忽略新消息）。
                    # 超时判断由 session.waiting_config 内部追踪，不依赖框架计时。
                    yield Wait(0)
                    continue
            else:
                # 无消息且不在等待，短等待一个 tick 周期让框架立即唤醒
                yield Wait(0)
                continue

            # ── 调用 LLM ──
            if is_unified:
                # unified 模式：单次调用主链
                if config.debug.show_prompt:
                    self._log_prompt(response)

                try:
                    response = await response.send(stream=False)
                    await response
                    await self.flush_unreads(unread_msgs if unread_msgs else [])
                except Exception as e:
                    logger.error(f"LLM 请求失败: {e}", exc_info=True)
                    yield Failure("LLM 请求失败", e)
                    continue

                result = strategy.parse_response(
                    response_text=response.message or "",
                    call_list=response.call_list,
                )

                # 调试日志：显示 LLM 响应概况
                if config.debug.show_prompt:
                    call_count = len(response.call_list) if response.call_list else 0
                    call_names = [c.name for c in response.call_list] if response.call_list else []
                    msg_len = len(response.message or "")
                    logger.debug(
                        f"[KFC] LLM 响应: "
                        f"tool_calls={call_count} {call_names} | "
                        f"message_len={msg_len} | "
                        f"parsed_actions={[a.get('type') for a in result.actions]}"
                    )
            else:
                # ── split 模式：决策步(sub_actor) + 回复步(actor) ──
                from .strategies.split import SplitStrategy

                split_strategy = strategy if isinstance(strategy, SplitStrategy) else SplitStrategy()
                result, response = await self._split_llm_call(
                    split_strategy=split_strategy,
                    response=response,
                    decision_model_set=decision_model_set,  # type: ignore[arg-type]
                    system_prompt=system_prompt,
                    formatted_text=formatted_text or "",
                    mental_summary=mental_summary,
                    media_items=media_items,
                    timeout_payload=timeout_payload,
                    config=config,
                )
                await self.flush_unreads(unread_msgs if unread_msgs else [])

            # 输出响应美化日志（调试用）
            self._log_strategy_result(result, config)

            # 记录到活动流
            session.add_bot_planning(
                thought=result.thought,
                actions=result.actions,
                expected_reaction=result.expected_reaction,
                max_wait_seconds=result.max_wait_seconds,
            )

            # ── 处理动作 ──
            _CORE_ACTION_TYPES = {_KFC_REPLY, "respond", "do_nothing"}
            trigger_msg = unread_msgs[-1] if unread_msgs else None

            for action in result.actions:
                action_type = action.get("type", "")
                if action_type in (_KFC_REPLY, "respond"):
                    # 核心动作：回复消息
                    content = action.get("content", "")
                    if content:
                        await self._execute_reply(content, config, trigger_msg)
                elif action_type not in _CORE_ACTION_TYPES:
                    # 第三方工具：通过 JSON actions 调用（非原生 Tool Calling）
                    args = {
                        k: v for k, v in action.items()
                        if k not in ("type", "reason")
                    }
                    logger.info(
                        f"JSON 动作调用 {action_type}，"
                        f"想法: {result.thought[:50]}"
                    )
                    await self._execute_json_action(
                        action_type, args, usable_map, trigger_msg
                    )

            # 既没有回复也没有工具调用 → 直接结束
            has_meaningful_action = any(
                a.get("type") in (_KFC_REPLY, "respond")
                or a.get("type") not in _CORE_ACTION_TYPES
                for a in result.actions
            )
            if not has_meaningful_action:
                if result.actions and result.actions[0].get("type") == "do_nothing":
                    logger.debug("策略返回 do_nothing，跳过本轮")
                elif response.message and response.message.strip():
                    logger.warning(
                        f"LLM 返回无法解析的内容: {response.message[:100]}"
                    )
                self._unskip_vlm(config)
                yield Stop(0)
                await self._save_session(session)
                return

            # ── 等待控制（统一由 max_wait_seconds 驱动） ──
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
                # Wait(0)：让框架每个 tick 唤醒，超时由 session.waiting_config 追踪
                yield Wait(0)
                continue

            # max_wait_seconds <= 0 → 话题结束，不再等待
            session.clear_waiting()
            await self._save_session(session)
            self._unskip_vlm(config)
            yield Stop(0)
            return

    def _unskip_vlm(self, config: KFCConfig) -> None:
        """恢复该聊天流的 VLM 识别（对话结束时清理）。"""
        if config.general.native_multimodal:
            try:
                from src.core.managers.media_manager import get_media_manager
                get_media_manager().unskip_vlm_for_stream(self.stream_id)
            except Exception:
                pass

    async def _execute_json_action(
        self,
        action_type: str,
        args: dict[str, Any],
        usable_map: Any,
        trigger_msg: Any | None,
    ) -> None:
        """执行 JSON actions 中的第三方工具调用。

        通过 usable_map 查找工具类，使用 exec_llm_usable 执行。
        结果仅记录日志，不回传 LLM（与原生 Tool Calling 不同）。

        Args:
            action_type: 动作类型名称（即工具名）
            args: 动作参数（已去除 type 和 reason 字段）
            usable_map: 工具注册表
            trigger_msg: 触发消息
        """
        usable_cls = usable_map.get(action_type)
        if not usable_cls:
            logger.warning(f"JSON 动作 {action_type} 未找到对应工具，跳过")
            return

        try:
            if trigger_msg is None:
                trigger_msg = await self._get_virtual_trigger_message()
                if trigger_msg is None:
                    logger.warning(f"执行 {action_type} 失败: 无触发消息")
                    return

            success, result = await self.exec_llm_usable(
                usable_cls, trigger_msg, **args
            )
            if success:
                logger.info(f"JSON 动作 {action_type} 执行成功: {str(result)[:100]}")
            else:
                logger.warning(f"JSON 动作 {action_type} 执行失败: {result}")
        except Exception as e:
            logger.error(f"JSON 动作 {action_type} 执行异常: {e}", exc_info=True)

    async def _execute_reply(
        self,
        content: str,
        config: KFCConfig,
        trigger_msg: Any | None = None,
    ) -> None:
        """通过框架标准路径发送回复。

        Args:
            content: 回复文本内容
            config: KFC 配置（KFCReplyAction 从 plugin.config 自行读取）
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

    async def _split_llm_call(
        self,
        split_strategy: Any,
        response: Any,
        decision_model_set: Any,
        system_prompt: str,
        formatted_text: str,
        mental_summary: str,
        media_items: Any,
        timeout_payload: Any,
        config: KFCConfig,
    ) -> tuple[Any, Any]:
        """Split 模式两步 LLM 调用：决策(sub_actor) + 回复(actor)。

        1. 决策步：创建一次性 sub_actor 请求，发送决策 payload，获取 JSON 决策
        2. 回复步：如果决策要求回复，将回复 payload 加入主链(actor)发送，获取回复内容

        Args:
            split_strategy: SplitStrategy 实例
            response: 主 response 链（actor 模型）
            decision_model_set: 决策模型配置（sub_actor）
            system_prompt: 系统提示词
            formatted_text: 格式化的未读消息文本
            mental_summary: 活动流摘要
            media_items: 多模态图片列表
            timeout_payload: 超时决策 payload（非超时场景为 None）
            config: KFC 配置

        Returns:
            tuple[StrategyResult, response]: 解析结果和更新后的 response 链
        """
        from .models import StrategyResult

        # ── 决策步：sub_actor 一次性请求 ──
        decision_request = create_llm_request(decision_model_set, "kfc_decision")
        decision_request.add_payload(LLMPayload(ROLE.SYSTEM, Text(system_prompt)))

        if timeout_payload is not None:
            # 超时场景：使用超时决策 payload
            decision_request.add_payload(timeout_payload)
        elif formatted_text:
            # 正常消息场景：使用决策 payload（含 JSON 指令）
            decision_payload = split_strategy.build_user_payload(
                formatted_unreads=formatted_text,
                mental_log_summary=mental_summary,
                media_items=media_items,
            )
            decision_request.add_payload(decision_payload)
        else:
            # 无消息也无超时（不应到达此处）
            return StrategyResult.create_error("split 模式无可用输入"), response

        if config.debug.show_prompt:
            self._log_prompt(decision_request)

        try:
            decision_response = await decision_request.send(stream=False)
            await decision_response
        except Exception as e:
            logger.error(f"Split 决策步 LLM 请求失败: {e}", exc_info=True)
            return StrategyResult.create_error(f"决策请求失败: {e}"), response

        result = split_strategy.parse_response(
            response_text=decision_response.message or "",
        )

        logger.debug(
            f"Split 决策: thought={result.thought[:50]}, "
            f"actions={[a.get('type') for a in result.actions]}"
        )

        # ── 回复步：仅在决策要求回复时，使用 actor 主链生成 ──
        needs_reply = any(
            a.get("type") in (_KFC_REPLY, "respond") for a in result.actions
        )

        if needs_reply and formatted_text:
            # 构建回复 payload（不含 JSON 指令），加入主链
            reply_payload = split_strategy.build_reply_payload(
                formatted_unreads=formatted_text,
                mental_log_summary=mental_summary,
                media_items=media_items,
            )
            response.add_payload(reply_payload)

            try:
                response = await response.send(stream=False)
                await response
            except Exception as e:
                logger.error(f"Split 回复步 LLM 请求失败: {e}", exc_info=True)
                return StrategyResult.create_error(f"回复请求失败: {e}"), response

            # 用主链生成的回复内容替换决策中的 kfc_reply content
            reply_content = response.message or ""
            for action in result.actions:
                if action.get("type") in (_KFC_REPLY, "respond"):
                    action["content"] = reply_content

        elif needs_reply and timeout_payload is not None:
            # 超时场景：决策要求追问，用决策中的 content 作为回复
            # （决策 JSON 中已包含 content 字段）
            pass

        return result, response

    async def _get_virtual_trigger_message(self) -> Any:
        """构造虚拟触发消息，用于超时主动发言等无真实触发消息的场景。"""
        from src.core.managers.stream_manager import get_stream_manager

        sm = get_stream_manager()
        chat_stream = sm._streams.get(self.stream_id)
        if not chat_stream:
            return None

        context = getattr(chat_stream, "context", None)
        # 从历史消息中取最后一条作为虚拟触发
        if context and hasattr(context, "history_messages") and context.history_messages:
            return context.history_messages[-1]

        # 实在没有消息，构造最小化 Message 对象
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
    def _build_history_text(chat_stream: ChatStream) -> str:
        """从 chat_stream context 构建历史消息文本。"""
        from datetime import datetime

        history_messages = getattr(
            getattr(chat_stream, "context", None),
            "history_messages",
            [],
        )
        if not history_messages:
            return ""

        lines: list[str] = []
        for msg in history_messages:
            raw_time = getattr(msg, "time", None)
            if isinstance(raw_time, (int, float)):
                try:
                    time_str = datetime.fromtimestamp(raw_time).strftime("%Y-%m-%d %H:%M:%S")
                except (OSError, ValueError, OverflowError):
                    time_str = str(raw_time)
            else:
                time_str = str(raw_time or "")
            sender = getattr(msg, "sender_name", "未知")
            text = getattr(msg, "processed_plain_text", "")
            lines.append(f"【{time_str}】{sender}: {text}")

        return "以下为最近的聊天历史记录：\n" + "\n".join(lines)

    @staticmethod
    def _extract_timestamp(msg: Any) -> float:
        """从消息对象提取时间戳。"""
        raw_time = getattr(msg, "time", None)
        if isinstance(raw_time, (int, float)):
            return float(raw_time)
        if raw_time is not None:
            try:
                return float(raw_time)
            except (TypeError, ValueError):
                pass
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
        prompt_text = self._format_prompt_for_log(response)
        logger.print_panel(
            prompt_text,
            title=f"KFC 提示词 (stream={self.stream_id[:8]})",
            border_style="cyan",
        )

    @staticmethod
    def _format_prompt_for_log(response: Any) -> str:
        """从 LLM request/response 的 payload 列表中提取并格式化提示词。"""
        _MAX_CONTENT_LEN = 10000

        payloads = getattr(response, "payloads", None)
        if not payloads:
            return "（无 payload）"

        parts: list[str] = []
        for payload in payloads:
            # 提取角色名
            role = getattr(payload, "role", None)
            role_name = str(role.value).upper() if hasattr(role, "value") else str(role)

            # payload.content 是 list[Content | LLMUsable]
            content_list = getattr(payload, "content", [])
            if not isinstance(content_list, list):
                content_list = [content_list]

            # 遍历每个 content 元素提取文本
            text_parts: list[str] = []
            tool_names: list[str] = []
            for item in content_list:
                if hasattr(item, "text"):
                    # Text 对象
                    text_parts.append(item.text)
                elif hasattr(item, "value") and hasattr(item, "__class__") and item.__class__.__name__ == "Image":
                    # Image 对象：只显示摘要，不输出完整 base64
                    data_preview = str(item.value)[:40]
                    text_parts.append(f"[图片: {data_preview}...]")
                elif hasattr(item, "to_text"):
                    # ToolResult 对象
                    text_parts.append(item.to_text())
                elif hasattr(item, "to_schema"):
                    # LLMUsable（工具定义），记录名称
                    schema = item.to_schema()
                    func_info = schema.get("function", schema)
                    name = func_info.get("name", type(item).__name__)
                    tool_names.append(name)
                else:
                    text_parts.append(str(item))

            # 组合输出
            tool_count = len(tool_names)
            tool_summary = f"[{tool_count} 个工具: {', '.join(tool_names)}]" if tool_names else ""
            if tool_count > 0 and not text_parts:
                text = tool_summary
            elif tool_count > 0:
                text = "\n".join(text_parts) + f"\n[+ {tool_summary}]"
            elif text_parts:
                text = "\n".join(text_parts)
            else:
                text = "（空）"

            # 截断过长内容
            if len(text) > _MAX_CONTENT_LEN:
                text = text[:_MAX_CONTENT_LEN] + "\n[...截断...]"

            parts.append(f"── {role_name} ──\n{text}")

        return "\n\n".join(parts)

    @staticmethod
    def _log_strategy_result(result: StrategyResult, config: KFCConfig) -> None:
        """美化输出 LLM 响应摘要。"""
        if not config.debug.show_response:
            return

        # 内心想法
        if result.thought:
            logger.info(
                f"[bold magenta]💭[/bold magenta] {result.thought}"
            )

        # 动作列表
        for action in result.actions:
            action_type = action.get("type", "")
            if action_type in ("kfc_reply", "respond"):
                content = action.get("content", "")
                if content:
                    logger.info(
                        f"[bold green]💬[/bold green] {content}"
                    )
            elif action_type == "kfc_wait":
                logger.info(
                    "[bold yellow]⏳[/bold yellow] 等待对方回复"
                )
            elif action_type == "kfc_stop":
                logger.info(
                    "[bold red]🛑[/bold red] 结束对话"
                )
            elif action_type not in ("do_nothing", "no_action"):
                logger.info(
                    f"[bold cyan]🎯[/bold cyan] {action_type}"
                )

        # 元数据
        meta_parts: list[str] = []
        if result.max_wait_seconds > 0:
            meta_parts.append(f"⏱ {result.max_wait_seconds:.0f}s")
        if result.expected_reaction:
            meta_parts.append(f"预期: {result.expected_reaction}")
        if result.mood:
            meta_parts.append(f"心情: {result.mood}")
        if meta_parts:
            logger.info(f"[dim]{' | '.join(meta_parts)}[/dim]")
