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
    get_model_set_by_name,
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
from .prompts.templates import KFC_PERCEIVE_FOLLOWUP_PROMPT, KFC_TOOL_INTENT_FOLLOWUP_PROMPT

if TYPE_CHECKING:
    from src.core.models.stream import ChatStream
    from src.kernel.llm import ToolRegistry

    from .config import KFCConfig
    from .multimodal import ImageBudget
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

    @staticmethod
    def format_message_line(msg: Any, time_format: str = "%H:%M") -> str:  # type: ignore[override]
        """将单条消息格式化为带标签的显示行（KFC 层覆盖）。

        格式：》时间》[QQ:xxx] 昵称 [\u6d88\u606fid:xxx]\uff1a \u5185\u5bb9
        两种括号将意義明确区分，避免模型将 QQ 号与消息 ID 混淡。
        """
        from datetime import datetime as _dt

        raw_time = getattr(msg, "time", None)
        if isinstance(raw_time, (int, float)):
            time_str = _dt.fromtimestamp(raw_time).strftime(time_format)
        elif isinstance(raw_time, _dt):
            time_str = raw_time.strftime(time_format)
        else:
            time_str = str(raw_time or "")

        role_raw = getattr(msg, "sender_role", None)
        role_str = BaseChatter._format_role(role_raw)
        role_part = f"<{role_str}> " if role_str else ""

        platform_id = getattr(msg, "sender_id", "") or ""
        id_part = f"[QQ:{platform_id}] " if platform_id else ""

        nickname = getattr(msg, "sender_name", "") or ""
        cardname = getattr(msg, "sender_cardname", None)
        if cardname and cardname != nickname:
            name_part = f"{nickname}${cardname}"
        else:
            name_part = nickname or "未知发送者"

        message_id = getattr(msg, "message_id", "") or ""
        msg_id_part = f"[消息id:{message_id}]" if message_id else ""

        content = getattr(msg, "processed_plain_text", None) or str(getattr(msg, "content", ""))
        return f"》{time_str}》{role_part}{id_part}{name_part} {msg_id_part}： {content}"

    async def _get_session(self) -> KFCSession:
        """获取当前 stream 的 Session（持有 per-stream 锁）。"""
        session_store = self._get_session_store()
        async with session_store.lock(self.stream_id):
            return await session_store.get_or_create(self.stream_id)

    def _get_session_store(self) -> KFCSessionStore:
        """获取 Session Store（由 plugin.__init__ 初始化）。"""
        return self.plugin._session_store  # type: ignore[attr-defined]

    async def modify_llm_usables(self, usables: list[Any]) -> list[Any]:
        """过滤掉 kfc_reply 和 do_nothing，回复决策改走 JSON 文本，第三方工具仍走 tool calling。
        额外过滤 config.general.blocked_tools 中指定的工具名。
        """
        config = self._get_config()
        _blocked = frozenset([KFC_REPLY, DO_NOTHING, *config.general.blocked_tools])

        def _is_reply_tool(u: Any) -> bool:
            try:
                schema = u.to_schema()
                name: str = schema.get("function", schema).get("name", "") or ""
            except Exception:
                name = str(getattr(u, "name", "") or "")
            # 归一化：兼容 "action-kfc_reply" / "action:kfc_reply" / "kfc_reply" 等格式
            n = name.rsplit(":", 1)[-1]
            for prefix in ("action-", "tool-", "agent-"):
                if n.startswith(prefix):
                    n = n[len(prefix):]
                    break
            return n in _blocked

        return [u for u in usables if not _is_reply_tool(u)]

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
        if chat_stream is None:
            logger.error(f"无法激活聊天流: {self.stream_id}")
            yield Failure("聊天流激活失败")
            return
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

            # ── 构建 LLM 上下文 ──
            (
                response, image_budget, usable_map, prompt_builder, has_history,
            ) = await self._build_initial_context(
                chat_stream, config, session, model_set
            )

            # 历史图片仅注入一次（首次有新消息时，用剩余配额填充）
            _history_images_injected = False
            _has_pending_tool_results = False

            # ── 对话循环 ──
            while True:
                # 读取未读消息
                formatted_text, unread_msgs = await self.fetch_unreads()

                if formatted_text and unread_msgs:
                    _has_pending_tool_results = False
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
                            message_id=getattr(msg, "message_id", ""),
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

                    # 多模态：新消息图片优先消耗预算
                    media_items = self._extract_media(
                        unread_msgs, config, image_budget
                    )

                    # 历史图片：新消息图片消耗预算后，将剩余配额分配给历史图片
                    # 仅在首次有新消息时注入一次，避免重复追加
                    if (
                        not _history_images_injected
                        and has_history
                        and image_budget is not None
                        and not image_budget.is_exhausted()
                    ):
                        _history_images_injected = True
                        history_imgs = self._extract_history_media(
                            chat_stream, image_budget
                        )
                        if history_imgs:
                            from .multimodal import build_multimodal_content

                            response.add_payload(
                                LLMPayload(
                                    ROLE.SYSTEM,
                                    build_multimodal_content(
                                        "[历史图片参考]", history_imgs
                                    ),
                                )
                            )

                    # 构建 user payload（委托给 PromptBuilder）
                    user_payload = prompt_builder.build_user_payload(
                        formatted_unreads=formatted_text,
                        media_items=media_items,
                    )

                    # 工具链闭合守卫：若 response 尾部为 tool_result，
                    # 直接追加 user 会形成非法序列（tool_result → user），
                    # 需先插入一个 assistant 桥接 payload 以满足 LLM 上下文规则。
                    if (
                        response.payloads
                        and response.payloads[-1].role == ROLE.TOOL_RESULT
                    ):
                        logger.debug(
                            "新消息到达时 response 尾部为 tool_result，"
                            "插入 assistant 桥接 payload 以闭合工具链"
                        )
                        response.add_payload(
                            LLMPayload(ROLE.ASSISTANT, Text("好的。"))
                        )

                    response.add_payload(user_payload)

                elif _has_pending_tool_results:
                    # 重置标志，避免 LLM 调用完成后再次触发（无限循环）
                    _has_pending_tool_results = False
                elif session.is_waiting():
                    # 正在等待中，检查是否有超时
                    from .thinker.timeout_handler import TimeoutHandler

                    timeout_handler = TimeoutHandler(config)
                    if timeout_handler.check_timeout(session):
                        # 若 response 尾部仍是 tool_result，说明上一轮工具链尚未被 LLM
                        # 承接闭合。此时不能直接注入 user 角色的超时 payload（会形成
                        # tool_result → user 的非法序列），需先续轮让 LLM 承接工具结果，
                        # 等工具链闭合后再处理超时。
                        if (
                            response.payloads
                            and response.payloads[-1].role == ROLE.TOOL_RESULT
                        ):
                            logger.debug(
                                "超时触发时 response 尾部为 tool_result，"
                                "先闭合工具链再处理超时"
                            )
                            _has_pending_tool_results = True
                            continue

                        timeout_ctx = timeout_handler.handle_timeout(session)

                        if timeout_handler.should_give_up(session):
                            logger.info("连续超时次数过多，结束对话")
                            await self._save_session(session)
                            yield Stop(0)
                            return

                        # 构建超时 payload（委托给 PromptBuilder）
                        from .prompts.builder import KFCPromptBuilder
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
                    await self._save_session(session)
                    yield Failure("LLM 请求失败", e)
                    continue

                # ── 解析 + 执行 ──
                # 超时主动触发时 unread_msgs 为空，trigger_msg 会是 None，
                # 导致所有 action 工具（send_emoji、music_search 等）执行被跳过。
                # 借用 _get_virtual_trigger_message() 补一个虚拟触发消息，
                # 确保超时场景下 action 工具能正常执行。
                trigger_msg = unread_msgs[-1] if unread_msgs else None
                if trigger_msg is None:
                    trigger_msg = await self._get_virtual_trigger_message()
                result = await parse_tool_calls(
                    response, usable_map, trigger_msg, config,
                    execute_reply_fn=self._execute_reply,
                    run_tool_call_fn=self.run_tool_call,
                    pre_execute_hook=lambda r: log_kfc_result(r, config),
                )

                # 活动流记录（同时保存原始 LLM 响应文本，供热启动使用）
                session.add_bot_planning(
                    thought=result.thought,
                    actions=result.actions,
                    expected_reaction=result.expected_reaction,
                    max_wait_seconds=result.max_wait_seconds,
                    raw_response=getattr(response, "message", "") or "",
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
                    if result.max_wait_seconds <= 0:
                        # 无等待时间 → 直接结束本轮对话
                        logger.debug("do_nothing（无等待），结束对话")
                        await self._save_session(session)
                        yield Stop(0)
                        return
                    # max_wait_seconds > 0 → 设置等待状态，继续走下方等待控制逻辑

                # ── 第三方工具回传：标记待处理，下轮循环继续 ──
                # has_info_tool（agent-/tool-）：有实际返回值，无论 content 是否为 []
                # 都需要立即续轮让 LLM 看到结果后正式回复。
                # 普通 action 工具（TTS、emoji 等）仍保持原有行为（不续轮）。
                if result.has_info_tool and not result.has_reply:
                    logger.debug(
                        "信息工具调用完成，tool_result 已积累到 response 链，立即续轮"
                    )
                    _has_pending_tool_results = True
                    continue
                if result.has_third_party and not result.has_reply and not result.has_do_nothing:
                    logger.debug(
                        "第三方工具调用完成，tool_result 已积累到 response 链，下轮循环继续"
                    )
                    _has_pending_tool_results = True
                    continue

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

    # ── LLM 上下文构建 ──────────────────────────────────────

    async def _build_initial_context(
        self,
        chat_stream: ChatStream,
        config: KFCConfig,
        session: KFCSession,
        model_set: Any,
    ) -> tuple[Any, ImageBudget | None, ToolRegistry, KFCPromptBuilder, bool]:
        """构建初始 LLM 上下文（系统提示 + 工具注册 + 图片预算）。

        组装 LLM 请求所需的全部初始 payload：系统提示词、人物关系、
        图片预算与历史叙事，并注册可用工具。

        Args:
            chat_stream: 当前聊天流
            config: KFC 配置
            session: 当前会话状态
            model_set: LLM 模型配置

        Returns:
            tuple: (request, image_budget, usable_map, prompt_builder, has_history)
        """
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

        # 注入自定义决策提示词（如果配置了）
        extra_vars: dict[str, str] = {}
        custom_prompt = config.general.custom_decision_prompt
        if custom_prompt and custom_prompt.strip():
            extra_vars["custom_decision_prompt"] = (
                f"# 决策指导\n{custom_prompt.strip()}"
            )

        system_prompt = await prompt_builder.build_system_prompt(
            chat_stream, extra_vars=extra_vars or None
        )
        request.add_payload(LLMPayload(ROLE.SYSTEM, Text(system_prompt)))

        relation_text = prompt_builder.build_relation_context(chat_stream)
        if relation_text:
            request.add_payload(LLMPayload(ROLE.SYSTEM, Text(relation_text)))

        # 图片预算初始化（bot 已发图片 > 用户新消息图片 > 历史补充，共用同一总配额）
        image_budget: ImageBudget | None = None
        if config.general.native_multimodal:
            from .multimodal import ImageBudget

            image_budget = ImageBudget(config.general.max_images_per_payload)
            # 预扣除 bot 自身近期发送的图片，确保其始终优先占用配额
            self._deduct_bot_sent_images(chat_stream, image_budget)

        # 历史消息 + 内心独白融合叙事（SYSTEM 角色，不会被上下文裁剪）
        # 注意：历史图片不在此处消耗预算，由对话循环在获知新消息图片数量后再填充，
        # 确保新消息图片优先占用预算，剩余配额才分配给历史图片。
        history_text = prompt_builder.build_fused_narrative(
            chat_stream, session.mental_log
        )
        if history_text:
            request.add_payload(LLMPayload(ROLE.SYSTEM, Text(history_text)))

        # ── 历史热启动：以真实 USER/ASSISTANT 对延续对话 ──
        # 融合叙事作为 SYSTEM 让模型"阅读历史"，但 SYSTEM role 权重远低于
        # 真实对话 turn；热启动 pair 让模型以第一人称 ASSISTANT role
        # "活在对话末尾"，新消息到达时情绪与记忆可无缝延续。
        warmup_rounds = config.prompt.warmup_rounds
        if warmup_rounds > 0:
            for warmup_payload in self._build_warmup_payloads(
                chat_stream, session, num_rounds=warmup_rounds
            ):
                request.add_payload(warmup_payload)

        # ── 注册工具（原生 Tool Calling） ──
        usable_map = await self.inject_usables(request)

        return request, image_budget, usable_map, prompt_builder, bool(history_text)

    # ── 历史热启动 ────────────────────────────────────────────

    def _build_warmup_payloads(
        self,
        chat_stream: Any,
        session: Any,
        num_rounds: int = 3,
    ) -> list[LLMPayload]:
        """从历史消息末尾重建若干轮 USER/ASSISTANT 对话链。

        目的：让模型在 execute() 重启后以第一人称"活在"对话中，
        ASSISTANT payload 优先使用 MentalLog 中保存的原始 LLM 响应文本
        （含 thought/content/expected_reaction 等完整字段），
        确保内心思考不在存储路径中丢失。

        策略：
        1. 从 history_messages 按连续同角色分组，取末尾 num_rounds 个 bot 组；
        2. 每个 bot 组找 MentalLog 中时间最近的 BOT_PLANNING 条目；
        3. 有 raw_response → 直接用；无则从 thought + processed_plain_text 重建 JSON。

        Args:
            chat_stream: 当前聊天流
            session: 当前 KFCSession（含 MentalLog）
            num_rounds: 取历史末尾最近几个 bot 回复轮次

        Returns:
            list[LLMPayload]: 热启动 payload 列表，可能为空
        """
        import json as _json
        from .models import KFCEventType

        history = list(
            getattr(getattr(chat_stream, "context", None), "history_messages", [])
        )
        if not history:
            return []

        bot_id = str(chat_stream.bot_id or "")

        def _is_bot(msg: Any) -> bool:
            sender_id = str(getattr(msg, "sender_id", ""))
            message_id = str(getattr(msg, "message_id", "") or "")
            return bool(
                (bot_id and sender_id == bot_id)
                or message_id.startswith("action_kfc_reply")
            )

        def _msg_time(msg: Any) -> float:
            t = getattr(msg, "time", None)
            return float(t) if isinstance(t, (int, float)) else 0.0

        # 仅保留有有效文本内容的消息
        valid = [
            m for m in history
            if (
                getattr(m, "processed_plain_text", "")
                or str(getattr(m, "content", ""))
            ).strip()
        ]
        if not valid:
            return []

        # 将连续同角色消息合并为组
        role_list: list[str] = []
        msg_groups: list[list[Any]] = []
        for msg in valid:
            role = "bot" if _is_bot(msg) else "user"
            if role_list and role_list[-1] == role:
                msg_groups[-1].append(msg)
            else:
                role_list.append(role)
                msg_groups.append([msg])

        # 从末尾往前数 num_rounds 个 bot 组，确定截取起始索引
        bot_seen = 0
        start_idx = 0
        for i in range(len(role_list) - 1, -1, -1):
            if role_list[i] == "bot":
                bot_seen += 1
                if bot_seen >= num_rounds:
                    start_idx = i - 1 if i > 0 and role_list[i - 1] == "user" else i
                    break

        role_list = role_list[start_idx:]
        msg_groups = msg_groups[start_idx:]

        # 结尾必须是 bot
        while role_list and role_list[-1] == "user":
            role_list.pop()
            msg_groups.pop()

        # 开头必须是 user
        while role_list and role_list[0] == "bot":
            role_list.pop(0)
            msg_groups.pop(0)

        if not role_list:
            return []

        # 准备 MentalLog BOT_PLANNING 条目（按时间升序）
        mental_log = getattr(session, "mental_log", None)
        planning_entries = []
        if mental_log:
            planning_entries = [
                e for e in mental_log.entries
                if e.event_type == KFCEventType.BOT_PLANNING
            ]

        def _find_planning_entry(msgs: list[Any]) -> Any | None:
            """找与该 bot 组时间最近的 BOT_PLANNING 条目。"""
            if not planning_entries or not msgs:
                return None
            group_ts = max(_msg_time(m) for m in msgs)
            # 取时间差最小的条目（允许条目时间略晚于消息存储时间）
            return min(
                planning_entries,
                key=lambda e: abs(e.timestamp - group_ts),
            )

        def _build_assistant_text(msgs: list[Any]) -> str:
            """构建 ASSISTANT payload 文本。

            优先使用 MentalLog raw_response（原始 LLM 输出 JSON）；
            无则用 thought + processed_plain_text 重建兼容格式，保留内心思考。
            """
            entry = _find_planning_entry(msgs)

            # 有 raw_response → 直接使用
            if entry and entry.metadata.get("raw_response"):
                return entry.metadata["raw_response"]

            # 无 raw_response → 从 MentalLog 字段 + 消息文本重建
            content_lines = [
                (
                    getattr(m, "processed_plain_text", "")
                    or str(getattr(m, "content", ""))
                ).strip()
                for m in msgs
            ]
            content_lines = [c for c in content_lines if c]

            if entry and entry.thought:
                # 重建成与 KFC JSON 模式一致的结构，保留 thought
                obj: dict[str, Any] = {
                    "thought": entry.thought,
                    "content": content_lines,
                }
                if entry.expected_reaction:
                    obj["expected_reaction"] = entry.expected_reaction
                if entry.max_wait_seconds:
                    obj["max_wait_seconds"] = entry.max_wait_seconds
                return _json.dumps(obj, ensure_ascii=False)

            # 兜底：纯文本拼接
            return "\n".join(content_lines)

        payloads: list[LLMPayload] = []
        for role, msgs in zip(role_list, msg_groups):
            if role == "user":
                lines = "\n".join(self.format_message_line(m) for m in msgs)
                payloads.append(LLMPayload(ROLE.USER, Text(f"[消息记录]\n{lines}")))
            else:
                text = _build_assistant_text(msgs)
                if text:
                    payloads.append(LLMPayload(ROLE.ASSISTANT, Text(text)))

        return payloads

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

            # JSON 回复模式：检查消息文本是否含有效 JSON
            from .reply_json import extract_json_reply, normalize_reply_data
            json_data = extract_json_reply(getattr(new_response, "message", None))

            if json_data:
                norm = normalize_reply_data(json_data)
                # JSON 有实际回复内容 → 完整响应，无需感知循环
                if not norm["is_do_nothing"]:
                    logger.debug("[KFC] 检测到含内容的 JSON 回复，直接返回")
                    return new_response

                # JSON 是 do_nothing（content=null）且没有工具调用 →
                # 模型把工具调用意图写进了 thought 但没有真正发出 tool_call，
                # 注入提示强制其发起实际调用
                if attempt < max_retries:
                    thought_preview = (norm.get("thought") or "")[:60]
                    logger.info(
                        f"[KFC] do_nothing + 无工具调用（可能存在工具意图未落地），"
                        f"注入强制提示重试 (第 {attempt + 1} 轮)"
                        f"{': ' + thought_preview + '...' if thought_preview else ''}"
                    )
                    new_response.add_payload(
                        LLMPayload(ROLE.USER, Text(KFC_TOOL_INTENT_FOLLOWUP_PROMPT))
                    )
                    response = new_response
                    continue

                # 重试耗尽，原样返回 do_nothing
                logger.debug("[KFC] do_nothing + 无工具调用，重试耗尽，直接返回")
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
        reply_to: str = "",
    ) -> bool:
        """通过框架标准路径发送回复。

        Args:
            content: 回复文本内容
            config: KFC 配置
            trigger_msg: 触发消息，为 None 时构造虚拟消息
            reply_to: 要引用的消息 ID（可选）

        Returns:
            bool: 是否发送成功
        """
        from .actions.reply import KFCReplyAction

        if trigger_msg is None:
            trigger_msg = await self._get_virtual_trigger_message()
            if trigger_msg is None:
                logger.warning("无触发消息，无法发送回复")
                return False

        try:
            kwargs: dict[str, Any] = {"content": content}
            if reply_to:
                kwargs["reply_to"] = reply_to
            await self.exec_llm_usable(KFCReplyAction, trigger_msg, **kwargs)
            return True
        except Exception as e:
            logger.error(f"通过框架执行 KFCReplyAction 失败: {e}", exc_info=True)
            return False

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

    def _deduct_bot_sent_images(
        self,
        chat_stream: Any,
        image_budget: Any,
    ) -> None:
        """从预算中预扣除 bot 自身近期发送的图片数量。

        bot 已发图片优先级最高，在图片预算初始化后立即调用，
        使后续的用户新消息图片和历史图片只能使用剩余配额。

        Args:
            chat_stream: 当前聊天流
            image_budget: 图片预算追踪器（刚完成初始化，尚未有任何消耗）
        """
        bot_id = str(getattr(chat_stream, "bot_id", "") or "")
        if not bot_id:
            return

        history_msgs = getattr(
            getattr(chat_stream, "context", None), "history_messages", []
        )
        if not history_msgs:
            return

        from .multimodal import extract_media_from_messages

        # 逆序取最近 20 条，仅保留 bot 自己发送的消息
        recent_bot_msgs = [
            m
            for m in reversed(history_msgs[-20:])
            if str(getattr(m, "sender_id", "")) == bot_id
        ]
        if not recent_bot_msgs:
            return

        bot_items = extract_media_from_messages(
            recent_bot_msgs, max_items=image_budget.remaining
        )
        if bot_items:
            image_budget.consume(len(bot_items))
            logger.debug(
                f"多模态: bot 已发图片预扣除 {len(bot_items)} 张"
                f" (剩余配额 {image_budget.remaining})"
            )

    def _extract_history_media(
        self,
        chat_stream: Any,
        image_budget: Any,
    ) -> list[Any] | None:
        """从聊天历史中提取用户侧图片，用剩余预算填充，最新优先。

        在 bot 已发图片（预扣除）和用户新消息图片（优先消耗）之后调用，
        仅扫描非 bot 发送的历史消息，避免与预扣除步骤重复计算。

        Args:
            chat_stream: 当前聊天流
            image_budget: 图片预算追踪器（已被 bot 图片和用户新消息消耗了对应配额）

        Returns:
            list | None: 历史图片列表，无可用图片或预算耗尽时返回 None
        """
        if image_budget.is_exhausted():
            return None

        history_msgs = getattr(
            getattr(chat_stream, "context", None), "history_messages", []
        )
        if not history_msgs:
            return None

        from .multimodal import MediaItem, get_media_list

        # 过滤掉 bot 自身发送的消息
        bot_id = str(getattr(chat_stream, "bot_id", "") or "")
        recent_msgs = list(reversed(history_msgs[-20:]))
        if bot_id:
            recent_msgs = [
                m for m in recent_msgs
                if str(getattr(m, "sender_id", "")) != bot_id
            ]

        if not recent_msgs:
            return None

        items: list[MediaItem] = []

        for msg in recent_msgs:
            if image_budget.is_exhausted() or len(items) >= image_budget.remaining:
                break
            msg_id = getattr(msg, "message_id", "")
            media_list = get_media_list(msg)
            for media in media_list:
                if len(items) >= image_budget.remaining:
                    break
                if media.get("type") not in ("image", "emoji"):
                    continue
                data = media.get("data", "")
                if not data:
                    continue
                items.append(
                    MediaItem(
                        media_type=media["type"],
                        base64_data=data,
                        source_message_id=msg_id,
                    )
                )

        if not items:
            return None

        image_budget.consume(len(items))
        logger.debug(
            f"历史多模态: 提取到 {len(items)} 张用户图片/表情包"
            f" (剩余配额 {image_budget.remaining})"
        )
        return items

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
