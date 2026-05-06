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
from datetime import datetime
from typing import TYPE_CHECKING, Any, AsyncGenerator

from src.app.plugin_system.api.llm_api import (
    LLMContextManager,
    create_llm_request,
)
from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.stream_api import get_stream
from src.app.plugin_system.base import (
    BaseChatter,
    Failure,
    Stop,
    Success,
    Wait,
)
from src.app.plugin_system.types import ChatType, Message

from .actions.reply import KFCReplyAction
from .debug.log_formatter import format_prompt_for_log
from .mental_log import MentalLogEntry
from .models import KFC_REPLY, DO_NOTHING, KFCEventType
from .multimodal import (
    ImageBudget,
    MediaItem,
    extract_media_from_messages,
    get_media_list,
)
from .prompts.builder import KFCPromptBuilder
from .runtime import (
    accumulate_message_buffer,
    execute_orchestrator,
    send_interruptable_response,
)

if TYPE_CHECKING:
    from src.app.plugin_system.types import ChatStream
    from src.app.plugin_system.api.llm_api import ToolRegistry

    from .config import KFCConfig
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
        from .plugin import KFCPlugin

        if isinstance(self.plugin, KFCPlugin) and isinstance(self.plugin.config, KFCConfig):
            return self.plugin.config
        return KFCConfig()

    @staticmethod
    def format_message_line(msg: "Message", time_format: str = "%Y-%m-%d %H:%M:%S") -> str:  # type: ignore[override]
        """将单条消息格式化为带标签的显示行（KFC 层覆盖）。

        格式：》时间》[QQ:xxx] 昵称 [消息id:xxx]： 内容
        两种括号将含义明确区分，避免模型将 QQ 号与消息 ID 混淆。
        """
        raw_time = msg.time
        if isinstance(raw_time, (int, float)):
            time_str = datetime.fromtimestamp(raw_time).strftime(time_format)
        elif isinstance(raw_time, datetime):
            time_str = raw_time.strftime(time_format)
        else:
            time_str = str(raw_time or "")

        role_str = BaseChatter._format_role(msg.sender_role)
        role_part = f"<{role_str}> " if role_str else ""

        platform_id = msg.sender_id or ""
        id_part = f"[QQ:{platform_id}] " if platform_id else ""

        nickname = msg.sender_name or ""
        cardname = msg.sender_cardname
        if cardname and cardname != nickname:
            name_part = f"{nickname}${cardname}"
        else:
            name_part = nickname or "未知发送者"

        message_id = msg.message_id or ""
        msg_id_part = f"[消息id:{message_id}]" if message_id else ""

        content = msg.processed_plain_text or str(msg.content or "")
        return f"》{time_str}》{role_part}{id_part}{name_part} {msg_id_part}： {content}"

    async def _get_session(self) -> KFCSession:
        """获取当前 stream 的 Session（持有 per-stream 锁）。"""
        session_store = self._get_session_store()
        async with session_store.lock(self.stream_id):
            return await session_store.get_or_create(self.stream_id)

    def _get_session_store(self) -> KFCSessionStore:
        """获取 Session Store（由 plugin.__init__ 初始化）。"""
        return self.plugin._session_store  # type: ignore[attr-defined]

    async def _accumulate_messages(
        self,
        config: KFCConfig,
    ) -> tuple[str, list[Any]]:
        """在积累窗口内等待并聚合连发消息。"""
        return await accumulate_message_buffer(self, config)

    async def modify_llm_usables(self, llm_usables: list[Any]) -> list[Any]:  # type: ignore[override]
        """过滤掉不需要的工具，保留 KFC 的正式 tool-calling 主链。"""
        config = self._get_config()
        _blocked = frozenset(
            name
            for name in config.general.blocked_tools
            if name not in {KFC_REPLY, DO_NOTHING}
        )

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

        return [u for u in llm_usables if not _is_reply_tool(u)]

    # ── 核心对话循环 ──────────────────────────────────────────

    async def execute(self) -> AsyncGenerator[Wait | Success | Failure | Stop, None]:  # type: ignore[override]
        """执行聊天器对话循环，委托 runtime orchestrator。"""
        async for result in execute_orchestrator(self):
            yield result

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
            with_reminder="actor",
        )

        # 系统提示词
        prompt_builder = KFCPromptBuilder()

        initial_payloads, has_history = await prompt_builder.build_initial_payloads(
            chat_stream,
            config,
            session,
        )
        for payload in initial_payloads:
            request.add_payload(payload)

        # 图片预算初始化（bot 已发图片 > 用户新消息图片 > 历史补充，共用同一总配额）
        image_budget: ImageBudget | None = None
        if config.general.native_multimodal:
            image_budget = ImageBudget(config.general.max_images_per_payload)
            # 预扣除 bot 自身近期发送的图片，确保其始终优先占用配额
            self._deduct_bot_sent_images(chat_stream, image_budget)

        # ── 注册工具（原生 Tool Calling） ──
        usable_map = await self.inject_usables(request)

        return request, image_budget, usable_map, prompt_builder, has_history

    # ── 可打断 LLM 调用 ─────────────────────────────────────

    async def _send_interruptable(
        self,
        response: Any,
        config: KFCConfig,
        known_unread_ids: frozenset[str],
    ) -> tuple[Any | None, list[Any]]:
        """以可打断方式发送 LLM 请求。"""
        return await send_interruptable_response(
            self,
            response,
            config,
            known_unread_ids,
        )

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
            from src.core.managers.media_manager import get_media_manager

            get_media_manager().skip_vlm_for_stream(self.stream_id)
        except Exception as e:
            logger.debug(f"注册 VLM 跳过失败（不影响功能）: {e}")

    def _unregister_vlm_skip(self) -> None:
        """注销当前聊天流的 VLM 跳过。

        在 execute() 结束时调用（通过 try/finally），
        恢复框架对该 stream 的 VLM 识别能力。
        """
        try:
            from src.core.managers.media_manager import get_media_manager

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
        bot_id = chat_stream.bot_id or ""
        if not bot_id:
            return

        history_msgs = chat_stream.context.history_messages
        if not history_msgs:
            return

        # 逆序取最近 20 条，仅保留 bot 自己发送的消息
        recent_bot_msgs = [
            m
            for m in reversed(history_msgs[-20:])
            if m.sender_id == bot_id
        ]

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

        history_msgs = chat_stream.context.history_messages
        if not history_msgs:
            return None

        # 过滤掉 bot 自身发送的消息
        bot_id = chat_stream.bot_id or ""
        recent_msgs = list(reversed(history_msgs[-20:]))
        if bot_id:
            recent_msgs = [
                m for m in recent_msgs
                if m.sender_id != bot_id
            ]

        if not recent_msgs:
            return None

        items: list[MediaItem] = []

        for msg in recent_msgs:
            if image_budget.is_exhausted() or len(items) >= image_budget.remaining:
                break
            msg_id = msg.message_id or ""
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
        chat_stream = await get_stream(self.stream_id)
        if not chat_stream:
            return None

        context = chat_stream.context
        if context and context.history_messages:
            return context.history_messages[-1]

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
    def _extract_timestamp(msg: "Message") -> float:
        """从消息对象提取时间戳。

        框架 ``Message.time`` 类型为 ``datetime | float | None``，
        这里只在 ``int|float`` 时接受，其余回退到当前时间。
        """
        raw_time = msg.time
        if isinstance(raw_time, (int, float)):
            return float(raw_time)
        return time.time()

    @staticmethod
    def _record_reply_timing(session: KFCSession) -> None:
        """记录回复时效到活动流。"""
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

    def _log_prompt(self, response: Any, chain_payloads: list[dict] | None = None) -> None:
        """输出发送给 LLM 的完整提示词（面板格式）。"""
        prompt_text = format_prompt_for_log(response, chain_payloads=chain_payloads)
        logger.print_panel(
            prompt_text,
            title=f"KFC 提示词 (stream={self.stream_id[:8]})",
            border_style="cyan",
        )
