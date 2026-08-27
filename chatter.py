"""KokoroFlowChatter 组件门面。

本类只承担两类职责：

1. **实现框架契约**——覆写 ``execute()``、``modify_llm_usables()``、
   ``format_message_line()`` 等 ``BaseChatter`` 接口；
2. **对 runtime 暴露能力**——把会话读写、回复发送、VLM 跳过等操作封装
   成公开方法，供主循环调用。

对话逻辑本身全部位于 ``runtime`` 包，本类不含流程控制。
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.stream_api import get_stream
from src.app.plugin_system.base import BaseChatter, Failure, Stop, Success, Wait
from src.app.plugin_system.types import ChatType, Message

from .actions.reply import KFCReplyAction
from .config import KFCConfig
from .debug.log_formatter import format_prompt_for_log
from .framework_compat import (
    clear_stream_recognition_skip,
    set_stream_recognition_skip,
)
from .mental_log import MentalLogEntry
from .models import DO_NOTHING, KFC_REPLY, KFCEventType
from .runtime import execute_orchestrator, send_interruptable_response

if TYPE_CHECKING:
    from .session import KFCSession, KFCSessionStore

logger = get_logger("kfc_chatter")

_VIRTUAL_TRIGGER_MESSAGE_ID = "virtual_timeout_trigger"
"""超时等无真实触发消息场景下使用的虚拟消息 ID。"""

_TOOL_NAME_PREFIXES: tuple[str, ...] = ("action-", "tool-", "agent-")


class KokoroFlowChatter(BaseChatter):
    """基于心理活动流的私聊聊天器。

    与常规聊天器的差异在于：每次决策都与内心独白绑定，对话历史与内心
    活动按时间线交织，使模型在回复时不仅看到"说了什么"，还能"回想起"
    当时在想什么。
    """

    name: str = "kokoro_flow_chatter"
    description: str = "心理活动流聊天器，模拟真实人类的连续心理活动和对话节奏"

    associated_platforms: list[str] = []
    chat_type: ChatType = ChatType.PRIVATE
    dependencies: list[str] = []

    # ── 框架契约 ──────────────────────────────────────────

    async def execute(self) -> AsyncGenerator[Wait | Success | Failure | Stop, None]:  # type: ignore[override]
        """执行对话循环，委托 runtime 编排器。"""
        async for result in execute_orchestrator(self):
            yield result

    async def modify_llm_usables(self, llm_usables: list[Any]) -> list[Any]:  # type: ignore[override]
        """在框架通用过滤之上，应用 KFC 屏蔽规则并稳定排序。

        排序是为了让工具列表在多轮请求间保持一致，避免顺序抖动破坏
        provider 的前缀缓存。

        Args:
            llm_usables: 框架给出的候选工具列表。

        Returns:
            list[Any]: 过滤并排序后的工具列表。
        """
        base_available = await super().modify_llm_usables(llm_usables)
        blocked_names = frozenset(
            name
            for name in self.get_config().general.blocked_tools
            # KFC 的核心控制动作不允许被配置屏蔽，否则模型将无法表达决策
            if name not in {KFC_REPLY, DO_NOTHING}
        )

        available = [
            usable
            for usable in base_available
            if _resolve_usable_name(usable) not in blocked_names
        ]
        return sorted(available, key=_resolve_usable_sort_key)

    @staticmethod
    def format_message_line(  # type: ignore[override]
        msg: Message,
        time_format: str = "%Y-%m-%d %H:%M:%S",
    ) -> str:
        """把单条消息渲染为带标签的展示行。

        格式为 ``》时间》[QQ:xxx] 昵称 [消息id:xxx]： 内容``——两种括号
        把平台账号与消息 ID 明确区分，避免模型混淆二者。

        Args:
            msg: 待渲染的消息。
            time_format: 时间格式。

        Returns:
            str: 渲染后的展示行。
        """
        raw_time = msg.time
        if isinstance(raw_time, (int, float)):
            time_str = datetime.fromtimestamp(raw_time).strftime(time_format)
        elif isinstance(raw_time, datetime):
            time_str = raw_time.strftime(time_format)
        else:
            time_str = ""

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

    # ── 配置与会话 ────────────────────────────────────────

    def get_config(self) -> KFCConfig:
        """获取 KFC 配置；插件上下文缺失时回退到默认配置。"""
        from .plugin import KFCPlugin

        if isinstance(self.plugin, KFCPlugin) and isinstance(
            self.plugin.config, KFCConfig
        ):
            return self.plugin.config
        return KFCConfig()

    @property
    def session_store(self) -> KFCSessionStore:
        """获取会话存储（由插件初始化时创建）。"""
        from .plugin import KFCPlugin

        if not isinstance(self.plugin, KFCPlugin):
            raise RuntimeError("KokoroFlowChatter 未运行在 KFCPlugin 上下文中")
        return self.plugin.session_store  # type: ignore[attr-defined]

    async def load_session(self) -> KFCSession:
        """读取当前流的会话（持有流级锁）。"""
        store = self.session_store
        async with store.lock(self.stream_id):
            return await store.get_or_create(self.stream_id)

    async def save_session(self, session: KFCSession) -> None:
        """保存会话（持有流级锁）。"""
        store = self.session_store
        async with store.lock(session.stream_id):
            await store.save(session)

    # ── 发送能力 ──────────────────────────────────────────

    async def send_reply(
        self,
        content: str,
        config: KFCConfig,
        trigger_msg: Message | None = None,
        reply_to: str = "",
    ) -> bool:
        """通过框架标准路径发送一段回复。

        Args:
            content: 回复文本。
            config: KFC 配置（保留参数以对齐执行层回调签名）。
            trigger_msg: 触发消息；为 ``None`` 时构造虚拟消息。
            reply_to: 要引用的消息 ID。

        Returns:
            bool: 是否发送成功。
        """
        _ = config
        if trigger_msg is None:
            trigger_msg = await self.build_virtual_trigger_message()
            if trigger_msg is None:
                logger.warning("无触发消息，无法发送回复")
                return False

        kwargs: dict[str, Any] = {"content": content}
        if reply_to:
            kwargs["reply_to"] = reply_to
        try:
            await self.exec_llm_usable(KFCReplyAction, trigger_msg, **kwargs)
            return True
        except Exception as error:
            logger.error(f"执行 KFCReplyAction 失败: {error}", exc_info=True)
            return False

    async def send_interruptable(
        self,
        send_target: Any,
        config: KFCConfig,
        known_unread_ids: frozenset[str],
    ) -> tuple[Any | None, list[Any]]:
        """以可打断方式发送 LLM 请求。"""
        return await send_interruptable_response(
            self, send_target, config, known_unread_ids
        )

    async def build_virtual_trigger_message(self) -> Message | None:
        """构造虚拟触发消息，用于超时主动发言等无真实触发的场景。

        优先复用最近一条历史消息以保留真实的会话上下文；历史为空时才
        退化为合成消息。

        Returns:
            Message | None: 触发消息；聊天流不可用时返回 ``None``。
        """
        chat_stream = await get_stream(self.stream_id)
        if chat_stream is None:
            return None

        history = chat_stream.context.history_messages
        if history:
            return history[-1]

        return Message(
            message_id=_VIRTUAL_TRIGGER_MESSAGE_ID,
            platform=chat_stream.platform or "unknown",
            stream_id=self.stream_id,
            sender_id="system",
            sender_name="system",
            content="[超时触发]",
            processed_plain_text="[超时触发]",
        )

    # ── 多模态识别跳过 ────────────────────────────────────

    def register_vlm_skip(self) -> None:
        """为当前流注册图片类型的 VLM 跳过。

        原生多模态模式下 KFC 直接把图片打包进 LLM payload，框架的 VLM
        转述属于冗余调用。表情包仍走 VLM 文字描述，以复用其哈希缓存。
        重复注册同一流是幂等的。
        """
        try:
            set_stream_recognition_skip(self.stream_id, ["image"])
        except Exception as error:
            logger.debug(f"注册识别跳过失败（不影响功能）: {error}")

    def unregister_vlm_skip(self) -> None:
        """注销当前流的识别跳过，恢复框架的默认识别行为。"""
        try:
            clear_stream_recognition_skip(self.stream_id)
        except Exception as error:
            logger.debug(f"注销识别跳过失败: {error}")

    # ── 辅助方法 ──────────────────────────────────────────

    @staticmethod
    def extract_timestamp(msg: Message) -> float:
        """提取消息时间戳。

        框架的 ``Message.time`` 类型为 ``datetime | float | None``，
        这里只接受数值形态，其余回退到当前时间。
        """
        raw_time = msg.time
        if isinstance(raw_time, (int, float)):
            return float(raw_time)
        return time.time()

    @staticmethod
    def record_reply_timing(session: KFCSession) -> None:
        """把本次回复的时效（及时 / 迟到）记入活动流。"""
        elapsed = session.waiting_config.get_elapsed_seconds()
        event_type = (
            KFCEventType.REPLY_IN_TIME
            if elapsed <= session.waiting_config.max_wait_seconds
            else KFCEventType.REPLY_LATE
        )
        session.mental_log.add(
            MentalLogEntry(
                event_type=event_type,
                timestamp=time.time(),
                elapsed_seconds=elapsed,
            )
        )

    def log_prompt(
        self,
        response: Any,
    ) -> None:
        """以面板格式输出发送给 LLM 的完整提示词。"""
        logger.print_panel(
            format_prompt_for_log(response),
            title=f"KFC 提示词 (stream={self.stream_id[:8]})",
            border_style="cyan",
        )


def _resolve_usable_name(usable: Any) -> str:
    """解析工具的末段名，用于匹配屏蔽列表。

    工具对象来自各插件，形态并不统一：优先读 schema 中的规范名，
    读取失败时退回类属性 ``name``。
    """
    try:
        schema = usable.to_schema()
        raw_name = str(schema.get("function", schema).get("name", "") or "")
    except (AttributeError, TypeError, KeyError):
        raw_name = str(usable.name or "") if hasattr(usable, "name") else ""

    normalized = raw_name.rsplit(":", 1)[-1]
    for prefix in _TOOL_NAME_PREFIXES:
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized


def _resolve_usable_sort_key(usable: Any) -> str:
    """解析工具的排序键。

    优先使用组件签名——它在插件间全局唯一且稳定；签名不可用时退回
    类名，保证排序结果仍然确定。
    """
    if hasattr(usable, "get_signature"):
        signature = str(usable.get_signature() or "")
        if signature:
            return signature
    return str(usable.__name__) if hasattr(usable, "__name__") else ""
