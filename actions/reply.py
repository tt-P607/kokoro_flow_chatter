"""KFC 回复动作。

包含最后一道防线：防御性清洗 content 中混入的元数据。
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any

from src.app.plugin_system.api.log_api import get_logger
from src.core.components.base.action import BaseAction

logger = get_logger("kfc_reply")

# 元数据关键字模式（最后防线）
# 仅当多个元数据关键字同时出现时才判定为泄漏，降低误伤概率
_METADATA_KEYWORDS = [
    r"(?:想法|内心想法|思考|thought|thinking)\s*[:：]",
    r"(?:预计反应|预期反应|expected_reaction)\s*[:：]",
    r"(?:最大等待秒数|max_wait_seconds)\s*[:：]",
    r"(?:心情|情绪|mood)\s*[:：]",
]
_METADATA_PATTERNS = [re.compile(kw, re.IGNORECASE) for kw in _METADATA_KEYWORDS]


class KFCReplyAction(BaseAction):
    """发送文本消息给对方。"""

    action_name = "kfc_reply"
    action_description = (
        "发送文本消息给对方。"
        "content 支持传入字符串数组来分段发送多条消息，每个元素是一条独立消息，"
        "系统会按顺序逐条发出并自动模拟打字延迟。"
        "也可以传单个字符串发送一条消息。"
        "可选的 reply_to 参数允许你引用消息（虽然私聊中较少用到，但引用旧消息时可能有用）。"
        "注意：本工具无法发送表情包等非文本内容。"
    )

    chatter_allow: list[str] = ["kokoro_flow_chatter"]

    async def execute(
        self,
        content: Annotated[
            str | list[str],
            "要发送的文本内容。"
            "可传字符串数组以分段发送多条消息，例如 [\"等等！\", \"你说什么意思啊\"]；"
            "也可传单个字符串发送一条消息。"
            "不要添加任何标记，只写你想说的话。",
        ],
        thought: Annotated[str, "你此刻的内心想法和感受，描述你为什么要这样回复"] = "",
        expected_reaction: Annotated[str, "你期望对方看到你这条消息后的反应"] = "",
        max_wait_seconds: Annotated[float, "你愿意等待对方回复的最长时间(秒)，0表示不等待"] = 0.0,
        mood: Annotated[str, "你当前的心情，用一两个词描述"] = "",
        reply_to: Annotated[str, "可选，要引用回复的消息 ID"] = "",
    ) -> tuple[bool, str]:
        """执行发送文本消息的逻辑。

        content 支持字符串或字符串列表，列表时逐条发送（parser 层处理延迟，
        此处仅作降级兜底：直接发送第一条非空内容）。
        reply_to 为可选的引用消息 ID。
        """
        # thought/expected_reaction/max_wait_seconds/mood 由 chatter.py 的策略层提取，
        # action 本身不使用这些参数

        # 统一为列表，兼容三种格式：
        # 1. 原生列表  2. JSON 字符串形式的列表  3. 普通字符串
        if isinstance(content, list):
            segments = [s.strip() for s in content if isinstance(s, str) and s.strip()]
        elif isinstance(content, str):
            stripped = content.strip()
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        segments = [s.strip() for s in parsed if isinstance(s, str) and s.strip()]
                    else:
                        segments = [stripped] if stripped else []
                except Exception:
                    segments = [stripped] if stripped else []
            else:
                segments = [stripped] if stripped else []
        else:
            segments = []

        if not segments:
            return False, "内容为空，未发送"

        # parser 层已处理多段逻辑；此处仅发送第一段作为兜底
        segment = segments[0]

        # 最后防线：仅当 >=2 个元数据关键字同时出现时才截断，降低误伤
        keyword_matches = [
            p.search(segment) for p in _METADATA_PATTERNS
        ]
        hit_count = sum(1 for m in keyword_matches if m is not None)
        if hit_count >= 2:
            # 找到最早的匹配位置进行截断
            earliest = min(
                (m.start() for m in keyword_matches if m is not None),
            )
            cleaned = segment[:earliest].strip()
            logger.warning(
                f"[最后防线] 检测到 content 中混入 {hit_count} 个元数据关键字，已截断。"
                f"原始长度={len(segment)}，截断后={len(cleaned)}"
            )
            segment = cleaned
            if not segment:
                return False, "清洗后内容为空，未发送"

        # 如果指定了 reply_to，创建带 reply_to 的 Message 对象
        if reply_to:
            from src.core.models.message import Message, MessageType
            from src.core.managers.adapter_manager import get_adapter_manager
            from uuid import uuid4
            
            target_stream_id = self.chat_stream.stream_id
            platform = self.chat_stream.platform
            chat_type = self.chat_stream.chat_type
            context = self.chat_stream.context
            
            bot_info = await get_adapter_manager().get_bot_info_by_platform(platform)
            
            target_user_id = None
            target_user_name = None
            
            def _get_last_context_message() -> Message | None:
                if context.unread_messages:
                    return context.unread_messages[-1]
                if context.history_messages:
                    return context.history_messages[-1]
                return context.current_message
            
            last_msg = _get_last_context_message()
            
            if last_msg:
                target_user_id = last_msg.sender_id
                target_user_name = last_msg.sender_name
            
            extra: dict[str, Any] = {}
            if target_user_id:
                extra["target_user_id"] = target_user_id
            if target_user_name:
                extra["target_user_name"] = target_user_name
            
            message = Message(
                message_id=f"action_{self.action_name}_{uuid4().hex}",
                content=segment,
                processed_plain_text=segment,
                message_type=MessageType.TEXT,
                sender_id=bot_info.get("bot_id", "") if bot_info else "",
                sender_name=bot_info.get("bot_nickname", "Bot") if bot_info else "Bot",
                platform=platform,
                chat_type=chat_type,
                stream_id=target_stream_id,
                reply_to=reply_to,
                **extra,
            )
            
            from src.core.transport.message_send import get_message_sender
            sender = get_message_sender()
            success = await sender.send_message(message)
            return success, f"已发送消息: {segment[:80]}"
        else:
            success = await self._send_to_stream(segment)
            if not success:
                return False, "消息发送失败"

            return True, f"已发送消息: {segment[:80]}"


