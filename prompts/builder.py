"""KFC 提示词构建器。

KFCPromptBuilder 负责在 execute() 中构建完整的系统提示词，
以及对话循环中的 User Payload 和 Timeout Payload。
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

from src.core.prompt import get_prompt_manager
from src.kernel.llm import Content, LLMPayload, ROLE, Text

from .modules import build_mental_log_hint

if TYPE_CHECKING:
    from src.core.models.stream import ChatStream


class KFCPromptBuilder:
    """KFC 提示词构建器。

    从 PromptManager 中获取已注册的模板，
    填入动态变量后构建最终的系统提示词。
    """

    def build_system_prompt(
        self,
        chat_stream: ChatStream,
        extra_vars: dict[str, Any] | None = None,
    ) -> str:
        """构建系统提示词。

        Args:
            chat_stream: 当前聊天流
            extra_vars: 额外模板变量

        Returns:
            str: 完整的系统提示词
        """
        pm = get_prompt_manager()
        tmpl = pm.get_template("kfc_system_prompt")
        if not tmpl:
            return ""

        # 复制模板以避免污染全局状态
        tmpl = tmpl.clone()

        # 设置动态变量
        # 只用 chat_stream 属性补充模板中未设置的字段
        # nickname/alias_names 等已在 modules.py 的 register_kfc_prompts() 中
        # 从 core_config.personality 正确设置，此处不再覆盖
        tmpl.set("platform", chat_stream.platform or "unknown")
        tmpl.set("chat_type", str(chat_stream.chat_type or "unknown"))
        tmpl.set("bot_id", chat_stream.bot_id or "")
        tmpl.set(
            "current_time",
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # 活动流格式提示
        tmpl.set("mental_log_hint", build_mental_log_hint())

        # 场景引导
        theme_guide = self._get_theme_guide(chat_stream)
        tmpl.set("theme_guide", theme_guide)

        # 额外变量
        if extra_vars:
            for key, value in extra_vars.items():
                tmpl.set(key, value)

        final_prompt = tmpl.build()

        # 尝试注入人物关系信息（由 person_impression 插件写入 PromptManager）
        impression_tmpl = pm.get_template(
            f"person_impression:{chat_stream.stream_id}"
        )
        if impression_tmpl:
            relation_text = impression_tmpl.get("relation_text")
            if relation_text:
                final_prompt += "\n\n" + relation_text

        return final_prompt

    def build_user_payload(
        self,
        formatted_unreads: str,
        media_items: list[Any] | None = None,
    ) -> LLMPayload:
        """构建用户消息 Payload。

        将格式化的未读消息构建为一个 USER 角色的 Payload。
        心理活动已在融合叙事时间线中展示，不再单独注入摘要。
        如果携带多模态图片，则打包为 Text + Image 混合内容。

        Args:
            formatted_unreads: 格式化后的未读消息文本
            media_items: 多模态图片列表（可选，来自 extract_media_from_messages）

        Returns:
            LLMPayload: USER 角色的 Payload
        """
        # 心理活动已在融合叙事时间线中展示，不再单独注入摘要
        user_text = f"[新消息]\n{formatted_unreads}"

        content: Content | list[Content]
        if media_items:
            from ..multimodal import build_multimodal_content

            content = build_multimodal_content(user_text, media_items)
        else:
            content = Text(user_text)

        return LLMPayload(ROLE.USER, content)

    @staticmethod
    def build_timeout_payload(
        elapsed_seconds: float,
        expected_reaction: str,
        consecutive_timeouts: int,
        last_bot_message: str = "",
        pending_thoughts: list[str] | None = None,
    ) -> LLMPayload:
        """构建等待超时 Payload。

        使用丰富的超时决策提示词，引导 LLM 根据消息类型分析
        做出合理的下一步决策（继续等待、追问或结束）。

        Args:
            elapsed_seconds: 已等待秒数
            expected_reaction: 之前预期的对方反应
            consecutive_timeouts: 连续超时次数
            last_bot_message: 最后一条 Bot 发送的消息
            pending_thoughts: 等待期间产生的想法列表

        Returns:
            LLMPayload: USER 角色的超时 Payload
        """
        from .modules import build_timeout_context

        timeout_text = build_timeout_context(
            elapsed_seconds=elapsed_seconds,
            expected_reaction=expected_reaction,
            consecutive_timeouts=consecutive_timeouts,
            last_bot_message=last_bot_message,
            pending_thoughts=pending_thoughts,
        )

        return LLMPayload(ROLE.USER, Text(timeout_text))

    @staticmethod
    def _get_theme_guide(chat_stream: ChatStream) -> str:
        """根据聊天类型返回场景引导文本。"""
        chat_type = str(chat_stream.chat_type or "").lower()

        if chat_type == "private":
            return (
                "你当前处于私聊环境。你可以更亲近地和对方交流，"
                "关注对方情绪并提供更直接、细腻的回应。"
            )
        if chat_type == "group":
            return (
                "你当前处于群聊环境。注意多人对话的上下文，"
                "确认对方确实在和你说话后再做出回应。"
                "群聊中不要总是抢话，保持自然。"
            )
        return ""

    @staticmethod
    def build_history_text(chat_stream: ChatStream) -> str:
        """从 chat_stream context 构建历史消息文本（纯聊天记录）。

        Args:
            chat_stream: 当前聊天流

        Returns:
            str: 格式化的历史消息文本，无历史时返回空串
        """
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
                    time_str = datetime.datetime.fromtimestamp(raw_time).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                except (OSError, ValueError, OverflowError):
                    time_str = str(raw_time)
            else:
                time_str = str(raw_time or "")
            sender = getattr(msg, "sender_name", "未知")
            text = getattr(msg, "processed_plain_text", "")
            lines.append(f"【{time_str}】{sender}: {text}")

        return "以下为最近的聊天历史记录：\n" + "\n".join(lines)

    @staticmethod
    def build_fused_narrative(
        chat_stream: ChatStream,
        mental_log: Any,
    ) -> str:
        """构建聊天历史与内心独白的融合叙事。

        将数据库聊天记录和心理活动日志按时间交织在一起，
        形成一个包含"说了什么"和"想了什么"的统一时间线。
        这是老版 KFC 的核心设计——让 LLM 在回顾历史时不仅看到
        对话内容，还能看到每个节点上自己当时的内心活动。

        Args:
            chat_stream: 当前聊天流
            mental_log: 心理活动日志（MentalLog 实例）

        Returns:
            str: 融合叙事文本，无内容时返回空串
        """
        from ..models import KFCEventType

        history_messages = getattr(
            getattr(chat_stream, "context", None),
            "history_messages",
            [],
        )
        bot_id = str(chat_stream.bot_id or "")

        # timeline: (timestamp, formatted_line)
        timeline: list[tuple[float, str]] = []

        # ── 聊天记录 ──
        for msg in history_messages:
            raw_time = getattr(msg, "time", None)
            if not isinstance(raw_time, (int, float)):
                continue
            ts = float(raw_time)
            try:
                time_str = datetime.datetime.fromtimestamp(ts).strftime(
                    "%H:%M:%S"
                )
            except (OSError, ValueError, OverflowError):
                continue

            sender = getattr(msg, "sender_name", "未知")
            sender_id = str(getattr(msg, "sender_id", ""))
            text = getattr(msg, "processed_plain_text", "")
            if not text or not text.strip():
                continue

            is_bot = bot_id and sender_id == bot_id
            if is_bot:
                timeline.append((ts, f"[{time_str}] 你回复：{text}"))
            else:
                timeline.append((ts, f"[{time_str}] {sender}说：{text}"))

        # ── 内心独白（仅展示最近 7 条聊天消息范围内的思考） ──
        # 找到倒数第 7 条聊天消息的时间戳作为截止点
        chat_timestamps = [ts for ts, _ in timeline]
        mental_cutoff = (
            chat_timestamps[-7] if len(chat_timestamps) >= 7 else 0.0
        )

        if mental_log:
            for entry in mental_log.entries:
                if entry.timestamp < mental_cutoff:
                    continue
                ts = entry.timestamp
                try:
                    time_str = datetime.datetime.fromtimestamp(ts).strftime(
                        "%H:%M:%S"
                    )
                except (OSError, ValueError, OverflowError):
                    continue

                if (
                    entry.event_type == KFCEventType.BOT_PLANNING
                    and entry.thought
                ):
                    timeline.append(
                        (ts, f"[{time_str}] （你的内心：{entry.thought}）")
                    )

        if not timeline:
            return ""

        # 按时间排序
        timeline.sort(key=lambda x: x[0])

        lines = [item[1] for item in timeline]
        return (
            "以下为融合了聊天记录与你内心活动的时间线：\n"
            + "\n".join(lines)
        )
