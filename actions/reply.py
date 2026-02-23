"""KFC 回复动作。

支持分段发送、模拟打字延迟和中断检测。
包含最后一道防线：防御性清洗 content 中混入的元数据。
"""

from __future__ import annotations

import asyncio
import re
from typing import Annotated

from src.app.plugin_system.api.log_api import get_logger
from src.core.components.base.action import BaseAction

logger = get_logger("kfc_reply")

# 元数据关键字模式（最后防线）
_METADATA_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:"
    r"(?:想法|内心想法|思考|thought|thinking)\s*[:：]|"
    r"(?:预计反应|预期反应|期望反应|expected_reaction|expected_user_reaction)\s*[:：]|"
    r"(?:最大等待秒数|等待时间|max_wait_seconds)\s*[:：]|"
    r"(?:心情|情绪|mood)\s*[:：]|"
    r"(?:理由|原因|reason)\s*[:：]"
    r")",
    re.IGNORECASE,
)


class KFCReplyAction(BaseAction):
    """发送文本消息给对方。

    支持长消息自动分段、模拟打字延迟、发送间中断检测。
    """

    action_name = "kfc_reply"
    action_description = (
        "发送一段文本消息给对方。"
        "你可以调用多次来分多段回复，每次提供你想说的话的纯文本内容。"
        "注意：本工具无法发送表情包等非文本内容。"
    )

    chatter_allow: list[str] = ["kokoro_flow_chatter"]

    @property
    def _reply_config(self) -> tuple[float, float, float, int]:
        """从插件配置中读取回复参数。

        Returns:
            tuple: (typing_chars_per_sec, typing_delay_min, typing_delay_max, max_segment_length)
        """
        try:
            from ..config import KFCConfig

            cfg = getattr(self.plugin, "config", None)
            if isinstance(cfg, KFCConfig):
                return (
                    cfg.reply.typing_chars_per_sec,
                    cfg.reply.typing_delay_min,
                    cfg.reply.typing_delay_max,
                    cfg.reply.max_segment_length,
                )
        except Exception:
            pass
        # 默认值
        return (15.0, 0.8, 4.0, 200)

    async def execute(
        self,
        content: Annotated[str, "要发送的文本内容，不要添加标记，只写你想说的话"],
        thought: Annotated[str, "你此刻的内心想法和感受，描述你为什么要这样回复"] = "",
        expected_reaction: Annotated[str, "你期望对方看到你这条消息后的反应"] = "",
        max_wait_seconds: Annotated[float, "你愿意等待对方回复的最长时间(秒)，0表示不等待"] = 0.0,
        mood: Annotated[str, "你当前的心情，用一两个词描述"] = "",
    ) -> tuple[bool, str]:
        """执行发送文本消息的逻辑。"""
        # thought/expected_reaction/max_wait_seconds/mood 由 chatter.py 的策略层提取，
        # action 本身不使用这些参数
        if not content or not content.strip():
            return False, "内容为空，未发送"

        content = content.strip()

        # 最后防线：检测并截断 content 中混入的元数据
        match = _METADATA_PATTERN.search(content)
        if match:
            cleaned = content[:match.start()].strip()
            logger.warning(
                f"[最后防线] 检测到 content 中混入元数据关键字，已截断。"
                f"原始长度={len(content)}，截断后={len(cleaned)}"
            )
            content = cleaned
            if not content:
                return False, "清洗后内容为空，未发送"
        segments = self._split_content(content)

        sent_count = 0
        _, delay_min, delay_max, _ = self._reply_config
        for i, segment in enumerate(segments):
            if not segment.strip():
                continue

            # 模拟打字延迟
            delay = self._calculate_typing_delay(segment)
            if i > 0 and delay > 0:
                await asyncio.sleep(delay)

            # 发送分段
            success = await self._send_to_stream(segment)
            if success:
                sent_count += 1

        if sent_count == 0:
            return False, "消息发送失败"

        if sent_count == 1:
            return True, f"已发送消息: {content[:80]}"
        return True, f"已分 {sent_count} 段发送消息"

    def _split_content(self, content: str) -> list[str]:
        """将长消息拆分为多个自然段。"""
        _, _, _, max_segment_length = self._reply_config
        if len(content) <= max_segment_length:
            return [content]

        segments: list[str] = []

        # 按换行拆分
        paragraphs = content.split("\n")
        current_segment = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_segment) + len(para) + 1 <= max_segment_length:
                if current_segment:
                    current_segment += "\n" + para
                else:
                    current_segment = para
            else:
                if current_segment:
                    segments.append(current_segment)
                # 如果单个段落超长，按标点分割
                if len(para) > max_segment_length:
                    sub_segments = self._split_by_punctuation(para)
                    segments.extend(sub_segments)
                    current_segment = ""
                else:
                    current_segment = para

        if current_segment:
            segments.append(current_segment)

        return segments if segments else [content]

    @staticmethod
    def _split_by_punctuation(text: str) -> list[str]:
        """按中英文标点分割长文本。"""
        # 在句号、问号、感叹号、分号等处分割
        parts = re.split(r"(?<=[。！？；\.\!\?;])\s*", text)
        result: list[str] = []
        current = ""

        for part in parts:
            if not part.strip():
                continue
            if len(current) + len(part) <= 200:
                current += part
            else:
                if current:
                    result.append(current.strip())
                current = part

        if current.strip():
            result.append(current.strip())

        return result if result else [text]

    def _calculate_typing_delay(self, segment: str) -> float:
        """计算模拟打字延迟（秒）。"""
        chars_per_sec, delay_min, delay_max, _ = self._reply_config
        if chars_per_sec <= 0:
            return 0.0

        char_count = len(segment)
        base_delay = char_count / chars_per_sec

        delay = max(delay_min, min(base_delay, delay_max))
        return delay
