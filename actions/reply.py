"""KFC 回复动作。

包含最后一道防线：防御性清洗 content 中混入的元数据。
"""

from __future__ import annotations

import re
from typing import Annotated

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
        "发送一段文本消息给对方。"
        "你可以调用多次来分多段回复，每次提供你想说的话的纯文本内容。"
        "注意：本工具无法发送表情包等非文本内容。"
    )

    chatter_allow: list[str] = ["kokoro_flow_chatter"]

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

        # 最后防线：仅当 >=2 个元数据关键字同时出现时才截断，降低误伤
        keyword_matches = [
            p.search(content) for p in _METADATA_PATTERNS
        ]
        hit_count = sum(1 for m in keyword_matches if m is not None)
        if hit_count >= 2:
            # 找到最早的匹配位置进行截断
            earliest = min(
                (m.start() for m in keyword_matches if m is not None),
            )
            cleaned = content[:earliest].strip()
            logger.warning(
                f"[最后防线] 检测到 content 中混入 {hit_count} 个元数据关键字，已截断。"
                f"原始长度={len(content)}，截断后={len(cleaned)}"
            )
            content = cleaned
            if not content:
                return False, "清洗后内容为空，未发送"

        success = await self._send_to_stream(content)
        if not success:
            return False, "消息发送失败"

        return True, f"已发送消息: {content[:80]}"



