"""KFC 回复动作。

包含最后一道防线：防御性清洗 content 中混入的元数据。
"""

from __future__ import annotations

import json
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
        "发送文本消息给对方。"
        "content 支持传入字符串数组来分段发送多条消息，每个元素是一条独立消息，"
        "系统会按顺序逐条发出并自动模拟打字延迟。"
        "也可以传单个字符串发送一条消息。"
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
    ) -> tuple[bool, str]:
        """执行发送文本消息的逻辑。

        content 支持字符串或字符串列表，列表时逐条发送（parser 层处理延迟，
        此处仅作降级兜底：直接发送第一条非空内容）。
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

        success = await self._send_to_stream(segment)
        if not success:
            return False, "消息发送失败"

        return True, f"已发送消息: {segment[:80]}"


