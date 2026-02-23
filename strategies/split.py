"""拆分策略实现。

SplitStrategy 将对话过程拆分为两步：
1. 规划步：LLM 生成内心想法和行动列表（纯文本 JSON）
2. 执行步：Chatter 根据动作列表执行（回复内容从规划结果取出）

注意：这只影响 payload 的构建和响应的解析方式，
LLM 调用仍然由 Chatter 控制。
"""

from __future__ import annotations

import json
from typing import Any

from src.kernel.llm import LLMPayload, ROLE, Text

from ..models import StrategyResult
from ..multimodal import MediaItem, build_multimodal_content


# 规划步 system prompt 补充指令
_PLANNING_INSTRUCTION = """
# 输出格式要求
以 JSON 格式输出你的想法和行动决策，不要调用任何工具。

```json
{
    "thought": "你此刻的内心想法",
    "actions": [
        {"type": "kfc_reply", "content": "你想说的话"}
    ],
    "expected_reaction": "你预计对方接下来会怎样",
    "max_wait_seconds": 120,
    "mood": "当前心情"
}
```

action type 只能是：
- kfc_reply — 发送消息，需要 content 字段
- do_nothing — 不发送任何消息

max_wait_seconds：
- 大于 0：愿意继续等对方回复
- 等于 0：话题结束，不再等待

注意：
- thought 必须填写你真实的内心想法
- 如果决定回复，actions 中包含 kfc_reply
- 如果不需要回复，使用 do_nothing 并设 max_wait_seconds 为 0
"""


class SplitStrategy:
    """拆分策略：分为决策 + 回复两步。

    决策步让 LLM（sub_actor）以 JSON 输出行动决策，
    回复步让 LLM（actor）根据对话上下文生成自然回复。
    """

    def build_user_payload(
        self,
        formatted_unreads: str,
        mental_log_summary: str,
        extra_context: dict[str, Any] | None = None,
        media_items: list[MediaItem] | None = None,
    ) -> LLMPayload:
        """构建决策步 payload（含 JSON 输出指令，用于 sub_actor）。"""
        parts: list[str] = []

        if mental_log_summary:
            parts.append(f"# 你的近期活动流\n{mental_log_summary}")

        if formatted_unreads:
            parts.append(f"# 新收到的消息\n{formatted_unreads}")

        if extra_context:
            context_lines = [
                f"- {k}: {v}" for k, v in extra_context.items()
            ]
            parts.append(
                "# 补充信息\n" + "\n".join(context_lines)
            )

        parts.append(_PLANNING_INSTRUCTION)

        text = "\n\n".join(parts)

        # 多模态分支：将图片与文本混合打包
        if media_items:
            content_list = build_multimodal_content(text, media_items)
            return LLMPayload(ROLE.USER, content_list)

        return LLMPayload(ROLE.USER, Text(text))

    def build_reply_payload(
        self,
        formatted_unreads: str,
        mental_log_summary: str,
        media_items: list[MediaItem] | None = None,
    ) -> LLMPayload:
        """构建回复步 payload（不含决策指令，用于 actor 生成自然回复）。"""
        parts: list[str] = []

        if mental_log_summary:
            parts.append(f"# 你的近期活动流\n{mental_log_summary}")

        if formatted_unreads:
            parts.append(f"# 新收到的消息\n{formatted_unreads}")

        parts.append("请自然地回复对方。")

        text = "\n\n".join(parts)

        if media_items:
            content_list = build_multimodal_content(text, media_items)
            return LLMPayload(ROLE.USER, content_list)

        return LLMPayload(ROLE.USER, Text(text))

    def parse_response(
        self,
        response_text: str,
        call_list: list[Any] | None = None,
    ) -> StrategyResult:
        """解析规划步 LLM 响应（纯文本 JSON）。"""
        result = StrategyResult()

        if not response_text:
            return StrategyResult.create_error("LLM 未返回任何内容")

        parsed = self._try_parse_json(response_text)
        if not parsed:
            # 兜底：当作回复内容
            result.thought = "想要说些什么"
            result.actions = [{"type": "kfc_reply", "content": response_text}]
            return result

        result.thought = parsed.get("thought", "")
        result.expected_reaction = parsed.get("expected_reaction", "")
        result.max_wait_seconds = float(parsed.get("max_wait_seconds", 0))
        result.mood = parsed.get("mood", "")

        actions_raw = parsed.get("actions", [])
        if isinstance(actions_raw, list):
            result.actions = actions_raw
        elif isinstance(actions_raw, dict):
            result.actions = [actions_raw]

        return result

    def generate_timeout_decision(
        self,
        elapsed_seconds: float,
        expected_reaction: str,
        consecutive_timeouts: int,
        pending_thoughts: list[str] | None = None,
    ) -> LLMPayload:
        """生成超时决策 payload。"""
        parts: list[str] = [
            f"# 等待超时通知",
            f"你已经等了 {elapsed_seconds:.0f} 秒，但对方还没有回复。",
        ]

        if expected_reaction:
            parts.append(f"你之前期望的回应：{expected_reaction}")

        if consecutive_timeouts > 1:
            parts.append(
                f"这已经是连续第 {consecutive_timeouts} 次超时了。"
            )

        if pending_thoughts:
            thoughts_text = "\n".join(
                f"  - {t}" for t in pending_thoughts[-3:]
            )
            parts.append(f"等待期间你的想法：\n{thoughts_text}")

        parts.append(
            "\n请以上述 JSON 格式输出你的决策。"
        )

        text = "\n".join(parts)
        return LLMPayload(ROLE.USER, Text(text))

    @staticmethod
    def _try_parse_json(text: str) -> dict[str, Any] | None:
        """尝试从文本中解析 JSON。"""
        try:
            import json_repair
            result = json_repair.loads(text)
            if isinstance(result, dict):
                return result
        except Exception:
            pass

        import re
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        return None
