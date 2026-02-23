"""统一策略实现。

UnifiedStrategy 让 LLM 以 JSON 格式响应，在单次调用中同时完成：
- 内心活动（thought）
- 动作决策（actions 数组）— 包含核心动作和第三方工具调用
- 期望对方回复（expected_user_reaction）
- 最大等待时间（max_wait_seconds）

JSON 格式提供结构化安全边界，防止 thought 等元数据泄露到发送给用户的消息中。
第三方工具（如 send_emoji）通过 JSON actions 调用，不使用原生 Tool Calling，
避免 tool_choice 冲突和参数泄露问题。
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.app.plugin_system.api.log_api import get_logger
from src.kernel.llm import LLMPayload, ROLE, Text

from ..models import StrategyResult
from ..multimodal import MediaItem, build_multimodal_content

logger = get_logger("kfc_unified")

# 元数据关键字模式，用于检测 content 中混入的内部信息
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


class UnifiedStrategy:
    """统一策略：LLM 以 JSON 格式响应，策略负责解析为 StrategyResult。

    优先从纯文本中解析 JSON 结构。当 LLM 使用 tool calling 时（第三方工具）
    作为兜底路径处理。
    """

    def build_user_payload(
        self,
        formatted_unreads: str,
        mental_log_summary: str,
        extra_context: dict[str, Any] | None = None,
        media_items: list[MediaItem] | None = None,
    ) -> LLMPayload:
        """构建用户消息 payload（支持多模态）。"""
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

        if not parts:
            parts.append("（暂无新消息）")

        text = "\n\n".join(parts)

        # 多模态分支：将图片与文本混合打包
        if media_items:
            content_list = build_multimodal_content(text, media_items)
            return LLMPayload(ROLE.USER, content_list)

        return LLMPayload(ROLE.USER, Text(text))

    def parse_response(
        self,
        response_text: str,
        call_list: list[Any] | None = None,
    ) -> StrategyResult:
        """解析 LLM 响应为结构化结果。

        优先级：
        1. 从 response_text 中解析 JSON（主路径）
        2. 如果有 tool call（第三方工具），处理 tool call
        3. 兜底：返回 do_nothing，不发送消息
        """
        result = StrategyResult()

        # ── 主路径：JSON 解析 ──
        if response_text:
            parsed = self._try_parse_json(response_text)
            if parsed:
                result.thought = parsed.get("thought", "")
                # 兼容两种字段名
                result.expected_reaction = (
                    parsed.get("expected_user_reaction", "")
                    or parsed.get("expected_reaction", "")
                )
                result.max_wait_seconds = float(
                    parsed.get("max_wait_seconds", 0)
                )
                result.mood = parsed.get("mood", "")

                actions_raw = parsed.get("actions", [])
                if isinstance(actions_raw, list):
                    # 对每个 action 的 content 做防御性清洗
                    cleaned_actions: list[dict[str, Any]] = []
                    for action in actions_raw:
                        if not isinstance(action, dict):
                            continue
                        action_type = action.get("type", "")
                        if action_type in ("kfc_reply", "respond"):
                            content = action.get("content", "")
                            content = self._sanitize_content(content)
                            if content:
                                cleaned_actions.append(
                                    {"type": "kfc_reply", "content": content}
                                )
                        else:
                            cleaned_actions.append(action)
                    result.actions = cleaned_actions if cleaned_actions else actions_raw

                # 兼容：kfc_stop 映射为 max_wait_seconds=0（不等待/话题结束）
                if any(a.get("type") == "kfc_stop" for a in result.actions):
                    result.max_wait_seconds = 0.0

                return result

        # ── 兜底路径：tool call（仅处理第三方工具） ──
        if call_list:
            for call in call_list:
                args = call.args if isinstance(call.args, dict) else {}
                action_entry: dict[str, Any] = {
                    "type": call.name,
                    "call_id": call.id,
                }

                # 提取 KFC 特有的元数据字段
                if "thought" in args:
                    result.thought = args.pop("thought")
                if "expected_reaction" in args:
                    result.expected_reaction = args.pop("expected_reaction")
                if "expected_user_reaction" in args:
                    result.expected_reaction = args.pop("expected_user_reaction")
                if "max_wait_seconds" in args:
                    result.max_wait_seconds = float(args.pop("max_wait_seconds"))
                if "mood" in args:
                    result.mood = args.pop("mood")
                if "reason" in args:
                    args.pop("reason")

                # 对 reply 类 action 的 content 做防御性清洗
                if call.name in ("kfc_reply", "respond") and "content" in args:
                    args["content"] = self._sanitize_content(args["content"])

                action_entry.update(args)
                result.actions.append(action_entry)

            return result

        # ── 无法解析：返回 do_nothing ──
        if response_text and response_text.strip():
            logger.warning(
                f"JSON 解析失败，丢弃响应（不发送消息）: "
                f"{response_text[:120]}"
            )
        result.thought = "响应格式异常，选择不回复"
        result.actions = [{"type": "do_nothing"}]
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
            "\n请决定下一步行动：你可以主动追问、换个话题、"
            "或者结束对话等待下次机会。"
            "\n请按照 JSON 格式回复。"
        )

        text = "\n".join(parts)
        return LLMPayload(ROLE.USER, Text(text))

    @staticmethod
    def _sanitize_content(content: str) -> str:
        """防御性清洗回复内容，截断元数据泄露部分。

        检测 content 中是否混入了 thought/expected_reaction 等
        元数据关键字，如果检测到，截断到第一个元数据关键字之前。
        """
        if not content:
            return content

        match = _METADATA_PATTERN.search(content)
        if match:
            cleaned = content[:match.start()].strip()
            logger.warning(
                f"检测到回复内容中混入元数据，已截断。"
                f"原始长度={len(content)}，截断后={len(cleaned)}"
            )
            return cleaned

        return content

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

        # 尝试提取 markdown 代码块中的 JSON
        import re
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        return None
