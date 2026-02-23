"""策略协议定义。

ChatStrategy 协议定义了策略的三个核心方法：
- build_user_payload: 构建用户消息 payload
- parse_response: 解析 LLM 响应为 StrategyResult
- generate_timeout_decision: 生成超时后的决策 payload
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.kernel.llm import LLMPayload

from ..models import StrategyResult


@runtime_checkable
class ChatStrategy(Protocol):
    """聊天策略协议。

    策略负责构建 payload 和解析响应，但不负责调用 LLM。
    LLM 调用由 Chatter 控制。
    """

    def build_user_payload(
        self,
        formatted_unreads: str,
        mental_log_summary: str,
        extra_context: dict[str, Any] | None = None,
        media_items: list[Any] | None = None,
    ) -> LLMPayload:
        """构建用户消息 payload。

        Args:
            formatted_unreads: 格式化后的未读消息 JSON
            mental_log_summary: 活动流摘要
            extra_context: 额外上下文信息
            media_items: 多模态媒体条目列表（来自 multimodal.MediaItem）

        Returns:
            LLMPayload: 用户角色的 payload
        """
        ...

    def parse_response(
        self,
        response_text: str,
        call_list: list[Any] | None = None,
    ) -> StrategyResult:
        """解析 LLM 响应为结构化结果。

        Args:
            response_text: LLM 返回的文本内容
            call_list: LLM 返回的 tool call 列表

        Returns:
            StrategyResult: 解析后的结构化结果
        """
        ...

    def generate_timeout_decision(
        self,
        elapsed_seconds: float,
        expected_reaction: str,
        consecutive_timeouts: int,
        pending_thoughts: list[str] | None = None,
    ) -> LLMPayload:
        """生成超时后的决策 payload。

        Args:
            elapsed_seconds: 已等待时间（秒）
            expected_reaction: 期望的用户回应
            consecutive_timeouts: 连续超时次数
            pending_thoughts: 等待期间累积的想法

        Returns:
            LLMPayload: 注入到上下文中的超时提示 payload
        """
        ...
