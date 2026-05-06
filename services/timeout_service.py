"""KFC 超时服务。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.types import LLMPayload

from ..thinker.timeout_handler import TimeoutHandler

if TYPE_CHECKING:
    from ..config import KFCConfig
    from ..session import KFCSession


logger = get_logger("kfc_timeout_service")


@dataclass(slots=True)
class TimeoutResult:
    """一次超时处理的输出。"""

    payload: LLMPayload
    is_final_timeout: bool


class TimeoutService:
    """封装等待超时处理与 payload 构建。

    本服务保持纯函数：不会对 ``response.payloads`` 做任何写入，
    调用方负责使用 ``safe_add_payload`` / ``ensure_tool_chain_closed``
    将返回的 payload 安全合并到上下文。
    """

    def __init__(self, config: KFCConfig) -> None:
        self._config = config
        self._handler = TimeoutHandler(config)

    def check_timeout(self, session: KFCSession) -> bool:
        """检查是否达到超时条件。"""
        return self._handler.check_timeout(session)

    def build_timeout_result(self, session: KFCSession) -> TimeoutResult:
        """处理超时并返回追加到 response 的 user payload。"""
        timeout_ctx = self._handler.handle_timeout(session)
        is_final_timeout = self._handler.should_give_up(session)

        from ..prompts.builder import KFCPromptBuilder

        payload = KFCPromptBuilder.build_timeout_payload(
            elapsed_seconds=timeout_ctx["elapsed_seconds"],  # type: ignore[arg-type]
            expected_reaction=timeout_ctx["expected_reaction"],  # type: ignore[arg-type]
            consecutive_timeouts=timeout_ctx["consecutive_timeouts"],  # type: ignore[arg-type]
            last_bot_message=timeout_ctx.get("last_bot_message", ""),  # type: ignore[arg-type]
            max_consecutive_timeouts=self._config.wait.max_consecutive_timeouts,
        )
        return TimeoutResult(payload=payload, is_final_timeout=is_final_timeout)