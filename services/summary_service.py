"""KFC 近期摘要服务。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.log_api import get_logger
from src.kernel.concurrency import get_task_manager

from ..compressor import compress_history, should_compress

if TYPE_CHECKING:
    from ..config import KFCConfig
    from ..prompts.builder import KFCPromptBuilder
    from ..session import KFCSession


logger = get_logger("kfc_summary_service")


class SummaryService:
    """处理对话链摘要压缩触发。"""

    @staticmethod
    def maybe_schedule_compression(
        session: KFCSession,
        prompt_builder: KFCPromptBuilder,
        config: KFCConfig,
        chat_stream: Any,
        session_store: Any = None,
    ) -> bool:
        """按当前 session 状态决定是否调度近期摘要压缩。"""
        trigger_empty = not session.history_summary
        trigger_periodic = should_compress(session, config)
        if not (trigger_empty or trigger_periodic):
            return False

        reason = (
            "摘要为空（首次生成）"
            if trigger_empty
            else f"满足周期条件（{session.compress_round_count}轮）"
        )
        logger.info(f"[KFC] 触发近期记忆压缩：流 {session.stream_id}，原因：{reason}")
        get_task_manager().create_task(
            compress_history(
                session, prompt_builder, config, chat_stream,
                session_store=session_store,
            ),
            name=f"kfc_compress_{session.stream_id}",
        )
        return True