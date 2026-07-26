"""KFC 近期摘要任务调度服务。

按聊天流去重后台压缩任务，并提供插件卸载时的集中取消入口。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from src.app.plugin_system.api.log_api import get_logger
from src.kernel.concurrency import get_task_manager

from ..compressor import compress_history, should_compress

if TYPE_CHECKING:
    from ..config import KFCConfig
    from ..prompts.builder import KFCPromptBuilder
    from ..session import KFCSession


logger = get_logger("kfc_summary_service")


class SummaryService:
    """处理对话链摘要压缩任务的调度、去重和取消。"""

    _task_ids: ClassVar[dict[str, str]] = {}

    @classmethod
    def maybe_schedule_compression(
        cls,
        session: KFCSession,
        prompt_builder: KFCPromptBuilder,
        config: KFCConfig,
        chat_stream: Any,
        session_store: Any = None,
    ) -> bool:
        """按当前 session 状态决定是否调度近期摘要压缩。"""

        stream_id = session.stream_id
        existing_task_id = cls._task_ids.get(stream_id)
        if existing_task_id is not None:
            try:
                if not get_task_manager().get_task(existing_task_id).is_done():
                    logger.debug(f"[KFC] 流 {stream_id} 已有摘要压缩任务，跳过重复调度")
                    return False
            except Exception:
                pass
            cls._task_ids.pop(stream_id, None)

        trigger_empty = not session.history_summary
        trigger_periodic = should_compress(session, config)
        if not (trigger_empty or trigger_periodic):
            return False

        reason = (
            "摘要为空（首次生成）"
            if trigger_empty
            else f"满足周期条件（{session.compress_round_count}轮）"
        )
        logger.info(f"[KFC] 触发近期记忆压缩：流 {stream_id}，原因：{reason}")

        async def _run_compression() -> None:
            """执行压缩并在任意退出路径释放流级任务登记。"""

            try:
                await compress_history(
                    session,
                    prompt_builder,
                    config,
                    chat_stream,
                    session_store=session_store,
                )
            finally:
                cls._task_ids.pop(stream_id, None)

        task_info = get_task_manager().create_task(
            _run_compression(),
            name=f"kfc_compress_{stream_id}",
            daemon=True,
            metadata={
                "plugin": "kokoro_flow_chatter",
                "purpose": "history_compression",
                "stream_id": stream_id,
            },
        )
        cls._task_ids[stream_id] = task_info.task_id
        return True

    @classmethod
    def cancel_all(cls) -> int:
        """取消所有仍在运行的 KFC 摘要压缩任务。"""

        task_manager = get_task_manager()
        cancelled = 0
        for task_id in tuple(cls._task_ids.values()):
            if task_manager.cancel_task(task_id):
                cancelled += 1
        cls._task_ids.clear()
        return cancelled
