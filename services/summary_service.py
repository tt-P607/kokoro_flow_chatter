"""KFC 初始日记与封存段压缩的后台调度服务。"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, ClassVar

from src.app.plugin_system.api.log_api import get_logger
from src.kernel.concurrency import get_task_manager

from ..context.sources.history_source import build_recent_chat_history
from ..diary_compressor import bootstrap_diary, compress_history

if TYPE_CHECKING:
    from src.app.plugin_system.types import ChatStream

    from ..config import KFCConfig
    from ..session import KFCSession, KFCSessionStore

logger = get_logger("kfc_summary")

_PLUGIN_NAME = "kokoro_flow_chatter"
_TASK_PURPOSE = "history_compression"
_INITIAL_DIARY_DAYS = 3


class SummaryService:
    """近期记忆压缩任务的调度、去重与取消。

    任务登记表为类级共享——压缩由无状态的模块函数完成，按流去重只需
    一份全局映射，无需实例化。
    """

    _task_ids: ClassVar[dict[str, str]] = {}
    """``stream_id`` → 正在运行的压缩任务 ID。"""

    @classmethod
    def maybe_schedule_compression(
        cls,
        session: KFCSession,
        config: KFCConfig,
        chat_stream: ChatStream,
        session_store: KFCSessionStore | None = None,
    ) -> bool:
        """按会话状态决定是否调度初始日记或封存段压缩。

        Args:
            session: 当前会话。
            config: KFC 配置。
            chat_stream: 当前聊天流，压缩时用于构建系统提示词。
            session_store: 会话存储；传入时压缩完成后立即持久化。

        Returns:
            bool: 是否成功调度了新任务。
        """
        stream_id = session.stream_id
        if cls._has_running_task(stream_id):
            logger.debug(f"流 {stream_id[:8]} 已有压缩任务在跑，跳过重复调度")
            return False

        if session.compression_paused:
            return False
        now = time.time()
        initial_history = (
            build_recent_chat_history(
                chat_stream, now - _INITIAL_DIARY_DAYS * 86400, now
            )
            if not session.history_summary.strip()
            else ""
        )
        if not initial_history and not session.sealed_segments:
            return False
        if session_store is None:
            raise ValueError("后台压缩需要会话存储以提交日记")
        logger.info(f"开始处理日记：流 {stream_id[:8]}")

        async def run_compression() -> None:
            """执行压缩，并在任意退出路径上释放流级登记。"""
            try:
                if initial_history and not session.history_summary.strip():
                    succeeded = await bootstrap_diary(
                        session, config, chat_stream, initial_history,
                        session_store=session_store,
                    )
                    if not succeeded:
                        session.compression_paused = True
                        return
                while session.sealed_segments and not session.compression_paused:
                    succeeded = await compress_history(
                        session, config, chat_stream, session_store=session_store
                    )
                    if not succeeded:
                        session.compression_paused = True
            finally:
                cls._task_ids.pop(stream_id, None)

        task_info = get_task_manager().create_task(
            run_compression(),
            name=f"kfc_compress_{stream_id}",
            daemon=True,
            metadata={
                "plugin": _PLUGIN_NAME,
                "purpose": _TASK_PURPOSE,
                "stream_id": stream_id,
            },
        )
        cls._task_ids[stream_id] = task_info.task_id
        return True

    @classmethod
    def cancel_all(cls) -> int:
        """取消所有仍在运行的压缩任务，返回取消数量。"""
        task_manager = get_task_manager()
        cancelled = sum(
            1 for task_id in tuple(cls._task_ids.values())
            if task_manager.cancel_task(task_id)
        )
        cls._task_ids.clear()
        return cancelled

    @classmethod
    def _has_running_task(cls, stream_id: str) -> bool:
        """检查指定流是否已有未结束的压缩任务。"""
        task_id = cls._task_ids.get(stream_id)
        if task_id is None:
            return False
        try:
            task_info: Any = get_task_manager().get_task(task_id)
            if not task_info.is_done():
                return True
        except Exception:
            # 任务已从管理器中移除，视为已结束
            pass
        cls._task_ids.pop(stream_id, None)
        return False
