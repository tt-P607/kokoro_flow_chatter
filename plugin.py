"""KokoroFlowChatter 插件入口。

负责组件注册、提示词模板注册、会话存储初始化，以及主动发起周期任务
的调度与清理。
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from src.app.plugin_system.api.event_api import publish_event
from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.base import BasePlugin, register_plugin
from src.kernel.concurrency import get_task_manager

from .actions.do_nothing import DoNothingAction
from .actions.memo import KFCMemoAction, KFCMemoDeleteAction
from .actions.pass_and_wait import PassAndWaitAction
from .actions.reply import KFCReplyAction
from .actions.schedule_proactive import ScheduleProactiveAction
from .chatter import KokoroFlowChatter
from .config import KFCConfig
from .framework_compat import set_stream_recognition_skip
from .handlers.proactive_handler import PROACTIVE_TRIGGER_EVENT, ProactiveHandler
from .handlers.voice_call_history_handler import VoiceCallHistoryHandler
from .services import ProactiveService, SummaryService
from .session import KFCSessionStore

if TYPE_CHECKING:
    from src.kernel.scheduler import UnifiedScheduler

logger = get_logger("kfc_plugin")

_PROACTIVE_SCHEDULE_NAME = "kfc_proactive_check"
"""主动发起周期任务名，卸载时据此兜底清理。"""

_SCHEDULER_WAIT_SECONDS = 30
"""等待统一调度器启动的最长秒数。"""

_DEFAULT_MAX_LOG_ENTRIES = 50


@register_plugin
class KFCPlugin(BasePlugin):
    """KokoroFlowChatter 插件。"""

    plugin_name = "kokoro_flow_chatter"
    plugin_author = "言柒"
    plugin_description = "心理活动流聊天器，模拟真实人类的连续心理活动和对话节奏"
    configs = [KFCConfig]

    session_store: KFCSessionStore
    """会话存储，由本插件持有并共享给 Chatter 与各 Action。"""

    def __init__(self, config: KFCConfig | None = None) -> None:
        """初始化会话存储与后台任务句柄。"""
        super().__init__(config)
        max_log_entries = (
            config.prompt.max_log_entries if config else _DEFAULT_MAX_LOG_ENTRIES
        )
        self.session_store = KFCSessionStore(max_log_entries=max_log_entries)
        self._scheduler_init_task_id: str | None = None
        self._proactive_schedule_id: str | None = None

    # ── 生命周期 ──────────────────────────────────────────

    async def on_plugin_loaded(self) -> None:
        """注册提示词模板，并延迟拉起后台调度任务。"""
        from .prompts.modules import register_kfc_prompts

        register_kfc_prompts()
        logger.info("KFC 提示词模板已注册")

        config = self.config
        if not isinstance(config, KFCConfig):
            return

        if config.proactive.schedule_guidance:
            ScheduleProactiveAction.set_guidance(config.proactive.schedule_guidance)

        if not config.general.enabled:
            logger.info("KFC 插件已禁用，跳过后台任务注册")
            return

        # 调度器可能晚于插件启动，用后台任务等待；保存 ID 以便卸载时取消
        task_info = get_task_manager().create_task(
            self._delayed_scheduler_register(),
            name="kfc_scheduler_init",
            daemon=True,
            metadata={"plugin": self.plugin_name, "purpose": "scheduler_init"},
        )
        self._scheduler_init_task_id = task_info.task_id
        logger.info("KFC 插件已加载")

    async def on_plugin_unloaded(self) -> None:
        """取消后台任务并移除周期调度。"""
        if self._scheduler_init_task_id is not None:
            get_task_manager().cancel_task(self._scheduler_init_task_id)
            self._scheduler_init_task_id = None

        try:
            from src.kernel.scheduler import get_unified_scheduler

            scheduler = get_unified_scheduler()
            if self._proactive_schedule_id is not None:
                await scheduler.remove_schedule(self._proactive_schedule_id)
            else:
                await scheduler.remove_schedule_by_name(_PROACTIVE_SCHEDULE_NAME)
        except Exception as error:
            logger.warning(f"移除主动发起调度失败: {error}")
        finally:
            self._proactive_schedule_id = None
            ScheduleProactiveAction.set_guidance("")

        cancelled = SummaryService.cancel_all()
        if cancelled:
            logger.info(f"已取消 {cancelled} 个摘要压缩任务")
        logger.info("KFC 插件运行时资源已清理")

    def get_components(self) -> list[type]:
        """返回插件提供的全部组件。

        ``general.enabled`` 为 False 时不注册 Chatter，使框架的选择器
        看不到 KFC，从而自然由其他 chatter 接管私聊流；Action 与
        EventHandler 不受影响，仍正常注册。
        """
        components: list[type] = [
            KFCReplyAction,
            DoNothingAction,
            PassAndWaitAction,
            ScheduleProactiveAction,
            KFCMemoAction,
            KFCMemoDeleteAction,
            ProactiveHandler,
            VoiceCallHistoryHandler,
        ]

        config = self.config
        if isinstance(config, KFCConfig) and config.general.enabled:
            components.append(KokoroFlowChatter)
        else:
            logger.info("KFC 插件已禁用，不注册 KokoroFlowChatter 组件")
        return components

    # ── 后台任务 ──────────────────────────────────────────

    async def _delayed_scheduler_register(self) -> None:
        """等待统一调度器启动后注册周期任务。"""
        from src.kernel.scheduler import get_unified_scheduler

        try:
            for _ in range(_SCHEDULER_WAIT_SECONDS):
                await asyncio.sleep(1.0)
                if not get_unified_scheduler().get_statistics().get("is_running", False):
                    continue

                config = self.config
                if isinstance(config, KFCConfig) and config.general.native_multimodal:
                    await self._preload_vlm_skip()
                await self._register_proactive_schedule()
                return

            logger.warning(
                f"等待调度器启动超时（{_SCHEDULER_WAIT_SECONDS}s），放弃注册后台任务"
            )
        except asyncio.CancelledError:
            logger.debug("调度器初始化任务已取消")
            raise
        except Exception as error:
            logger.warning(f"调度器初始化失败: {error}")
        finally:
            self._scheduler_init_task_id = None

    async def _preload_vlm_skip(self) -> None:
        """为已知会话预注册 VLM 跳过。

        原生多模态模式下 KFC 自行处理图片，无需框架的 VLM 转述。重启后
        提前注册可避免历史会话的首条消息触发冗余识别。

        首次对话的新用户无法预注册，其第一条消息仍会经过 VLM——但原始
        图片数据始终保留在消息中，不影响功能正确性。
        """
        try:
            stream_ids = await self.session_store.list_all_stream_ids()
            for stream_id in stream_ids:
                set_stream_recognition_skip(stream_id)
            if stream_ids:
                logger.info(f"已预注册 {len(stream_ids)} 个聊天流的识别跳过")
        except Exception as error:
            logger.warning(f"预加载识别跳过失败（不影响功能）: {error}")

    async def _register_proactive_schedule(self) -> None:
        """注册主动发起的周期检查任务。"""
        config = self.config
        if not isinstance(config, KFCConfig) or not config.proactive.enabled:
            return

        try:
            from src.kernel.scheduler import get_unified_scheduler
            from src.kernel.scheduler.types import TriggerType

            scheduler: UnifiedScheduler = get_unified_scheduler()
        except Exception as error:
            logger.warning(f"获取调度器失败: {error}")
            return

        proactive_service = ProactiveService(
            config=config,
            session_store=self.session_store,
        )

        async def check_proactive() -> None:
            """扫描会话并为满足条件的流发布触发事件。"""
            for stream_id in await proactive_service.collect_triggered_streams():
                scheduled_reason = await proactive_service.mark_triggered(stream_id)
                logger.info(f"主动发起触发: {stream_id[:8]}")
                await publish_event(
                    PROACTIVE_TRIGGER_EVENT,
                    {"stream_id": stream_id, "scheduled_reason": scheduled_reason},
                )

        self._proactive_schedule_id = await scheduler.create_schedule(
            callback=check_proactive,
            trigger_type=TriggerType.TIME,
            trigger_config={"interval_seconds": config.proactive.check_interval},
            is_recurring=True,
            task_name=_PROACTIVE_SCHEDULE_NAME,
            force_overwrite=True,
        )
        logger.info("KFC 调度器任务注册完成")
