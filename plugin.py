"""KokoroFlowChatter 插件入口。

注册插件、加载配置、注册提示词模板、初始化 Scheduler 任务。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
from .handlers.proactive_handler import ProactiveHandler
from .handlers.voice_call_history_handler import VoiceCallHistoryHandler
from .framework_compat import set_stream_recognition_skip
from .services.summary_service import SummaryService
from .session import KFCSessionStore

if TYPE_CHECKING:
    from src.kernel.scheduler import UnifiedScheduler

logger = get_logger("kfc_plugin")


@register_plugin
class KFCPlugin(BasePlugin):
    """KokoroFlowChatter 插件。"""

    plugin_name = "kokoro_flow_chatter"
    plugin_author = "言柒"
    plugin_description = "心理活动流聊天器，模拟真实人类的连续心理活动和对话节奏"
    configs = [KFCConfig]

    _session_store: KFCSessionStore

    def __init__(self, config: KFCConfig | None = None) -> None:
        """初始化会话存储和插件级后台资源状态。"""

        super().__init__(config)
        max_log_entries = config.prompt.max_log_entries if config else 50
        self._session_store = KFCSessionStore(max_log_entries=max_log_entries)
        self._scheduler_init_task_id: str | None = None
        self._proactive_schedule_id: str | None = None

    async def on_plugin_loaded(self) -> None:
        """插件加载时注册提示词模板。调度任务延迟到调度器启动后注册。"""
        config = self.config
        is_enabled = isinstance(config, KFCConfig) and config.general.enabled

        # 注册提示词模板
        from .prompts.modules import register_kfc_prompts

        register_kfc_prompts()
        logger.info("KFC 提示词模板已注册")

        # 将配置中的 schedule_proactive 指导语写入 Action 类变量
        if isinstance(config, KFCConfig) and config.proactive.schedule_guidance:
            ScheduleProactiveAction._guidance = config.proactive.schedule_guidance

        if not is_enabled:
            logger.info("KFC 插件已禁用，跳过调度器任务注册")
            return

        # 延迟注册调度器任务：等待调度器启动。保存任务 ID，确保插件卸载时
        # 能取消尚未结束的初始化流程。
        task_info = get_task_manager().create_task(
            self._delayed_scheduler_register(),
            name="kfc_scheduler_init",
            daemon=True,
            metadata={"plugin": self.plugin_name, "purpose": "scheduler_init"},
        )
        self._scheduler_init_task_id = task_info.task_id

        logger.info("KFC 插件已加载")

    async def _delayed_scheduler_register(self) -> None:
        """延迟注册调度器任务，等待调度器启动（最多 30 秒）。"""
        import asyncio

        from src.kernel.scheduler import get_unified_scheduler

        config = self.config
        try:
            for _ in range(30):
                await asyncio.sleep(1.0)
                scheduler = get_unified_scheduler()
                if scheduler.get_statistics().get("is_running", False):
                    if isinstance(config, KFCConfig) and config.general.native_multimodal:
                        await self._preload_vlm_skip()
                    await self._register_scheduler_tasks()
                    return
            logger.warning("等待调度器启动超时(30s)，放弃注册后台任务")
        except asyncio.CancelledError:
            logger.debug("KFC Scheduler 初始化任务已取消")
            raise
        except Exception as error:
            logger.warning(f"KFC Scheduler 初始化失败: {error}")
        finally:
            self._scheduler_init_task_id = None

    async def _preload_vlm_skip(self) -> None:
        """从持久化存储预加载已知聊天流，注册 VLM 跳过。

        在 native_multimodal 模式下，KFC 自行处理图片数据，
        无需框架的 VLM 管线对图片进行文本转述。此方法在插件加载时
        将所有已知会话的 stream_id 注册到 MediaManager 的跳过列表，
        确保重启后已有会话的消息不再触发冗余 VLM 调用。

        注意：首次对话的新用户无法预注册，其第一条消息仍会经过 VLM。
        但由于 base64 数据始终保留在 Message.media 中，KFC 仍能正常
        提取原始图片，不影响功能正确性。
        """
        try:
            stream_ids = await self._session_store.list_all_stream_ids()
            if not stream_ids:
                logger.debug("无历史会话需要预注册 VLM 跳过")
                return

            for stream_id in stream_ids:
                set_stream_recognition_skip(stream_id)

            logger.info(
                f"已预注册 {len(stream_ids)} 个聊天流的识别跳过"
            )
        except Exception as e:
            logger.warning(f"预加载 VLM 跳过失败（不影响功能）: {e}")

    async def _register_scheduler_tasks(self) -> None:
        """注册后台调度任务。"""
        config = self.config
        if not isinstance(config, KFCConfig):
            return

        try:
            from src.kernel.scheduler import get_unified_scheduler
            from src.kernel.scheduler.types import TriggerType

            scheduler: UnifiedScheduler = get_unified_scheduler()
        except Exception as e:
            logger.warning(f"获取 Scheduler 失败: {e}")
            return

        # 主动发起检查
        if config.proactive.enabled:
            from .thinker.proactive import ProactiveThinker

            proactive = ProactiveThinker(
                config=config,
                session_store=self._session_store,
            )

            async def proactive_check() -> None:
                """定期检查是否需要主动发起。"""
                triggered = await proactive.check_all_sessions()
                for stream_id in triggered:
                    scheduled_reason = await proactive.mark_triggered(stream_id)
                    logger.info(f"主动发起触发: {stream_id[:8]}")
                    # 通过事件 API 触发 chatter
                    from src.app.plugin_system.api.event_api import publish_event

                    await publish_event(
                        "kfc.proactive_trigger",
                        {"stream_id": stream_id, "scheduled_reason": scheduled_reason},
                    )

            # 注册周期性主动发起检查任务，并保存 schedule ID 供卸载清理。
            self._proactive_schedule_id = await scheduler.create_schedule(
                callback=proactive_check,
                trigger_type=TriggerType.TIME,
                trigger_config={"interval_seconds": config.proactive.check_interval},
                is_recurring=True,
                task_name="kfc_proactive_check",
                force_overwrite=True,
            )

        logger.info("KFC 调度器任务注册完成")

    async def on_plugin_unloaded(self) -> None:
        """取消插件级任务并移除主动发起周期调度。"""

        if self._scheduler_init_task_id is not None:
            get_task_manager().cancel_task(self._scheduler_init_task_id)
            self._scheduler_init_task_id = None

        try:
            from src.kernel.scheduler import get_unified_scheduler

            scheduler = get_unified_scheduler()
            if self._proactive_schedule_id is not None:
                await scheduler.remove_schedule(self._proactive_schedule_id)
            else:
                await scheduler.remove_schedule_by_name("kfc_proactive_check")
        except Exception as error:
            logger.warning(f"移除 KFC 主动发起调度失败: {error}")
        finally:
            self._proactive_schedule_id = None
            ScheduleProactiveAction._guidance = ""

        cancelled_compressions = SummaryService.cancel_all()
        if cancelled_compressions:
            logger.info(f"已取消 {cancelled_compressions} 个 KFC 摘要压缩任务")
        logger.info("KFC 插件运行时资源已清理")

    def get_components(self) -> list[type]:
        """获取插件内所有组件类。

        当 ``general.enabled`` 为 False 时，不注册 ``KokoroFlowChatter``，
        使框架的 chatter 选择器看不到 KFC，从而自然由其他 chatter（如
        default_chatter）接管私聊流。其余 Action / EventHandler 组件
        不受影响，仍正常注册。
        """
        config = self.config
        is_enabled = isinstance(config, KFCConfig) and config.general.enabled

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
        if is_enabled:
            components.append(KokoroFlowChatter)
        else:
            logger.info("KFC 插件已禁用，不注册 KokoroFlowChatter 组件")
        return components
