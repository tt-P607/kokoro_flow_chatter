"""KokoroFlowChatter 插件入口。

注册插件、加载配置、注册提示词模板、初始化 Scheduler 任务。
"""

from __future__ import annotations

from src.app.plugin_system.api.log_api import get_logger
from src.core.components.base import BasePlugin
from src.core.components.loader import register_plugin
from src.kernel.concurrency import get_task_manager

from .actions.do_nothing import DoNothingAction
from .actions.reply import KFCReplyAction
from .chatter import KokoroFlowChatter
from .config import KFCConfig
from .handlers.proactive_handler import ProactiveHandler
from .session import KFCSessionStore

logger = get_logger("kfc_plugin")


@register_plugin
class KFCPlugin(BasePlugin):
    """KokoroFlowChatter 插件。"""

    plugin_name = "kokoro_flow_chatter"
    plugin_version = "2.0.0"
    plugin_author = "MoFox Team"
    plugin_description = "心理活动流聊天器，模拟真实人类的连续心理活动和对话节奏"
    configs = [KFCConfig]

    def __init__(self, config: KFCConfig | None = None) -> None:
        super().__init__(config)
        self._session_store = KFCSessionStore()

    async def on_plugin_loaded(self) -> None:
        """插件加载时注册提示词模板。调度任务延迟到调度器启动后注册。"""
        # 注册提示词模板
        from .prompts.modules import register_kfc_prompts

        register_kfc_prompts()
        logger.info("KFC 提示词模板已注册")

        # 预注册已知聊天流的 VLM 跳过（原生多模态模式）
        config = self.config
        if isinstance(config, KFCConfig) and config.general.native_multimodal:
            await self._preload_vlm_skip()

        # 延迟注册调度器任务：等待调度器启动
        get_task_manager().create_task(
            self._delayed_scheduler_register(),
            name="kfc_scheduler_init",
            daemon=True,
        )

        logger.info("KFC 插件已加载")

    async def _delayed_scheduler_register(self) -> None:
        """延迟注册调度器任务，等待调度器启动（最多 30 秒）。"""
        import asyncio

        for _ in range(30):
            await asyncio.sleep(1.0)
            try:
                from src.kernel.scheduler import get_unified_scheduler

                scheduler = get_unified_scheduler()
                if scheduler._running:  # HACK: 需要框架公开 API (scheduler.is_running)
                    await self._register_scheduler_tasks()
                    return
            except ImportError:
                logger.warning("Scheduler 不可用，放弃注册")
                return
        logger.warning("等待调度器启动超时(30s)，放弃注册后台任务")

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

            from src.core.managers import get_media_manager

            media_manager = get_media_manager()
            for stream_id in stream_ids:
                media_manager.skip_vlm_for_stream(stream_id)

            logger.info(
                f"已预注册 {len(stream_ids)} 个聊天流的 VLM 跳过"
            )
        except Exception as e:
            logger.warning(f"预加载 VLM 跳过失败（不影响功能）: {e}")

    async def _register_scheduler_tasks(self) -> None:
        """注册后台调度任务。"""
        config = self.config
        if not isinstance(config, KFCConfig):
            return

        try:
            from src.kernel.scheduler import get_unified_scheduler, TriggerType

            scheduler = get_unified_scheduler()
        except ImportError:
            logger.warning("Scheduler 不可用，跳过后台任务注册")
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
                    await proactive.mark_triggered(stream_id)
                    logger.info(f"主动发起触发: {stream_id[:8]}")
                    # 通过事件总线触发 chatter
                    try:
                        from src.kernel.event import get_event_bus

                        bus = get_event_bus()
                        await bus.publish(
                            "kfc.proactive_trigger",
                            {"stream_id": stream_id},
                        )
                    except ImportError:
                        logger.debug("事件总线不可用")

            # 注册周期性主动发起检查任务
            await scheduler.create_schedule(
                callback=proactive_check,
                trigger_type=TriggerType.TIME,
                trigger_config={"delay_seconds": config.proactive.check_interval},
                is_recurring=True,
                task_name="kfc_proactive_check",
                force_overwrite=True,
            )

        # 等待检查（连续思考）
        if config.continuous_thinking.enabled:
            from .thinker.wait_checker import WaitChecker

            wait_checker = WaitChecker(config=config)

            async def wait_check() -> None:
                """定期检查等待中的 Session 并触发连续思考。"""
                sessions = self._session_store.get_all_cached()
                for stream_id, session in sessions.items():
                    if session.is_waiting():
                        async with self._session_store.lock(stream_id):
                            await wait_checker.check_and_think(session)
                            await self._session_store.save(session)

            # 注册周期性连续思考检查任务
            await scheduler.create_schedule(
                callback=wait_check,
                trigger_type=TriggerType.TIME,
                trigger_config={"delay_seconds": int(config.continuous_thinking.min_interval)},
                is_recurring=True,
                task_name="kfc_wait_check",
                force_overwrite=True,
            )

        logger.info("KFC 调度器任务注册完成")

    def get_components(self) -> list[type]:
        """获取插件内所有组件类。"""
        return [
            KokoroFlowChatter,
            KFCReplyAction,
            DoNothingAction,
            ProactiveHandler,
        ]
