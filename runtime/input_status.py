"""KFC「正在输入」状态上报。

在 LLM 生成期间向客户端上报输入状态，让对方看到"对方正在输入…"，
使等待过程更接近真人对话节奏。

该能力目前仅 SnowLuma 适配器提供，因此本模块是 KFC 与特定平台之间的
唯一耦合点；适配器缺失或调用失败时静默降级，不影响对话主流程。
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

from src.app.plugin_system.api.log_api import get_logger
from src.kernel.concurrency import get_task_manager

logger = get_logger("kfc_input_status")

_SNOWLUMA_ADAPTER_SIGN = "snowluma_adapter:adapter:snowluma_adapter"
_INPUT_STATUS_API = "set_input_status"
_API_TIMEOUT_SECONDS = 5.0

_STATUS_TYPING = 1
_STATUS_IDLE = 0

_REFRESH_BASE_SECONDS = 2.5
"""客户端的输入状态只维持 2~3 秒，需按此间隔持续刷新。"""

_REFRESH_JITTER_RANGE = (0.5, 1.5)
"""刷新间隔的随机抖动，避免固定周期显得机械。"""

_PLUGIN_NAME = "kokoro_flow_chatter"


class InputStatusReporter:
    """LLM 生成期间的「正在输入」状态上报器。

    用法为一次生成配一个实例：``await start()`` 开始上报并拉起刷新
    任务，``await stop()`` 撤下状态并取消任务。
    """

    def __init__(self, stream_id: str, user_id: str) -> None:
        """初始化上报器。

        Args:
            stream_id: 当前聊天流 ID，用于命名后台任务。
            user_id: 目标用户 ID，必须是纯数字的平台账号。
        """
        self._stream_id = stream_id
        self._user_id = user_id
        self._refresh_task_id: str | None = None

    @property
    def is_available(self) -> bool:
        """当前会话是否具备上报条件。"""
        return bool(self._user_id and self._user_id.isdigit())

    async def start(self) -> None:
        """上报「正在输入」并拉起周期刷新任务。"""
        if not self.is_available:
            return

        await self._send_status(_STATUS_TYPING)
        task_info = get_task_manager().create_task(
            self._refresh_loop(),
            name=f"kfc_input_status_{self._stream_id[:8]}",
            daemon=True,
            metadata={
                "plugin": _PLUGIN_NAME,
                "purpose": "input_status",
                "stream_id": self._stream_id,
            },
        )
        self._refresh_task_id = task_info.task_id

    async def stop(self) -> None:
        """取消刷新任务并撤下「正在输入」状态。"""
        if self._refresh_task_id is not None:
            get_task_manager().cancel_task(self._refresh_task_id)
            self._refresh_task_id = None
        if self.is_available:
            await self._send_status(_STATUS_IDLE)

    async def _refresh_loop(self) -> None:
        """按带抖动的间隔持续刷新输入状态。"""
        while True:
            await asyncio.sleep(
                _REFRESH_BASE_SECONDS + random.uniform(*_REFRESH_JITTER_RANGE)
            )
            await self._send_status(_STATUS_TYPING)

    async def _send_status(self, event_type: int) -> None:
        """向适配器发送一次状态上报；任何失败都静默降级。"""
        try:
            from src.app.plugin_system.api.adapter_api import get_adapter

            adapter: Any = get_adapter(_SNOWLUMA_ADAPTER_SIGN)
            if adapter is None:
                logger.debug("跳过输入状态上报：SnowLuma 适配器未启动")
                return

            await adapter.send_snowluma_api(
                _INPUT_STATUS_API,
                {"user_id": int(self._user_id), "event_type": event_type},
                timeout=_API_TIMEOUT_SECONDS,
            )
        except Exception as error:
            logger.debug(f"输入状态上报失败（不影响对话）: {error}")
