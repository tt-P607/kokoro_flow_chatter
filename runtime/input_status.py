"""KFC「正在输入」状态上报。

在 LLM 生成期间向客户端上报输入状态，让对方看到"对方正在输入…"，
使等待过程更接近真人对话节奏。

该能力目前仅 SnowLuma 适配器提供，因此本模块是 KFC 与特定平台之间的
唯一耦合点；适配器缺失或调用失败时静默降级，不影响对话主流程。
"""

from __future__ import annotations

from typing import Any

from src.app.plugin_system.api.log_api import get_logger

logger = get_logger("kfc_input_status")

_SNOWLUMA_ADAPTER_SIGN = "snowluma_adapter:adapter:snowluma_adapter"
_INPUT_STATUS_API = "set_input_status"
_API_TIMEOUT_SECONDS = 5.0

_STATUS_TYPING = 1
_STATUS_IDLE = 0


class InputStatusReporter:
    """LLM 生成期间的「正在输入」状态上报器。

    用法为一次生成配一个实例：``await start()`` 发送一次 typing，
    ``await stop()`` 撤下状态。不做周期刷新，避免频繁 API 调用。
    """

    def __init__(self, stream_id: str, user_id: str) -> None:
        """初始化上报器。

        Args:
            stream_id: 当前聊天流 ID（未使用，保留参数兼容性）。
            user_id: 目标用户 ID，必须是纯数字的平台账号。
        """
        self._stream_id = stream_id
        self._user_id = user_id
        self._started = False

    @property
    def is_available(self) -> bool:
        """当前会话是否具备上报条件。"""
        return bool(self._user_id and self._user_id.isdigit())

    async def start(self) -> None:
        """发送一次「正在输入」状态。"""
        if not self.is_available:
            return
        self._started = True
        await self._send_status(_STATUS_TYPING)

    async def stop(self) -> None:
        """撤下「正在输入」状态。"""
        if not self._started:
            return
        self._started = False
        if self.is_available:
            await self._send_status(_STATUS_IDLE)

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
