"""KFC 多模态服务。"""

from __future__ import annotations

from typing import Any

from src.app.plugin_system.types import LLMPayload, ROLE

from ..multimodal import MediaItem, build_multimodal_content


class MultimodalService:
    """处理运行时多模态上下文拼装。"""

    @staticmethod
    def append_history_reference(response: Any, history_images: list[MediaItem]) -> None:
        """将历史图片参考追加到 response 链。"""
        if not history_images:
            return
        response.add_payload(
            LLMPayload(
                ROLE.SYSTEM,
                build_multimodal_content("[历史图片参考]", history_images),
            )
        )