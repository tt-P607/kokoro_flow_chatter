"""KFC 多模态服务。"""

from __future__ import annotations

from src.app.plugin_system.types import LLMPayload, ROLE

from ..multimodal import MediaItem, build_multimodal_content


class MultimodalService:
    """处理运行时多模态上下文拼装。"""

    @staticmethod
    def build_history_reference_payload(
        history_images: list[MediaItem],
    ) -> LLMPayload | None:
        """构造历史图片参考的 SYSTEM payload。

        无图片时返回 ``None``；调用方负责使用 ``response.add_payload`` 或
        ``safe_add_payload`` 追加，本方法保持纯函数（不操作 response）。
        """
        if not history_images:
            return None
        return LLMPayload(
            ROLE.SYSTEM,
            build_multimodal_content("[历史图片参考]", history_images),
        )