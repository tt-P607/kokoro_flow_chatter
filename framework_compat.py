"""KFC 对框架暂未公开运行时能力的集中兼容边界。

本模块只封装 native multimodal 的流级识别跳过，以及主动发起冷启动时的
流循环启动。公开插件 API 尚未覆盖这两项能力，因此调用方通过这里集中处理
内部路径依赖，避免业务模块散落导入框架管理器。
"""

from __future__ import annotations

from collections.abc import Iterable


def set_stream_recognition_skip(
    stream_id: str,
    media_types: Iterable[str] | None = None,
) -> None:
    """为聊天流设置媒体识别跳过。

    Args:
        stream_id: 聊天流 ID。
        media_types: 需要跳过的媒体类型；为 ``None`` 时沿用管理器默认行为。

    Raises:
        ValueError: ``stream_id`` 为空。
        ImportError: 当前框架未提供兼容的 MediaManager。
    """

    if not stream_id.strip():
        raise ValueError("stream_id 不能为空")

    from src.core.managers.media_manager import get_media_manager

    normalized_types = list(media_types) if media_types is not None else None
    if normalized_types is None:
        get_media_manager().skip_recognition_for_stream(stream_id)
        return
    get_media_manager().skip_recognition_for_stream(stream_id, normalized_types)


def clear_stream_recognition_skip(stream_id: str) -> None:
    """移除聊天流的媒体识别跳过。

    Args:
        stream_id: 聊天流 ID。

    Raises:
        ValueError: ``stream_id`` 为空。
        ImportError: 当前框架未提供兼容的 MediaManager。
    """

    if not stream_id.strip():
        raise ValueError("stream_id 不能为空")

    from src.core.managers.media_manager import get_media_manager

    get_media_manager().unskip_recognition_for_stream(stream_id)


async def start_stream_loop(stream_id: str) -> None:
    """启动指定聊天流的运行循环。

    Args:
        stream_id: 聊天流 ID。

    Raises:
        ValueError: ``stream_id`` 为空。
        ImportError: 当前框架未提供兼容的 StreamLoopManager。
    """

    if not stream_id.strip():
        raise ValueError("stream_id 不能为空")

    from src.core.transport.distribution.stream_loop_manager import (
        get_stream_loop_manager,
    )

    await get_stream_loop_manager().start_stream_loop(stream_id)


__all__ = [
    "clear_stream_recognition_skip",
    "set_stream_recognition_skip",
    "start_stream_loop",
]
