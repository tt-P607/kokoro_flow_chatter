"""KFC 原生多模态支持。

原生多模态模式下，图片以原始数据直接进入 LLM payload，由主模型在对话
上下文中理解——比框架 VLM 管线的"转述为文字"保留更多信息。

表情包仍走 VLM 文字描述路径，以复用其哈希缓存，因此这里显式排除。

本模块保持纯函数形态，不依赖运行时单例，便于单测覆盖。
"""

from __future__ import annotations

from typing import Any

from src.app.plugin_system.types import Content, Image, Text

_IMAGE_MEDIA_TYPE = "image"


def _extract_dict_items(raw: Any) -> list[dict[str, Any]]:
    """把原始值转换为仅含 dict 元素的列表。"""
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _read_raw_media(msg: Any) -> list[dict[str, Any]]:
    """读取消息中尚未被剥离原始数据的 media 列表。

    按优先级检查两个候选位置：

    1. ``msg.content["media"]``——要求至少一项含 ``data``（完整媒体）；
    2. ``msg.extra["media"]``——非空即可。

    流管理器在持久化时会剔除超大的 ``data`` 字段，但本函数仅在 Chatter
    运行期内调用，因此能拿到完整字节。
    """
    content = msg.content
    if isinstance(content, dict):
        items = _extract_dict_items(content.get("media"))
        if items and any(item.get("data") for item in items):
            return items

    extra = msg.extra
    if isinstance(extra, dict):
        items = _extract_dict_items(extra.get("media"))
        if items:
            return items

    return []


def get_image_media_list(msg: Any) -> list[dict[str, Any]]:
    """提取消息中的图片媒体项。

    Args:
        msg: 待检查的消息。

    Returns:
        list[dict[str, Any]]: 含有效数据的 ``image`` 类型媒体项。
    """
    return [
        item
        for item in _read_raw_media(msg)
        if item.get("type") == _IMAGE_MEDIA_TYPE and item.get("data")
    ]


def extract_images_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """按消息顺序提取全部图片。

    Args:
        messages: 待扫描的消息列表。

    Returns:
        list[dict[str, Any]]: 保持原始时序的图片媒体项列表。
    """
    items: list[dict[str, Any]] = []
    for msg in messages:
        items.extend(get_image_media_list(msg))
    return items


def build_multimodal_content(
    text: str,
    media_items: list[dict[str, Any]],
) -> list[Content]:
    """把文本与图片打包为 payload 可接受的内容列表。

    Args:
        text: 文本主体。
        media_items: 按时序排列的图片媒体项。

    Returns:
        list[Content]: ``[Text, Image, Image, ...]`` 形式的内容列表。
    """
    content_list: list[Content] = [Text(text)]
    content_list.extend(Image(str(item["data"])) for item in media_items)
    return content_list
