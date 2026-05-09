"""KFC 原生多模态辅助模块。

提供与 DFC 多模态相同语义的图片提取与 LLM 内容拼装能力。
表情包仍交由框架 VLM 走文字描述路径，以利用其哈希缓存。

模块保持纯函数形态，不依赖运行时单例，便于单测覆盖。
"""

from __future__ import annotations

from typing import Any

from src.app.plugin_system.types import Content, Image, Text


def _extract_dict_list(raw: Any) -> list[dict[str, Any]] | None:
    """将原始值转换为仅含 dict 元素的列表；非列表或空列表返回 None。"""
    if isinstance(raw, list) and raw:
        return [item for item in raw if isinstance(item, dict)]
    return None


def _read_raw_media(msg: Any) -> list[dict[str, Any]]:
    """读取消息中尚未被剥离 base64 的原始 media 列表。

    按优先级依次检查三个候选位置：
    1. ``msg.content["media"]`` — 要求至少一项含 ``data`` 字段（完整媒体）
    2. ``msg.extra["media"]`` — 非空列表即可
    3. ``msg.media`` 属性 — 非空列表即可

    stream_manager 持久化时会剔除超大 ``data``，此处仅在 Chatter 运行期
    内使用，因此能拿到完整字节。
    """
    content = msg.content
    if isinstance(content, dict):
        items = _extract_dict_list(content.get("media"))
        if items and any(item.get("data") for item in items):
            return items

    extra = msg.extra
    if isinstance(extra, dict):
        items = _extract_dict_list(extra.get("media"))
        if items:
            return items

    media = getattr(msg, "media", None)
    if isinstance(media, list) and media:
        return [item for item in media if isinstance(item, dict)]

    return []


def get_image_media_list(msg: Any) -> list[dict[str, Any]]:
    """从消息中提取仅包含 ``image`` 类型的媒体列表。

    表情包继续走 VLM 文字描述（受益于哈希缓存），
    因此这里显式过滤掉 ``emoji`` / ``voice`` 等非图片类型。
    """
    media = _read_raw_media(msg)
    return [item for item in media if item.get("type") == "image" and item.get("data")]


def extract_images_from_messages(
    messages: list[Any],
) -> list[dict[str, Any]]:
    """按顺序从消息列表中提取全部图片。

    Args:
        messages: 待扫描的消息（可为未读消息或历史消息子集）

    Returns:
        提取到的媒体字典列表，保持原始消息顺序
    """
    items: list[dict[str, Any]] = []
    for msg in messages:
        for media in get_image_media_list(msg):
            items.append(media)
    return items


def build_multimodal_content(
    text: str,
    media_items: list[dict[str, Any]],
) -> list[Content]:
    """将文本与图片打包为 LLMPayload 可接受的 content 列表。

    Args:
        text: 文本主体
        media_items: 按消息时序排列的图片媒体字典列表

    Returns:
        ``[Text(text), Image(data1), Image(data2), ...]`` 格式的内容列表
    """
    content_list: list[Content] = [Text(text)]
    for item in media_items:
        content_list.append(Image(str(item["data"])))
    return content_list
