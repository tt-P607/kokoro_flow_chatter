"""KFC 多模态辅助模块。

高内聚：所有图片提取、格式转换逻辑集中在此。
低耦合：仅依赖 Message 对象的公开属性和 kernel.llm 的 Content 类型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from src.kernel.llm import Content, Image, Text

if TYPE_CHECKING:
    from src.core.models.message import Message


@dataclass
class MediaItem:
    """从消息中提取的媒体条目。"""

    media_type: str  # "image" | "emoji"
    base64_data: str  # 原始 base64 数据（"base64|..." 格式，来自 normalize_base64）
    source_message_id: str  # 来源消息 ID


def extract_media_from_messages(
    messages: list[Message],
    max_items: int = 4,
) -> list[MediaItem]:
    """从未读消息列表中提取图片/表情包的 base64 数据。

    只提取当前轮的未读消息中的 media。历史消息中的图片
    已经在 LLMResponse 链的上下文中，不需要重复提取。

    Args:
        messages: 当前轮的未读消息列表
        max_items: 最大提取数量

    Returns:
        提取到的 MediaItem 列表（按消息顺序，截断至 max_items）
    """
    items: list[MediaItem] = []

    for msg in messages:
        if len(items) >= max_items:
            break

        media_list = _get_media_list(msg)
        if not media_list:
            continue

        msg_id = getattr(msg, "message_id", "")

        for media in media_list:
            if len(items) >= max_items:
                break
            if media.get("type") not in ("image", "emoji"):
                continue
            data = media.get("data", "")
            if not data:
                continue

            items.append(
                MediaItem(
                    media_type=media["type"],
                    base64_data=data,
                    source_message_id=msg_id,
                )
            )

    return items


def build_multimodal_content(
    text: str,
    media_items: list[MediaItem],
) -> list[Content]:
    """构建混合 Text + Image 的 content 列表，用于 LLMPayload。

    Args:
        text: 文本内容
        media_items: 媒体条目列表

    Returns:
        [Text(text), Image(data1), Image(data2), ...] 格式的 content 列表

    Note:
        MediaItem.base64_data 已经是 ``"base64|..."`` 格式
        （来自 converter 的 ``normalize_base64``）。
        框架的 ``openai_client._image_to_data_url`` 会自动将其
        转换为 ``"data:image/png;base64,..."`` 格式。
    """
    content_list: list[Content] = [Text(text)]
    for item in media_items:
        # 表情包类型添加标注，帮助模型区分贴纸/表情包与普通照片
        if item.media_type == "emoji":
            content_list.append(Text("[表情包]"))
        content_list.append(Image(item.base64_data))
    return content_list


# ──────────────────────────────────────────
# 内部辅助函数
# ──────────────────────────────────────────


def _get_media_list(msg: Message) -> list[dict[str, Any]]:
    """从 Message 中提取 media 列表。

    按优先级尝试三种路径获取媒体数据。

    Args:
        msg: 消息对象

    Returns:
        媒体字典列表，每项为 ``{"type": str, "data": str}``
    """
    # 路径 1: content 是 dict（含媒体消息）
    content = getattr(msg, "content", None)
    if isinstance(content, dict):
        media = content.get("media")
        if isinstance(media, list) and media:
            return media

    # 路径 2: extra 中的 media（converter 构造时通过 **extra 传入）
    extra = getattr(msg, "extra", {})
    if isinstance(extra, dict):
        media = extra.get("media")
        if isinstance(media, list) and media:
            return media

    # 路径 3: 直接属性（**extra 展开后成为实例属性）
    media = getattr(msg, "media", None)
    if isinstance(media, list) and media:
        return media

    # 路径 4: EMOJI 类型消息的原始 content（base64 字符串）
    # Bot 发送的表情包通过 send_api 构建，content 是原始 base64 数据
    msg_type = getattr(msg, "message_type", None)
    if (
        msg_type is not None
        and str(msg_type) == "emoji"
        and isinstance(content, str)
        and len(content) > 100  # base64 图片数据通常远大于 100 字符
    ):
        # 统一为 "base64|..." 格式
        data = content if content.startswith("base64|") else f"base64|{content}"
        return [{"type": "emoji", "data": data}]

    return []
