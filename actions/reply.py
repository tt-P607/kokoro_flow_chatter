"""KFC 回复动作。

模型的所有可见发言都经由本动作发出。执行层已完成分段拆分与打字节奏
控制，这里只负责把单段文本交给框架发送，并做一道元数据泄漏兜底。
"""

from __future__ import annotations

import re
from typing import Annotated, Any

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.send_api import send_text
from src.app.plugin_system.base import BaseAction

logger = get_logger("kfc_reply")

_METADATA_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:想法|内心想法|思考|thought|thinking)\s*[:：]",
        r"(?:预计反应|预期反应|expected_reaction)\s*[:：]",
        r"(?:最大等待秒数|max_wait_seconds)\s*[:：]",
        r"(?:心情|情绪|mood)\s*[:：]",
    )
)
"""元数据泄漏特征。模型偶尔会把 thought 等参数写进正文，需要拦截。"""

_METADATA_HIT_THRESHOLD = 2
"""命中阈值。要求同时出现多个特征才判定为泄漏，避免误伤正常表达。"""

KFC_METADATA_FIELDS: tuple[str, ...] = (
    "thought",
    "expected_reaction",
    "max_wait_seconds",
    "mood",
)
"""KFC 元数据字段。schema 中强制必填，确保模型每次决策都给出内心活动。"""

_REPLY_PREVIEW_LIMIT = 80


def force_kfc_metadata_required(schema: dict[str, Any]) -> dict[str, Any]:
    """把元数据字段在 schema 中标记为必填。

    ``execute()`` 签名保留默认值是为了兼容运行时由执行层直接调用（只传
    content/reply_to）的场景，避免 TypeError；但暴露给模型的 schema 必须
    强制必填，防止模型省略关键的决策上下文。

    Args:
        schema: 由 ``BaseAction.to_schema()`` 生成的原始 schema。

    Returns:
        dict[str, Any]: 就地修改后的 schema。
    """
    parameters = schema.get("function", {}).get("parameters", {})
    properties = parameters.get("properties", {}) or {}
    required = list(parameters.get("required", []) or [])

    for field_name in KFC_METADATA_FIELDS:
        if field_name not in properties:
            continue
        # 必填字段不应再携带 default 元信息，否则部分 provider 会放宽校验
        properties[field_name].pop("default", None)
        if field_name not in required:
            required.append(field_name)

    parameters["required"] = required
    return schema


def _strip_leaked_metadata(segment: str) -> str:
    """截断正文中泄漏的元数据尾巴。

    Args:
        segment: 待发送的文本。

    Returns:
        str: 清洗后的文本；未检测到泄漏时原样返回。
    """
    matches = [pattern.search(segment) for pattern in _METADATA_PATTERNS]
    hit_positions = [match.start() for match in matches if match is not None]
    if len(hit_positions) < _METADATA_HIT_THRESHOLD:
        return segment

    cleaned = segment[: min(hit_positions)].strip()
    logger.warning(
        f"检测到正文混入 {len(hit_positions)} 个元数据关键字，已截断。"
        f"原始长度={len(segment)}，截断后={len(cleaned)}"
    )
    return cleaned


class KFCReplyAction(BaseAction):
    """发送文本消息给对方。"""

    name = "kfc_reply"
    associated_types: list[str] = ["text"]
    description = (
        "发送文本消息给对方。"
        "content 为消息段落列表，每个元素是一条独立消息，系统会依次发出。"
        "可选的 reply_to 参数允许你引用消息（虽然私聊中较少用到，"
        "但引用旧消息时可能有用）。"
        "注意：本工具无法发送表情包等非文本内容。"
        "**严禁在一次响应中多次调用此工具**。若有多段内容，"
        "请通过 content 列表一次性传入。"
        "**调用时必须明确给出 thought / expected_reaction / max_wait_seconds / mood "
        "这四个字段，它们承载你这次决策的内心活动、对对方反应的预期、"
        "等待时长和当前情绪。**"
    )
    chatter_allow: list[str] = ["kokoro_flow_chatter"]

    @classmethod
    def to_schema(cls) -> dict:  # type: ignore[override]
        """生成 schema，并强制元数据字段必填。"""
        return force_kfc_metadata_required(super().to_schema())

    async def execute(
        self,
        content: Annotated[
            list[str],
            "要发送的消息段落列表，每个元素是一条独立消息，系统会依次发出。",
        ],
        thought: Annotated[
            str,
            "**必填**。你此刻的内心想法和感受，描述你为什么要这样回复。",
        ] = "",
        expected_reaction: Annotated[
            str,
            "**必填**。你期望对方看到你这条消息后的反应。",
        ] = "",
        max_wait_seconds: Annotated[
            float,
            "**必填**。你愿意等待对方回复的最长时间（秒），0 表示不等待。",
        ] = 0.0,
        mood: Annotated[
            str,
            "**必填**。你当前的心情，用一两个词描述。",
        ] = "",
        reply_to: Annotated[str, "可选，要引用回复的消息 ID"] = "",
    ) -> tuple[bool, str]:
        """发送一段文本消息。

        四个元数据参数由执行层提取用于状态记录，本方法不使用它们——
        保留在签名中是为了让 schema 能向模型暴露这些必填字段。

        Args:
            content: 消息内容。执行层调用时传入单段字符串，模型直接
                调用时可能传入列表。
            thought: 内心想法（由执行层消费）。
            expected_reaction: 预期反应（由执行层消费）。
            max_wait_seconds: 等待时长（由执行层消费）。
            mood: 当前心情（由执行层消费）。
            reply_to: 要引用的消息 ID。

        Returns:
            tuple: ``(是否成功, 回执描述)``。
        """
        _ = (thought, expected_reaction, max_wait_seconds, mood)

        if isinstance(content, str):
            segment = content.strip()
        else:
            segment = " ".join(str(item).strip() for item in content if str(item).strip())
        if not segment:
            return False, "内容为空，未发送"

        segment = _strip_leaked_metadata(segment)
        if not segment:
            return False, "清洗后内容为空，未发送"

        if reply_to:
            success = await send_text(
                content=segment,
                stream_id=self.chat_stream.stream_id,
                reply_to=reply_to,
            )
        else:
            success = await self._send_to_stream(segment)

        if not success:
            return False, "消息发送失败"
        return True, f"已发送消息: {segment[:_REPLY_PREVIEW_LIMIT]}"
