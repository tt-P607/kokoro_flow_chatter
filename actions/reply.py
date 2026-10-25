"""KFC 文本与通话语音回复动作。

文本回复交给框架发送，通话语音回复通过公开语音服务合成并播放。
"""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar
from typing import Annotated, Any, ClassVar

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.send_api import format_send_result, send_text_result
from src.app.plugin_system.api.service_api import get_service
from src.app.plugin_system.base import BaseAction

from ..models import KFC_VOICE_REPLY

logger = get_logger("kfc_reply")
VOICE_REPLY_TRACE: ContextVar[str] = ContextVar("kfc_voice_reply_trace", default="")

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
        "即使正在语音通话，本动作仍发送文字，不触发语音合成。"
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
            result = await send_text_result(
                content=segment,
                stream_id=self.chat_stream.stream_id,
                reply_to=reply_to,
            )
            success, receipt = format_send_result(result)
        else:
            result = await self._send_to_stream_result(segment)
            success, receipt = format_send_result(result)

        return success, f"{receipt}: {segment[:_REPLY_PREVIEW_LIMIT]}"


async def execute_voice_reply(
    chat_stream: Any,
    content: str,
    voice_style: str,
    *,
    trace_id: str = "",
) -> tuple[bool, str]:
    """在当前已接通的 QQ 私聊通话中发送一段语音。

    Args:
        chat_stream: 当前 KFC 聊天流。
        content: 仅供朗读的正文。
        voice_style: 传给语音服务的风格选择参数。
        trace_id: 本次语音动作的关联标识，不包含正文。

    Returns:
        tuple[bool, str]: 发送状态和语音服务回执。
    """
    stream_id = str(getattr(chat_stream, "stream_id", "") or "")
    if (
        not stream_id
        or getattr(chat_stream, "platform", None) != "qq"
        or str(getattr(chat_stream, "chat_type", "")) != "private"
    ):
        return False, "仅支持当前 QQ 私聊通话，未发送"

    service = get_service("anima_voice:service:voice_call")
    if service is None:
        return False, "Anima Voice 通话服务不可用，未发送"
    state = service.get_state(stream_id)
    if (
        not isinstance(state, dict)
        or state.get("stream_id") != stream_id
        or state.get("phase") != "connected"
    ):
        return False, "当前 QQ 私聊没有已接通的通话，未发送"

    try:
        result = await service.speak(
            stream_id,
            content,
            voice_style=voice_style,
            **({"trace_id": trace_id} if trace_id else {}),
        )
    except Exception as error:  # noqa: BLE001
        logger.warning(f"通话语音发送失败: {error}")
        return False, f"通话语音发送失败: {error}"
    if result.get("drained") is not True:
        return False, "通话语音未确认完成发送"
    return True, "通话语音已完成发送"


class KFCVoiceReplyAction(BaseAction):
    """通过当前 QQ 私聊通话朗读回复正文。"""

    name = KFC_VOICE_REPLY
    associated_types: ClassVar[list[str]] = ["text"]
    description = (
        "仅在当前同一 QQ 私聊已接通通话时，通过通话发送语音回复；"
        "若没有可用的已接通通话会明确失败，不发送文字，也不使用本机扬声器。"
        "与 kfc_reply 的文本回复相互独立，可在同一轮分别调用；"
        "每轮本动作最多调用一次，content 为需要朗读的完整正文字符串。"
        "全文一次提交并实时流式播音，不拆成多次发送。voice_style 用于选择语音风格，"
        "只作为合成参数，绝不要将 voice_style 的值读出来。"
        "必须提供 thought、expected_reaction、max_wait_seconds 和 mood。"
    )
    chatter_allow: ClassVar[list[str]] = ["kokoro_flow_chatter"]

    @classmethod
    def to_schema(cls) -> dict:  # type: ignore[override]
        """生成带必填 KFC 元数据的动作 schema。"""
        return force_kfc_metadata_required(super().to_schema())

    async def execute(
        self,
        content: Annotated[str, "要朗读的完整正文字符串，不包含语音风格说明。"],
        voice_style: Annotated[str, "语音风格选择参数，不会被朗读。"] = "",
        thought: Annotated[str, "你此刻的内心想法和感受。"] = "",
        expected_reaction: Annotated[str, "你预期对方听到回复后的反应。"] = "",
        max_wait_seconds: Annotated[float, "你愿意等待对方回复的最长秒数。"] = 0.0,
        mood: Annotated[str, "你当前的心情。"] = "",
    ) -> tuple[bool, str]:
        """将完整正文一次提交到当前已接通的 QQ 私聊通话。"""
        _ = (thought, expected_reaction, max_wait_seconds, mood)
        text = content.strip()
        if not text:
            return False, "内容为空，未发送"
        trace_id = VOICE_REPLY_TRACE.get() or uuid.uuid4().hex
        logger.info(
            f"KFC 语音回复：trace_id={trace_id}, text_chars={len(text)}"
        )
        return await execute_voice_reply(self.chat_stream, text, voice_style, trace_id=trace_id)
