"""KFC 回复动作。

包含最后一道防线：防御性清洗 content 中混入的元数据。
"""

from __future__ import annotations

import re
from typing import Annotated

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.send_api import send_text
from src.app.plugin_system.base import BaseAction

logger = get_logger("kfc_reply")

# 元数据关键字模式（最后防线）
# 仅当多个元数据关键字同时出现时才判定为泄漏，降低误伤概率
_METADATA_KEYWORDS = [
    r"(?:想法|内心想法|思考|thought|thinking)\s*[:：]",
    r"(?:预计反应|预期反应|expected_reaction)\s*[:：]",
    r"(?:最大等待秒数|max_wait_seconds)\s*[:：]",
    r"(?:心情|情绪|mood)\s*[:：]",
]
_METADATA_PATTERNS = [re.compile(kw, re.IGNORECASE) for kw in _METADATA_KEYWORDS]


# KFC 元数据字段：通过 schema 强制 LLM 每次调用都明确给出，
# 但 execute() 签名保留默认值以兼容运行时只传 content/reply_to 的调用路径。
# parser 层会在 LLM 返回的 args 中把这些字段提取走（extract_metadata），
# 不会真的把这些值传到 execute()。
_KFC_METADATA_REQUIRED_FIELDS: tuple[str, ...] = (
    "thought",
    "expected_reaction",
    "max_wait_seconds",
    "mood",
)


def _force_kfc_metadata_required(schema: dict) -> dict:
    """把 KFC 元数据字段在 schema 里标成 required。

    通用工具：从 schema 的 properties 里取出元数据字段对应的描述，
    并把它们追加到 required 列表（去重保序）。
    """
    func = schema.get("function", {})
    params = func.get("parameters", {})
    properties = params.get("properties", {}) or {}
    existing_required = list(params.get("required", []) or [])

    # 元数据字段如果没在 properties（理论不会），跳过，避免 schema 不一致
    for field_name in _KFC_METADATA_REQUIRED_FIELDS:
        if field_name not in properties:
            continue
        # 同时把字段的 default 字段去掉（required 字段不应该有 default 元信息）
        properties[field_name].pop("default", None)
        if field_name not in existing_required:
            existing_required.append(field_name)

    params["required"] = existing_required
    return schema


class KFCReplyAction(BaseAction):
    """发送文本消息给对方。"""

    action_name = "kfc_reply"
    associated_types: list[str] = ["text"]
    action_description = (
        "发送文本消息给对方。"
        "content 为消息段落列表，每个元素是一条独立消息，系统会依次发出。"
        "可选的 reply_to 参数允许你引用消息（虽然私聊中较少用到，但引用旧消息时可能有用）。"
        "注意：本工具无法发送表情包等非文本内容。"
        "**调用时必须明确给出 thought / expected_reaction / max_wait_seconds / mood 这四个字段，"
        "它们承载你这次决策的内心活动、对对方反应的预期、等待时长和当前情绪。**"
    )

    chatter_allow: list[str] = ["kokoro_flow_chatter"]

    @classmethod
    def to_schema(cls) -> dict:  # type: ignore[override]
        """把 KFC 元数据字段在 schema 里标记为 required。

        execute() 签名保留默认值是为了兼容运行时由 chatter 直接调用
        （只传 content/reply_to）的场景，避免 TypeError；但暴露给 LLM
        的 schema 里这些字段必须必填，防止模型遗漏关键决策上下文。
        """
        return _force_kfc_metadata_required(super().to_schema())

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
        """执行发送文本消息的逻辑。

        parser 层已处理分段拆分逻辑，此处仅作兜底：直接发送完整内容。
        reply_to 为可选的引用消息 ID。
        """
        # thought/expected_reaction/max_wait_seconds/mood 由 chatter.py 的策略层提取，
        # action 本身不使用这些参数

        # parser 层已处理分段拆分逻辑；此处仅作兜底，直接发送完整内容
        # content 注解为 list[str]（供 LLM schema 使用），运行时由 parser 传入单条字符串
        if isinstance(content, str):
            segment = content.strip()
        else:
            segment = " ".join(str(s).strip() for s in content if str(s).strip())
        if not segment:
            return False, "内容为空，未发送"

        # 最后防线：仅当 >=2 个元数据关键字同时出现时才截断，降低误伤
        keyword_matches = [
            p.search(segment) for p in _METADATA_PATTERNS
        ]
        hit_count = sum(1 for m in keyword_matches if m is not None)
        if hit_count >= 2:
            # 找到最早的匹配位置进行截断
            earliest = min(
                (m.start() for m in keyword_matches if m is not None),
            )
            cleaned = segment[:earliest].strip()
            logger.warning(
                f"[最后防线] 检测到 content 中混入 {hit_count} 个元数据关键字，已截断。"
                f"原始长度={len(segment)}，截断后={len(cleaned)}"
            )
            segment = cleaned
            if not segment:
                return False, "清洗后内容为空，未发送"

        # reply_to 非空时用 send_api 发带引用的消息，否则走标准 _send_to_stream
        if reply_to:
            success = await send_text(
                content=segment,
                stream_id=self.chat_stream.stream_id,
                reply_to=reply_to,
            )
            return success, f"已发送消息: {segment[:80]}"

        success = await self._send_to_stream(segment)
        if not success:
            return False, "消息发送失败"
        return True, f"已发送消息: {segment[:80]}"


