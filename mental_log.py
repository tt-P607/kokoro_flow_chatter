"""心理活动流。

``MentalLogEntry`` 记录活动流中的单个事件节点；``MentalLog`` 作为容器
负责条目的追加、去重、上限裁剪与序列化。

活动流的渲染不在本模块内完成——融合叙事由
``context.sources.history_source.build_fused_narrative()`` 按时间线
与聊天记录交织后统一产出。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .models import KFC_REPLY, KFCEventType


@dataclass
class MentalLogEntry:
    """心理活动流中的单个事件节点。"""

    event_type: KFCEventType
    timestamp: float

    content: str = ""
    """通用文本内容，语义随 ``event_type`` 变化。"""

    user_name: str = ""
    user_id: str = ""
    message_id: str = ""
    """``USER_MESSAGE`` 专用：发送者信息与消息 ID（消息 ID 同时用于去重）。"""

    thought: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    expected_reaction: str = ""
    max_wait_seconds: float = 0.0
    """``BOT_PLANNING`` 专用：本轮决策的内心活动与动作快照。"""

    elapsed_seconds: float = 0.0
    """等待相关事件专用：已等待秒数。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """附加信息，如回复时效标记、备忘 id 等。"""

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "event_type": str(self.event_type),
            "timestamp": self.timestamp,
            "content": self.content,
            "user_name": self.user_name,
            "user_id": self.user_id,
            "message_id": self.message_id,
            "thought": self.thought,
            "actions": self.actions,
            "expected_reaction": self.expected_reaction,
            "max_wait_seconds": self.max_wait_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MentalLogEntry:
        """从字典反序列化；未知事件类型回退为 ``USER_MESSAGE``。"""
        try:
            event_type = KFCEventType(data.get("event_type", "user_message"))
        except ValueError:
            event_type = KFCEventType.USER_MESSAGE

        return cls(
            event_type=event_type,
            timestamp=data.get("timestamp", time.time()),
            content=data.get("content", ""),
            user_name=data.get("user_name", ""),
            user_id=data.get("user_id", ""),
            message_id=data.get("message_id", ""),
            thought=data.get("thought", ""),
            actions=data.get("actions", []),
            expected_reaction=data.get("expected_reaction", ""),
            max_wait_seconds=float(data.get("max_wait_seconds", 0)),
            elapsed_seconds=float(data.get("elapsed_seconds", 0)),
            metadata=data.get("metadata", {}),
        )


class MentalLog:
    """心理活动流容器，管理条目的追加、查询与上限裁剪。"""

    def __init__(self, max_entries: int = 50) -> None:
        """初始化容器。

        Args:
            max_entries: 保留的最大条目数，超出时从头裁剪。
        """
        self._entries: list[MentalLogEntry] = []
        self._max_entries = max_entries

    @property
    def entries(self) -> list[MentalLogEntry]:
        """返回所有条目的只读副本。"""
        return list(self._entries)

    def __len__(self) -> int:
        """返回当前条目数。"""
        return len(self._entries)

    def add(self, entry: MentalLogEntry) -> None:
        """追加条目，超出上限时裁剪最旧的。

        对 ``USER_MESSAGE`` 按 ``message_id`` 做幂等去重：LLM 调用失败后
        下一 Tick 会重新消费同一批未读，若不去重，同一条用户消息会被反复
        写入并撑爆活动流。其它事件类型允许重复——它们记录的是时间序列上
        不同的决策瞬间。
        """
        if (
            entry.event_type == KFCEventType.USER_MESSAGE
            and entry.message_id
            and any(
                existing.event_type == KFCEventType.USER_MESSAGE
                and existing.message_id == entry.message_id
                for existing in self._entries
            )
        ):
            return
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]

    def get_last_bot_reply_content(self) -> str:
        """返回最近一次 Bot 回复的文本内容；无回复时返回空串。

        供超时处理构建"你最后说的是什么"的上下文。
        """
        for entry in reversed(self._entries):
            if entry.event_type != KFCEventType.BOT_PLANNING:
                continue
            for action in entry.actions:
                if action.get("type") != KFC_REPLY:
                    continue
                content = action.get("content", "")
                if isinstance(content, list):
                    joined = " ".join(str(item) for item in content if item)
                    if joined:
                        return joined
                elif isinstance(content, str) and content:
                    return content
        return ""

    def to_list(self) -> list[dict[str, Any]]:
        """序列化为字典列表。"""
        return [entry.to_dict() for entry in self._entries]

    @classmethod
    def from_list(
        cls,
        data: list[dict[str, Any]],
        max_entries: int = 50,
    ) -> MentalLog:
        """从字典列表反序列化，并裁剪到上限。"""
        log = cls(max_entries=max_entries)
        for item in data:
            log._entries.append(MentalLogEntry.from_dict(item))
        if len(log._entries) > max_entries:
            log._entries = log._entries[-max_entries:]
        return log
