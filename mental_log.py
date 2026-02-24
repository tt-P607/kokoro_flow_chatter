"""心理活动流管理。

MentalLog 容器负责条目的添加、查询、上限裁剪和格式化。
MentalLogEntry 记录活动流中的每一个事件节点。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .models import KFCEventType


@dataclass
class MentalLogEntry:
    """心理活动日志条目，记录活动流中的单个事件。"""

    event_type: KFCEventType
    timestamp: float

    # 通用字段
    content: str = ""

    # 用户消息相关
    user_name: str = ""
    user_id: str = ""

    # Bot 规划相关
    thought: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    expected_reaction: str = ""
    max_wait_seconds: float = 0.0

    # 等待相关
    elapsed_seconds: float = 0.0
    waiting_thought: str = ""
    mood: str = ""

    # 元数据
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_time_str(self, fmt: str = "%H:%M") -> str:
        """获取格式化的时间字符串。"""
        return time.strftime(fmt, time.localtime(self.timestamp))

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "event_type": str(self.event_type),
            "timestamp": self.timestamp,
            "content": self.content,
            "user_name": self.user_name,
            "user_id": self.user_id,
            "thought": self.thought,
            "actions": self.actions,
            "expected_reaction": self.expected_reaction,
            "max_wait_seconds": self.max_wait_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "waiting_thought": self.waiting_thought,
            "mood": self.mood,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MentalLogEntry:
        """从字典反序列化。"""
        event_type_str = data.get("event_type", "user_message")
        try:
            event_type = KFCEventType(event_type_str)
        except ValueError:
            event_type = KFCEventType.USER_MESSAGE

        return cls(
            event_type=event_type,
            timestamp=data.get("timestamp", time.time()),
            content=data.get("content", ""),
            user_name=data.get("user_name", ""),
            user_id=data.get("user_id", ""),
            thought=data.get("thought", ""),
            actions=data.get("actions", []),
            expected_reaction=data.get("expected_reaction", ""),
            max_wait_seconds=float(data.get("max_wait_seconds", 0)),
            elapsed_seconds=float(data.get("elapsed_seconds", 0)),
            waiting_thought=data.get("waiting_thought", ""),
            mood=data.get("mood", ""),
            metadata=data.get("metadata", {}),
        )


class MentalLog:
    """心理活动流容器。

    管理 MentalLogEntry 的添加、查询、裁剪和格式化。
    """

    def __init__(self, max_entries: int = 50) -> None:
        self._entries: list[MentalLogEntry] = []
        self._max_entries = max_entries

    @property
    def entries(self) -> list[MentalLogEntry]:
        """获取所有条目（只读视图）。"""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, entry: MentalLogEntry) -> None:
        """添加条目，超出上限时自动裁剪最旧的。"""
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]

    def get_recent(self, n: int = 20) -> list[MentalLogEntry]:
        """获取最近 n 条条目。"""
        return self._entries[-n:] if self._entries else []

    def get_last_by_type(self, event_type: KFCEventType) -> MentalLogEntry | None:
        """获取指定类型的最后一条条目。"""
        for entry in reversed(self._entries):
            if entry.event_type == event_type:
                return entry
        return None

    def get_last_bot_reply_content(self) -> str:
        """获取最近一次 Bot 回复的文本内容。"""
        for entry in reversed(self._entries):
            if entry.event_type == KFCEventType.BOT_PLANNING:
                for action in entry.actions:
                    if action.get("type") in ("kfc_reply", "respond"):
                        content = action.get("content", "")
                        if content:
                            return content
        return ""

    def format_narrative(self) -> str:
        """以线性叙事格式输出活动流。"""
        if not self._entries:
            return "（暂无活动记录）"

        lines: list[str] = []
        for entry in self._entries:
            time_str = entry.get_time_str()
            line = self._format_entry_narrative(entry, time_str)
            if line:
                lines.append(line)
        return "\n".join(lines)

    def format_as_summary(self, max_entries: int = 10) -> str:
        """格式化为简短摘要，用于 system prompt 注入。"""
        recent = self.get_recent(max_entries)
        if not recent:
            return ""

        lines: list[str] = []
        for entry in recent:
            time_str = entry.get_time_str()
            summary = self._get_entry_summary(entry)
            lines.append(f"[{time_str}] {summary}")
        return "\n".join(lines)

    def to_list(self) -> list[dict[str, Any]]:
        """序列化为字典列表。"""
        return [e.to_dict() for e in self._entries]

    @classmethod
    def from_list(cls, data: list[dict[str, Any]], max_entries: int = 50) -> MentalLog:
        """从字典列表反序列化。"""
        log = cls(max_entries=max_entries)
        for item in data:
            log._entries.append(MentalLogEntry.from_dict(item))
        # 裁剪到上限
        if len(log._entries) > max_entries:
            log._entries = log._entries[-max_entries:]
        return log

    def clear(self) -> None:
        """清空所有条目。"""
        self._entries.clear()

    @staticmethod
    def _format_entry_narrative(entry: MentalLogEntry, time_str: str) -> str:
        """将单个条目格式化为叙事行。"""
        event = entry.event_type

        if event == KFCEventType.USER_MESSAGE:
            name = entry.user_name or "用户"
            return f"[{time_str}] {name} 说：{entry.content}"

        if event == KFCEventType.BOT_PLANNING:
            parts = [f"[{time_str}] 你的内心想法：{entry.thought}"]
            if entry.actions:
                action_desc = ", ".join(
                    a.get("type", "unknown") for a in entry.actions
                )
                parts.append(f"  执行动作：{action_desc}")
            if entry.expected_reaction:
                parts.append(f"  期望对方回应：{entry.expected_reaction}")
            return "\n".join(parts)

        if event == KFCEventType.WAITING_UPDATE:
            return f"[{time_str}] (等待中的内心活动) {entry.waiting_thought}"

        if event == KFCEventType.WAIT_TIMEOUT:
            return f"[{time_str}] 等待超时，已等待 {entry.elapsed_seconds:.0f} 秒"

        if event == KFCEventType.REPLY_IN_TIME:
            return f"[{time_str}] 在预期时间内收到了对方回复"

        if event == KFCEventType.REPLY_LATE:
            return f"[{time_str}] 对方回复较晚（已等待 {entry.elapsed_seconds:.0f} 秒）"

        if event == KFCEventType.PROACTIVE_TRIGGER:
            return f"[{time_str}] (主动发起) {entry.content}"

        if event == KFCEventType.WAITING_START:
            return f"[{time_str}] 开始等待对方回复（最多 {entry.max_wait_seconds:.0f} 秒）"

        return f"[{time_str}] {entry.content}"

    @staticmethod
    def _get_entry_summary(entry: MentalLogEntry) -> str:
        """获取条目的简短摘要。"""
        event = entry.event_type

        if event == KFCEventType.USER_MESSAGE:
            name = entry.user_name or "用户"
            text = entry.content[:60]
            return f"{name}: {text}"

        if event == KFCEventType.BOT_PLANNING:
            return entry.thought[:60] if entry.thought else "(无想法)"

        if event == KFCEventType.WAITING_UPDATE:
            return entry.waiting_thought[:60] if entry.waiting_thought else "(思考中)"

        if event == KFCEventType.WAIT_TIMEOUT:
            return f"等待超时 ({entry.elapsed_seconds:.0f}s)"

        if event == KFCEventType.REPLY_IN_TIME:
            return "及时收到回复"

        if event == KFCEventType.REPLY_LATE:
            return f"延迟回复 ({entry.elapsed_seconds:.0f}s)"

        if event == KFCEventType.PROACTIVE_TRIGGER:
            return entry.content[:60] if entry.content else "主动发起"

        return entry.content[:60] if entry.content else str(event)
