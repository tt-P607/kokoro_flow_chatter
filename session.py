"""KFC 会话状态管理。

维护每个用户/流的 KFCSession，通过 KFCSessionStore 持久化。
KFCSession 包含等待配置、连续超时计数、心理活动流等状态。
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .mental_log import MentalLog, MentalLogEntry
from .models import KFCEventType, WaitingConfig


@dataclass
class KFCSession:
    """KFC 会话状态数据。"""

    user_id: str
    stream_id: str

    # 等待状态
    waiting_config: WaitingConfig = field(default_factory=WaitingConfig)
    consecutive_timeout_count: int = 0

    # 时间戳
    created_at: float = field(default_factory=time.time)
    last_activity_at: float = field(default_factory=time.time)
    last_user_message_at: float | None = None
    last_proactive_at: float | None = None

    # 心理活动流
    mental_log: MentalLog = field(default_factory=MentalLog)

    # 连续思考期间的待注入想法（Scheduler 写入，execute() 读取后清空）
    pending_thoughts: list[str] = field(default_factory=list)

    # 统计
    total_interactions: int = 0

    def set_waiting(self, config: WaitingConfig) -> None:
        """设置等待状态。"""
        if config.max_wait_seconds <= 0:
            self.clear_waiting()
            return
        self.waiting_config = config

    def clear_waiting(self) -> None:
        """清除等待状态。"""
        self.waiting_config.reset()
        self.last_activity_at = time.time()

    def is_waiting(self) -> bool:
        """是否处于等待状态。"""
        return self.waiting_config.is_active()

    def add_user_message(
        self,
        content: str,
        user_name: str,
        user_id: str,
        timestamp: float | None = None,
    ) -> MentalLogEntry:
        """记录用户消息到活动流。"""
        msg_time = timestamp or time.time()
        entry = MentalLogEntry(
            event_type=KFCEventType.USER_MESSAGE,
            timestamp=msg_time,
            content=content,
            user_name=user_name,
            user_id=user_id,
        )

        # 标记回复时效
        if self.waiting_config.is_active():
            elapsed = self.waiting_config.get_elapsed_seconds()
            max_wait = self.waiting_config.max_wait_seconds
            if elapsed <= max_wait:
                entry.metadata["reply_status"] = "in_time"
            else:
                entry.metadata["reply_status"] = "late"
            entry.metadata["elapsed_seconds"] = elapsed
            entry.metadata["max_wait_seconds"] = max_wait

        self.mental_log.add(entry)
        self.consecutive_timeout_count = 0
        self.last_user_message_at = msg_time
        self.last_activity_at = msg_time
        return entry

    def add_bot_planning(
        self,
        thought: str,
        actions: list[dict[str, Any]],
        expected_reaction: str = "",
        max_wait_seconds: float = 0.0,
    ) -> MentalLogEntry:
        """记录 Bot 规划到活动流。"""
        entry = MentalLogEntry(
            event_type=KFCEventType.BOT_PLANNING,
            timestamp=time.time(),
            thought=thought,
            actions=actions,
            expected_reaction=expected_reaction,
            max_wait_seconds=max_wait_seconds,
        )
        self.mental_log.add(entry)
        self.total_interactions += 1
        self.last_activity_at = time.time()
        return entry

    def add_waiting_update(
        self, waiting_thought: str, mood: str = ""
    ) -> MentalLogEntry:
        """记录等待期间的心理活动。"""
        entry = MentalLogEntry(
            event_type=KFCEventType.WAITING_UPDATE,
            timestamp=time.time(),
            waiting_thought=waiting_thought,
            mood=mood,
            elapsed_seconds=self.waiting_config.get_elapsed_seconds(),
        )
        self.mental_log.add(entry)
        return entry

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "user_id": self.user_id,
            "stream_id": self.stream_id,
            "waiting_config": self.waiting_config.to_dict(),
            "consecutive_timeout_count": self.consecutive_timeout_count,
            "created_at": self.created_at,
            "last_activity_at": self.last_activity_at,
            "last_user_message_at": self.last_user_message_at,
            "last_proactive_at": self.last_proactive_at,
            "mental_log": self.mental_log.to_list(),
            "pending_thoughts": self.pending_thoughts,
            "total_interactions": self.total_interactions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KFCSession:
        """从字典反序列化。"""
        session = cls(
            user_id=data.get("user_id", ""),
            stream_id=data.get("stream_id", ""),
        )
        session.waiting_config = WaitingConfig.from_dict(
            data.get("waiting_config", {})
        )
        session.consecutive_timeout_count = int(
            data.get("consecutive_timeout_count", 0)
        )
        session.created_at = float(data.get("created_at", time.time()))
        session.last_activity_at = float(data.get("last_activity_at", time.time()))
        session.last_user_message_at = data.get("last_user_message_at")
        session.last_proactive_at = data.get("last_proactive_at")
        session.mental_log = MentalLog.from_list(
            data.get("mental_log", []),
            max_entries=50,
        )
        session.pending_thoughts = data.get("pending_thoughts", [])
        session.total_interactions = int(data.get("total_interactions", 0))
        return session


class KFCSessionStore:
    """KFC 会话持久化存储。

    使用 JSONStore 进行简单 JSON 文件持久化。
    Session 按 stream_id 索引。
    """

    def __init__(self) -> None:
        self._sessions: dict[str, KFCSession] = {}
        self._store_initialized = False
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, stream_id: str) -> asyncio.Lock:
        """获取指定 stream_id 的锁（惰性创建）。"""
        if stream_id not in self._locks:
            self._locks[stream_id] = asyncio.Lock()
        return self._locks[stream_id]

    @asynccontextmanager
    async def lock(self, stream_id: str) -> AsyncIterator[None]:
        """获取指定 stream_id 的互斥锁上下文。

        确保同一 stream 的 Session 读写串行化，
        防止 Scheduler 回调与 execute() 并发读写同一 Session。

        Args:
            stream_id: 流 ID

        Yields:
            None
        """
        async with self._get_lock(stream_id):
            yield

    async def _ensure_store(self) -> None:
        """延迟初始化 JSONStore。"""
        if self._store_initialized:
            return
        try:
            from src.kernel.storage import JSONStore

            self._json_store = JSONStore(
                storage_dir="data/kokoro_flow_chatter/sessions"
            )
            self._store_initialized = True
        except ImportError:
            self._json_store = None
            self._store_initialized = True

    async def get_or_create(self, stream_id: str) -> KFCSession:
        """获取或创建 Session。

        注意：此方法不持有 per-stream 锁。调用方应使用 ``async with store.lock(stream_id)``
        包裹完整的读写周期以避免并发竞态。
        """
        if stream_id in self._sessions:
            return self._sessions[stream_id]

        await self._ensure_store()

        # 尝试从持久化加载
        if self._json_store is not None:
            try:
                data = await self._json_store.load(stream_id)
                if data and isinstance(data, dict):
                    session = KFCSession.from_dict(data)
                    self._sessions[stream_id] = session
                    return session
            except Exception:
                pass

        # 创建新 Session
        session = KFCSession(user_id="", stream_id=stream_id)
        self._sessions[stream_id] = session
        return session

    async def save(self, session: KFCSession) -> None:
        """保存 Session 到持久化存储。

        注意：此方法不持有 per-stream 锁。调用方应使用 ``async with store.lock(stream_id)``
        包裹完整的读写周期以避免并发竞态。
        """
        self._sessions[session.stream_id] = session
        await self._ensure_store()

        if self._json_store is not None:
            try:
                await self._json_store.save(session.stream_id, session.to_dict())
            except Exception:
                pass

    async def get(self, stream_id: str) -> KFCSession | None:
        """获取 Session（不创建）。"""
        if stream_id in self._sessions:
            return self._sessions[stream_id]

        await self._ensure_store()
        if self._json_store is not None:
            try:
                data = await self._json_store.load(stream_id)
                if data and isinstance(data, dict):
                    session = KFCSession.from_dict(data)
                    self._sessions[stream_id] = session
                    return session
            except Exception:
                pass
        return None

    def get_all_cached(self) -> dict[str, KFCSession]:
        """获取所有缓存中的 Session（不触发 IO）。"""
        return dict(self._sessions)
