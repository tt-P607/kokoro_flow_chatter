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

from src.app.plugin_system.api.log_api import get_logger

from .mental_log import MentalLog, MentalLogEntry
from .models import KFCEventType, WaitingConfig

logger = get_logger("kfc_session")


@dataclass
class KFCSession:
    """KFC 会话状态数据。"""

    user_id: str
    stream_id: str
    platform: str = ""

    # 等待状态
    waiting_config: WaitingConfig = field(default_factory=WaitingConfig)
    consecutive_timeout_count: int = 0

    # 时间戳
    created_at: float = field(default_factory=time.time)
    last_activity_at: float = field(default_factory=time.time)
    last_user_message_at: float | None = None
    last_proactive_at: float | None = None

    # 模型预约的下一次主动思考时间（Unix 时间戳）
    # 若存在，条件主动发起逻辑不生效，直到预约时间到来或被清除
    scheduled_proactive_at: float | None = None

    # 心理活动流
    mental_log: MentalLog = field(default_factory=MentalLog)

    # 持久化对话链：序列化的 USER/ASSISTANT payload 列表，跨 execute() 重启保留上下文。
    # 每条条目格式：{"role": "user"|"assistant", "text": "...", "ts": <float，仅 user>}
    # chain_cutoff_ts 为链头第一个 user 条目的时间戳，供 build_fused_narrative 做截断。
    chain_payloads: list[dict[str, Any]] = field(default_factory=list)
    chain_cutoff_ts: float = 0.0

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
        message_id: str = "",
    ) -> MentalLogEntry:
        """记录用户消息到活动流。"""
        msg_time = timestamp or time.time()
        entry = MentalLogEntry(
            event_type=KFCEventType.USER_MESSAGE,
            timestamp=msg_time,
            content=content,
            user_name=user_name,
            user_id=user_id,
            message_id=message_id,
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
        raw_response: str = "",
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
        if raw_response:
            entry.metadata["raw_response"] = raw_response
        self.mental_log.add(entry)
        self.total_interactions += 1
        self.last_activity_at = time.time()
        return entry

    def set_scheduled_proactive(self, at: float | None) -> None:
        """设置（或清除）模型预约的主动思考时间。

        Args:
            at: Unix 时间戳，None 表示清除预约
        """
        self.scheduled_proactive_at = at

    def update_chain(
        self, new_entries: list[dict[str, Any]], max_payloads: int
    ) -> None:
        """追加新的对话条目到持久化链，并裁剪至 max_payloads 条目。

        Args:
            new_entries: 要追加的条目列表，每条格式为
                ``{"role": "user"|"assistant", "text": "...", "ts": float}``。
                USER 条目应携带 ``ts``（第一条未读消息的时间戳），
                ASSISTANT 条目无需携带 ``ts``。
            max_payloads: 链最大条目数，超出时删除最老的条目。
        """
        self.chain_payloads.extend(new_entries)
        if len(self.chain_payloads) > max_payloads:
            self.chain_payloads = self.chain_payloads[-max_payloads:]
        # 更新截止时间戳为链头第一个 user 条目的时间
        self.chain_cutoff_ts = 0.0
        for entry in self.chain_payloads:
            if entry.get("role") == "user":
                ts = entry.get("ts", 0.0)
                if isinstance(ts, (int, float)) and ts > 0:
                    self.chain_cutoff_ts = float(ts)
                break

    def clear_chain(self) -> None:
        """清空持久化对话链（重置上下文）。"""
        self.chain_payloads = []
        self.chain_cutoff_ts = 0.0

    def add_interrupt_event(self, interrupt_msgs: list[Any]) -> MentalLogEntry:
        """记录用户打断事件到活动流。

        当 LLM 生成期间检测到新消息时调用，让模型在下一轮上下文
        中感知到"我正在思考时被打断"这一事实，从而做出更自然的响应。

        Args:
            interrupt_msgs: 打断时到达的消息列表

        Returns:
            MentalLogEntry: 写入活动流的条目
        """
        count = len(interrupt_msgs)
        senders = {
            getattr(m, "sender_name", "") or getattr(m, "sender_id", "未知")
            for m in interrupt_msgs
        }
        sender_str = "、".join(sorted(senders))
        entry = MentalLogEntry(
            event_type=KFCEventType.USER_INTERRUPTED,
            timestamp=time.time(),
            content=(
                f"我正在思考时，{sender_str} 发来了 {count} 条新消息，"
                "我的回复是在没看到这些消息的情况下做出的。"
            ),
        )
        self.mental_log.add(entry)
        return entry

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "user_id": self.user_id,
            "stream_id": self.stream_id,
            "platform": self.platform,
            "waiting_config": self.waiting_config.to_dict(),
            "consecutive_timeout_count": self.consecutive_timeout_count,
            "created_at": self.created_at,
            "last_activity_at": self.last_activity_at,
            "last_user_message_at": self.last_user_message_at,
            "last_proactive_at": self.last_proactive_at,
            "scheduled_proactive_at": self.scheduled_proactive_at,
            "mental_log": self.mental_log.to_list(),
            "total_interactions": self.total_interactions,
            "chain_payloads": self.chain_payloads,
            "chain_cutoff_ts": self.chain_cutoff_ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], max_log_entries: int = 50) -> KFCSession:
        """从字典反序列化。

        Args:
            data: 序列化的字典数据
            max_log_entries: 活动流最大条目数（来自配置）
        """
        session = cls(
            user_id=data.get("user_id", ""),
            stream_id=data.get("stream_id", ""),
            platform=data.get("platform", ""),
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
        session.scheduled_proactive_at = data.get("scheduled_proactive_at")
        session.mental_log = MentalLog.from_list(
            data.get("mental_log", []),
            max_entries=max_log_entries,
        )
        session.total_interactions = int(data.get("total_interactions", 0))
        # 持久化对话链
        session.chain_payloads = data.get("chain_payloads", [])
        session.chain_cutoff_ts = float(data.get("chain_cutoff_ts", 0.0))
        return session


class KFCSessionStore:
    """KFC 会话持久化存储。

    使用 JSONStore 进行简单 JSON 文件持久化。
    Session 按 stream_id 索引。
    """

    def __init__(self, max_log_entries: int = 50) -> None:
        self._sessions: dict[str, KFCSession] = {}
        self._store_initialized = False
        self._locks: dict[str, asyncio.Lock] = {}
        self._max_log_entries = max_log_entries

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
                    session = KFCSession.from_dict(data, max_log_entries=self._max_log_entries)
                    self._sessions[stream_id] = session
                    return session
            except Exception as e:
                logger.warning(f"Session 加载失败 (stream={stream_id[:8]}): {e}")

        # 创建新 Session
        session = KFCSession(user_id="", stream_id=stream_id, platform="")
        session.mental_log = MentalLog(max_entries=self._max_log_entries)
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
            except Exception as e:
                logger.warning(
                    f"Session 持久化失败 (stream={session.stream_id[:8]}): {e}"
                )

        # 锁字典膨胀时定期清理不活跃的锁
        if len(self._locks) > 100:
            cleaned = self.cleanup_inactive_locks()
            if cleaned:
                logger.debug(f"清理了 {cleaned} 个不活跃的锁")

    async def get(self, stream_id: str) -> KFCSession | None:
        """获取 Session（不创建）。"""
        if stream_id in self._sessions:
            return self._sessions[stream_id]

        await self._ensure_store()
        if self._json_store is not None:
            try:
                data = await self._json_store.load(stream_id)
                if data and isinstance(data, dict):
                    session = KFCSession.from_dict(data, max_log_entries=self._max_log_entries)
                    self._sessions[stream_id] = session
                    return session
            except Exception as e:
                logger.warning(f"Session 加载失败 (stream={stream_id[:8]}): {e}")
        return None

    def get_all_cached(self) -> dict[str, KFCSession]:
        """获取所有缓存中的 Session（不触发 IO）。"""
        return dict(self._sessions)

    def cleanup_inactive_locks(self) -> int:
        """清理不活跃 stream 的锁，释放内存。

        移除不在缓存中且当前未被持有的锁。

        Returns:
            int: 被清理的锁数量
        """
        stale = [
            sid for sid, lock in self._locks.items()
            if sid not in self._sessions and not lock.locked()
        ]
        for sid in stale:
            del self._locks[sid]
        return len(stale)

    async def list_all_stream_ids(self) -> list[str]:
        """列出所有已持久化的 stream_id。

        从 JSON 存储中读取所有会话文件名，
        用于在插件启动时预注册 VLM 跳过等批量操作。

        Returns:
            list[str]: 所有已知的 stream_id 列表
        """
        await self._ensure_store()
        if self._json_store is not None:
            try:
                return await self._json_store.list_all()
            except Exception as e:
                logger.warning(f"Session 列举失败: {e}")
                return []
        return []
