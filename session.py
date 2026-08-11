"""KFC 会话状态与持久化存储。

``KFCSession`` 保存单个聊天流的全部跨轮状态：等待配置、心理活动流、
持久化对话链、近期记忆摘要、备忘录与主动发起预约。
``KFCSessionStore`` 负责按 ``stream_id`` 索引这些会话，并通过
``JSONStore`` 落盘到 ``data/kokoro_flow_chatter/sessions/``。

所有跨协程的读写都应通过 ``async with store.lock(stream_id)`` 串行化，
避免 Scheduler 回调与 ``execute()`` 主循环竞态。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from src.app.plugin_system.api.log_api import get_logger

from .mental_log import MentalLog, MentalLogEntry
from .models import MEMO_MAX_ENTRIES, KFCEventType, Memo, WaitingConfig

logger = get_logger("kfc_session")

_STORAGE_DIR = "data/kokoro_flow_chatter/sessions"
"""会话文件所在目录，按 ``stream_id`` 分文件存放。"""

_INDEX_FILENAME = "_index.json"
"""可读索引文件名，维护 ``stream_id`` → 平台/用户 的映射便于人工排查。"""

_LOCK_CLEANUP_THRESHOLD = 100
"""锁字典超过此规模时触发一次不活跃锁清理。"""


@dataclass
class KFCSession:
    """单个聊天流的 KFC 会话状态。"""

    user_id: str
    stream_id: str
    platform: str = ""

    waiting_config: WaitingConfig = field(default_factory=WaitingConfig)
    consecutive_timeout_count: int = 0
    """等待状态与连续超时计数。"""

    created_at: float = field(default_factory=time.time)
    last_activity_at: float = field(default_factory=time.time)
    last_user_message_at: float | None = None
    last_proactive_at: float | None = None
    """各类时间戳，供主动发起的沉默判定使用。"""

    scheduled_proactive_at: float | None = None
    scheduled_proactive_reason: str = ""
    """模型预约的下次主动思考时间与理由。存在预约时，条件主动发起逻辑
    暂停，直到预约到期或被显式清除。"""

    mental_log: MentalLog = field(default_factory=MentalLog)
    """心理活动流。"""

    chain_payloads: list[dict[str, Any]] = field(default_factory=list)
    chain_cutoff_ts: float = 0.0
    """持久化对话链。每条形如 ``{"role", "text", "ts"?, "tool_calls"?}``；
    ``chain_cutoff_ts`` 记录链头首个 user 条目的时间戳，供融合叙事截断，
    避免叙事与链内容重叠。"""

    context_snapshot: list[dict[str, Any]] | None = None
    """上下文快照（来自主链 ``response.payloads``）。重启后优先恢复为
    多角色 LLM 原始返回，缺失或校验失败时回退 ``chain_payloads``。"""

    history_summary: str = ""
    last_compress_at: float = 0.0
    compress_round_count: int = 0
    """近期记忆摘要（替换式滚动压缩）及其触发计数。"""

    memos: list[Memo] = field(default_factory=list)
    """私人备忘录。渲染为 turn 级上下文注入用户提示词末尾，不进对话链。"""

    total_interactions: int = 0
    """累计 Bot 决策次数。"""

    # ── 等待状态 ──────────────────────────────────────────

    def set_waiting(self, config: WaitingConfig) -> None:
        """设置等待状态；``max_wait_seconds <= 0`` 时等价于清除。"""
        if config.max_wait_seconds <= 0:
            self.clear_waiting()
            return
        self.waiting_config = config

    def clear_waiting(self) -> None:
        """清除等待状态并刷新活跃时间。"""
        self.waiting_config.reset()
        self.last_activity_at = time.time()

    def is_waiting(self) -> bool:
        """是否处于等待对方回复的状态。"""
        return self.waiting_config.is_active()

    # ── 心理活动流写入 ────────────────────────────────────

    def add_user_message(
        self,
        content: str,
        user_name: str,
        user_id: str,
        timestamp: float | None = None,
        message_id: str = "",
    ) -> MentalLogEntry:
        """记录一条用户消息到活动流，并顺带标注回复时效。"""
        msg_time = timestamp or time.time()
        entry = MentalLogEntry(
            event_type=KFCEventType.USER_MESSAGE,
            timestamp=msg_time,
            content=content,
            user_name=user_name,
            user_id=user_id,
            message_id=message_id,
        )

        if self.waiting_config.is_active():
            elapsed = self.waiting_config.get_elapsed_seconds()
            max_wait = self.waiting_config.max_wait_seconds
            entry.metadata["reply_status"] = (
                "in_time" if elapsed <= max_wait else "late"
            )
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
        """记录一次 Bot 决策到活动流。"""
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

    def add_interrupt_event(self, interrupt_msgs: list[Any]) -> MentalLogEntry:
        """记录一次"生成期间被新消息打断"事件。

        让模型在下一轮上下文中感知到"我刚才的回复是在没看到这些消息的
        情况下做出的"，从而做出更自然的衔接。

        Args:
            interrupt_msgs: 打断时到达的消息列表。

        Returns:
            MentalLogEntry: 已写入活动流的条目。
        """
        senders = {
            (msg.sender_name or msg.sender_id or "未知") for msg in interrupt_msgs
        }
        entry = MentalLogEntry(
            event_type=KFCEventType.USER_INTERRUPTED,
            timestamp=time.time(),
            content=(
                f"我正在思考时，{'、'.join(sorted(senders))} 发来了 "
                f"{len(interrupt_msgs)} 条新消息，"
                "我的回复是在没看到这些消息的情况下做出的。"
            ),
        )
        self.mental_log.add(entry)
        return entry

    # ── 主动发起预约 ──────────────────────────────────────

    def set_scheduled_proactive(self, at: float | None, reason: str = "") -> None:
        """设置或清除模型预约的主动思考时间。

        Args:
            at: Unix 时间戳；``None`` 表示清除预约。
            reason: 预约理由，触发时注入提示词；清除时一并置空。
        """
        self.scheduled_proactive_at = at
        self.scheduled_proactive_reason = reason if at is not None else ""

    # ── 持久化对话链 ──────────────────────────────────────

    def update_chain(
        self,
        new_entries: list[dict[str, Any]],
        max_payloads: int,
    ) -> None:
        """追加对话条目到持久化链并裁剪至上限。

        user 条目按 ``(text, ts)`` 幂等去重：LLM 失败后下一 Tick 会重新
        消费同一批未读，不去重会导致同一条消息被反复 append。assistant
        条目不去重——每次 commit 对应一次真实的模型输出，即便文本相同
        也应保留时序结构。

        Args:
            new_entries: 待追加的条目，形如
                ``{"role": "user"|"assistant", "text": str, "ts"?: float}``。
            max_payloads: 链最大条目数，超出时从头裁剪。
        """
        filtered: list[dict[str, Any]] = []
        for entry in new_entries:
            if entry.get("role") == "user":
                duplicated = any(
                    existing.get("role") == "user"
                    and existing.get("text", "") == entry.get("text", "")
                    and existing.get("ts", 0.0) == entry.get("ts", 0.0)
                    for existing in self.chain_payloads
                )
                if duplicated:
                    continue
            filtered.append(entry)

        if not filtered:
            return

        self.chain_payloads.extend(filtered)
        if len(self.chain_payloads) > max_payloads:
            self.chain_payloads = self.chain_payloads[-max_payloads:]
            # 裁剪后链头必须是 user，否则孤立的 assistant 会让上下文非法
            while self.chain_payloads and self.chain_payloads[0].get("role") != "user":
                self.chain_payloads.pop(0)

        self.chain_cutoff_ts = 0.0
        for entry in self.chain_payloads:
            if entry.get("role") == "user":
                ts = entry.get("ts", 0.0)
                if isinstance(ts, (int, float)) and ts > 0:
                    self.chain_cutoff_ts = float(ts)
                break

    # ── 备忘录管理 ────────────────────────────────────────

    def prune_expired_memos(self) -> list[Memo]:
        """删除已过期备忘并返回被删除的列表（懒清理入口）。

        返回值供调用方决定是否补写 ``MEMO_EXPIRED`` 事件。
        """
        if not self.memos:
            return []
        now = time.time()
        expired = [memo for memo in self.memos if memo.is_expired(now)]
        if expired:
            self.memos = [memo for memo in self.memos if not memo.is_expired(now)]
        return expired

    def upsert_memo(self, memo: Memo) -> tuple[Memo, bool]:
        """写入或刷新一条备忘。

        若已存在 ``content`` 完全相同的有效备忘，只刷新其过期时间和
        intent（保留原 ``created_at``）；否则在必要时淘汰最早的一条后追加。

        Args:
            memo: 待写入的备忘。

        Returns:
            tuple: ``(最终落盘的备忘, 是否为新建)``。
        """
        self.prune_expired_memos()

        normalized_content = memo.content.strip()
        if normalized_content:
            for existing in self.memos:
                if existing.content.strip() == normalized_content:
                    existing.expires_at = memo.expires_at
                    if memo.intent.strip():
                        existing.intent = memo.intent
                    return existing, False

        while len(self.memos) >= MEMO_MAX_ENTRIES:
            oldest_index = min(
                range(len(self.memos)),
                key=lambda index: self.memos[index].created_at,
            )
            self.memos.pop(oldest_index)

        self.memos.append(memo)
        return memo, True

    def delete_memos(self, memo_ids: list[str]) -> list[Memo]:
        """按 id 删除备忘。

        Args:
            memo_ids: 待删除的备忘 id 列表。

        Returns:
            list[Memo]: 实际被删除的备忘，供调用方补写活动流事件。
        """
        if not memo_ids or not self.memos:
            return []
        target_ids = set(memo_ids)
        deleted = [memo for memo in self.memos if memo.memo_id in target_ids]
        if deleted:
            self.memos = [
                memo for memo in self.memos if memo.memo_id not in target_ids
            ]
        return deleted

    def add_memo_event(
        self,
        event_type: KFCEventType,
        memo: Memo,
    ) -> MentalLogEntry:
        """把备忘相关事件写入心理活动流。

        Args:
            event_type: ``MEMO_WRITTEN`` / ``MEMO_DELETED`` / ``MEMO_EXPIRED`` 之一。
            memo: 关联的备忘对象。
        """
        entry = MentalLogEntry(
            event_type=event_type,
            timestamp=time.time(),
            content=memo.content,
            metadata={
                "memo_id": memo.memo_id,
                "intent": memo.intent,
                "expires_at": memo.expires_at,
            },
        )
        self.mental_log.add(entry)
        return entry

    # ── 序列化 ────────────────────────────────────────────

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
            "scheduled_proactive_reason": self.scheduled_proactive_reason,
            "mental_log": self.mental_log.to_list(),
            "total_interactions": self.total_interactions,
            "chain_payloads": self.chain_payloads,
            "chain_cutoff_ts": self.chain_cutoff_ts,
            "context_snapshot": self.context_snapshot,
            "history_summary": self.history_summary,
            "last_compress_at": self.last_compress_at,
            "compress_round_count": self.compress_round_count,
            "memos": [memo.to_dict() for memo in self.memos],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        max_log_entries: int = 50,
    ) -> KFCSession:
        """从字典反序列化。

        Args:
            data: 序列化字典。
            max_log_entries: 活动流最大条目数（来自配置）。
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
        session.scheduled_proactive_reason = data.get(
            "scheduled_proactive_reason", ""
        )
        session.mental_log = MentalLog.from_list(
            data.get("mental_log", []),
            max_entries=max_log_entries,
        )
        session.total_interactions = int(data.get("total_interactions", 0))
        session.chain_payloads = data.get("chain_payloads", [])
        session.chain_cutoff_ts = float(data.get("chain_cutoff_ts", 0.0))
        session.context_snapshot = data.get("context_snapshot")
        session.history_summary = data.get("history_summary", "")
        session.last_compress_at = float(data.get("last_compress_at", 0.0))
        session.compress_round_count = int(data.get("compress_round_count", 0))
        session.memos = [
            Memo.from_dict(item)
            for item in data.get("memos", []) or []
            if isinstance(item, dict)
        ]
        return session


class KFCSessionStore:
    """按 ``stream_id`` 索引的 KFC 会话存储。

    内存缓存 + ``JSONStore`` 落盘，并为每个流提供独立的 asyncio 锁。
    """

    def __init__(self, max_log_entries: int = 50) -> None:
        """初始化存储。

        Args:
            max_log_entries: 反序列化会话时使用的活动流上限。
        """
        self._sessions: dict[str, KFCSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._max_log_entries = max_log_entries
        self._json_store: Any = None
        self._store_initialized = False

    # ── 并发控制 ──────────────────────────────────────────

    @asynccontextmanager
    async def lock(self, stream_id: str) -> AsyncIterator[None]:
        """获取指定流的互斥锁上下文。

        确保同一流的会话读写串行化，防止 Scheduler 回调与主循环竞态。

        Args:
            stream_id: 流 ID。

        Yields:
            None
        """
        if stream_id not in self._locks:
            self._locks[stream_id] = asyncio.Lock()
        async with self._locks[stream_id]:
            yield

    def cleanup_inactive_locks(self) -> int:
        """清理不在缓存中且未被持有的锁，返回清理数量。"""
        stale = [
            stream_id
            for stream_id, lock in self._locks.items()
            if stream_id not in self._sessions and not lock.locked()
        ]
        for stream_id in stale:
            del self._locks[stream_id]
        return len(stale)

    # ── 读写 ──────────────────────────────────────────────

    async def get_or_create(self, stream_id: str) -> KFCSession:
        """获取会话，不存在时创建新会话。

        注意：本方法不持有锁，调用方应用 ``async with store.lock(...)``
        包裹完整的读-改-写周期。
        """
        cached = self._sessions.get(stream_id)
        if cached is not None:
            return cached

        loaded = await self._load_from_disk(stream_id)
        if loaded is not None:
            self._sessions[stream_id] = loaded
            return loaded

        session = KFCSession(user_id="", stream_id=stream_id)
        session.mental_log = MentalLog(max_entries=self._max_log_entries)
        self._sessions[stream_id] = session
        return session

    async def get(self, stream_id: str) -> KFCSession | None:
        """获取会话，不存在时返回 ``None``（会写入内存缓存）。"""
        cached = self._sessions.get(stream_id)
        if cached is not None:
            return cached

        loaded = await self._load_from_disk(stream_id)
        if loaded is not None:
            self._sessions[stream_id] = loaded
        return loaded

    async def peek(self, stream_id: str) -> KFCSession | None:
        """读取会话但不写入内存缓存。

        适用于只需查看持久化字段、不希望污染缓存的批量扫描场景
        （如主动发起对磁盘会话的预约检查）。
        """
        cached = self._sessions.get(stream_id)
        if cached is not None:
            return cached
        return await self._load_from_disk(stream_id)

    async def save(self, session: KFCSession) -> None:
        """保存会话到内存缓存与磁盘。

        注意：本方法不持有锁，调用方应用 ``async with store.lock(...)``
        包裹完整的读-改-写周期。
        """
        self._sessions[session.stream_id] = session
        await self._ensure_store()

        if self._json_store is not None:
            try:
                await self._json_store.save(session.stream_id, session.to_dict())
                await self._update_index(session)
            except Exception as error:
                logger.warning(
                    f"会话持久化失败 (stream={session.stream_id[:8]}): {error}"
                )

        if len(self._locks) > _LOCK_CLEANUP_THRESHOLD:
            cleaned = self.cleanup_inactive_locks()
            if cleaned:
                logger.debug(f"清理了 {cleaned} 个不活跃的会话锁")

    def get_all_cached(self) -> dict[str, KFCSession]:
        """返回所有内存缓存中的会话副本（不触发 IO）。"""
        return dict(self._sessions)

    async def list_all_stream_ids(self) -> list[str]:
        """列出所有已持久化的 ``stream_id``（跳过 ``_`` 开头的辅助文件）。"""
        await self._ensure_store()
        if self._json_store is None:
            return []
        try:
            all_ids = await self._json_store.list_all()
        except Exception as error:
            logger.warning(f"会话列举失败: {error}")
            return []
        return [sid for sid in all_ids if not sid.startswith("_")]

    # ── 内部实现 ──────────────────────────────────────────

    async def _ensure_store(self) -> None:
        """延迟初始化 ``JSONStore``。"""
        if self._store_initialized:
            return
        self._store_initialized = True
        try:
            from src.app.plugin_system.api.storage_api import JSONStore

            self._json_store = JSONStore(storage_dir=_STORAGE_DIR)
        except ImportError:
            self._json_store = None

    async def _load_from_disk(self, stream_id: str) -> KFCSession | None:
        """从磁盘读取并反序列化会话；不存在或损坏时返回 ``None``。"""
        await self._ensure_store()
        if self._json_store is None:
            return None
        try:
            data = await self._json_store.load(stream_id)
        except Exception as error:
            logger.warning(f"会话加载失败 (stream={stream_id[:8]}): {error}")
            return None
        if not data or not isinstance(data, dict):
            return None
        return KFCSession.from_dict(data, max_log_entries=self._max_log_entries)

    async def _update_index(self, session: KFCSession) -> None:
        """刷新可读索引文件，便于人工对照文件名与账号。"""
        if self._json_store is None:
            return

        index_path = self._json_store.get_storage_dir() / _INDEX_FILENAME
        try:
            raw = await asyncio.to_thread(index_path.read_bytes)
            index: dict[str, dict[str, str]] = json.loads(raw)
        except (FileNotFoundError, json.JSONDecodeError):
            index = {}

        index[session.stream_id] = {
            "platform": session.platform,
            "user_id": session.user_id,
        }

        try:
            payload = json.dumps(index, ensure_ascii=False, indent=2).encode("utf-8")
            await asyncio.to_thread(index_path.write_bytes, payload)
        except Exception as error:
            logger.debug(f"会话索引写入失败: {error}")
