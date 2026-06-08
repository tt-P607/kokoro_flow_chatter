"""KFC 备忘录动作。

包含两个工具：
- ``kfc_memo``：写入或刷新一条带过期时间的私人备忘录。
- ``kfc_memo_delete``：按 id 主动删除已不再需要的备忘录。

备忘录定位：LLM 显式标记的、带过期时间的中短期关键事项，
覆盖"接下来一段时间需要明确意识到的事"这一语义层。
不是长期记忆（那是 history_summary 的职责）；
不是自动事件流（那是 mental_log 的职责）。

数据落地：直接写入 ``KFCSession.memos`` 字段，跟随 session 一起
持久化到 ``data/kokoro_flow_chatter/sessions/<stream_id>.json``。
渲染：作为 turn 级 ContextContribution 注入到用户提示词末尾，
不进入持久化对话链，不破坏 LLM provider 的前缀缓存。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Annotated

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.base import BaseAction

from ..models import (
    MEMO_DEFAULT_EXPIRE_HOURS,
    MEMO_MAX_ENTRIES,
    MEMO_MAX_EXPIRE_HOURS,
    MEMO_MIN_EXPIRE_HOURS,
    KFCEventType,
    Memo,
    clamp_expire_hours,
)

if TYPE_CHECKING:
    from ..session import KFCSession, KFCSessionStore

logger = get_logger("kfc_memo_action")


# ── 私有辅助：访问 session_store ─────────────────────────


def _resolve_session_store(plugin_instance: object) -> "KFCSessionStore | None":
    """从 plugin 实例上取出 KFCSessionStore。

    访问 ``plugin._session_store`` 属性；不可用时返回 None。
    """
    from ..plugin import KFCPlugin

    if isinstance(plugin_instance, KFCPlugin):
        return plugin_instance._session_store  # type: ignore[attr-defined]
    return None


async def _load_session(
    plugin_instance: object,
    stream_id: str,
) -> "KFCSession | None":
    """读取或创建指定流的 KFCSession（未持有锁）。"""
    store = _resolve_session_store(plugin_instance)
    if store is None or not stream_id:
        return None
    return await store.get_or_create(stream_id)


# ── 写入工具 ─────────────────────────────────────────────


_KFC_MEMO_DESCRIPTION = (
    "记录一条带过期时间的私人备忘录。"
    "备忘条目会自动渲染到提示词末尾，让你保持对它的意识，"
    "但不需要时刻提起或反复念叨——只在恰当的时机自然地用上。\n\n"
    f"`expire_hours` 范围：{MEMO_MIN_EXPIRE_HOURS:g}~{MEMO_MAX_EXPIRE_HOURS:g}（"
    f"{MEMO_MIN_EXPIRE_HOURS:g} 小时至 {int(MEMO_MAX_EXPIRE_HOURS / 24)} 天）；"
    f"超出会自动夹到边界。默认 {MEMO_DEFAULT_EXPIRE_HOURS:g} 小时。\n"
    f"同时最多保留 {MEMO_MAX_ENTRIES} 条；超过会自动淘汰创建最早的。\n"
    "已有 content 完全相同的备忘会自动刷新过期时间，不会重复创建。\n\n"
    "**用途定位（语义比较宽，不必拘泥）：**\n"
    "- 对方提到的待办、约定、需要兑现的承诺\n"
    "- 当前需要避开 / 留意的话题或情绪状态\n"
    "- 对当前关系状态的小观察\n"
    "- 未来想问对方的问题、想分享的事\n"
    "只要你觉得「过几个小时或几天后回看时还想知道这件事」，就可以记。\n\n"
    "**重要：** 当某条备忘对应的事情已经做了 / 兑现了 / 不再相关时，"
    "请主动使用 `kfc_memo_delete` 删除它，避免备忘录和实际状态对不上。"
    "过期时间只是兜底，不要依赖它。"
)


class KFCMemoAction(BaseAction):
    """写入或刷新一条私人备忘录。"""

    action_name: str = "kfc_memo"
    associated_types: list[str] = ["text"]
    action_description: str = _KFC_MEMO_DESCRIPTION
    chatter_allow: list[str] = ["kokoro_flow_chatter"]

    async def execute(
        self,
        content: Annotated[
            str,
            "备忘的核心内容。建议简洁明了，几十字以内描述清楚要记的事。",
        ],
        intent: Annotated[
            str,
            "为什么记这条 / 未来看到时希望提醒自己什么。"
            "建议填写，让未来的你看到时更容易理解它的意义。",
        ] = "",
        expire_hours: Annotated[
            float,
            f"备忘的存活时长（小时）。范围 "
            f"{MEMO_MIN_EXPIRE_HOURS:g}~{MEMO_MAX_EXPIRE_HOURS:g}，"
            f"默认 {MEMO_DEFAULT_EXPIRE_HOURS:g}。超出会被夹到边界。",
        ] = MEMO_DEFAULT_EXPIRE_HOURS,
        reason: Annotated[
            str,
            "此刻你想记下这条备忘的真实想法（可留空）。仅用于审计，不参与渲染。",
        ] = "",
    ) -> tuple[bool, str]:
        """写入或刷新一条备忘录。"""
        _ = reason
        normalized_content = (content or "").strip()
        if not normalized_content:
            return False, "content 不能为空"

        normalized_intent = (intent or "").strip()
        clamped_hours = clamp_expire_hours(expire_hours)
        if abs(clamped_hours - float(expire_hours or 0.0)) > 1e-6:
            logger.debug(
                f"[KFC-Memo] expire_hours 被夹到 {clamped_hours}（"
                f"原值 {expire_hours}）"
            )

        now = time.time()
        new_memo = Memo(
            content=normalized_content,
            intent=normalized_intent,
            created_at=now,
            expires_at=now + clamped_hours * 3600.0,
        )

        stream_id = self.chat_stream.stream_id if self.chat_stream else ""
        store = _resolve_session_store(self.plugin)
        if store is None:
            logger.warning("[KFC-Memo] 无法访问 session_store，写入失败")
            return False, "插件未就绪，无法写入备忘"

        async with store.lock(stream_id):
            session = await store.get_or_create(stream_id)
            saved_memo, is_new = session.upsert_memo(new_memo)
            session.add_memo_event(KFCEventType.MEMO_WRITTEN, saved_memo)
            await store.save(session)

        action_word = "已记下" if is_new else "已刷新"
        logger.info(
            f"[KFC-Memo] {action_word} 备忘 id={saved_memo.memo_id}，"
            f"过期={clamped_hours:g}h，content={saved_memo.content[:40]}"
        )
        return True, (
            f"{action_word}备忘（id={saved_memo.memo_id}，"
            f"约 {clamped_hours:g} 小时后过期）"
        )


# ── 删除工具 ─────────────────────────────────────────────


_KFC_MEMO_DELETE_DESCRIPTION = (
    "删除一条或多条已不再需要的备忘录。\n"
    "**典型场景：你看到备忘录里某条事情你刚刚已经做了/兑现了/不再相关了，"
    "就主动调用此工具删掉它，避免脑门便签和实际状态对不上。**\n"
    "`memo_ids`：从备忘录显示中读取的 id 列表（每条备忘渲染时都会显示其 id）。"
)


class KFCMemoDeleteAction(BaseAction):
    """按 id 删除一条或多条备忘录。"""

    action_name: str = "kfc_memo_delete"
    associated_types: list[str] = ["text"]
    action_description: str = _KFC_MEMO_DELETE_DESCRIPTION
    chatter_allow: list[str] = ["kokoro_flow_chatter"]

    async def execute(
        self,
        memo_ids: Annotated[
            list[str],
            "要删除的备忘 id 列表。从备忘录渲染中读取（每条都会显示其 id）。",
        ],
        reason: Annotated[
            str,
            "此刻删除这些备忘的真实想法（可留空）。仅用于审计，不参与渲染。",
        ] = "",
    ) -> tuple[bool, str]:
        """删除指定 id 的备忘。"""
        _ = reason
        if not memo_ids:
            return False, "memo_ids 不能为空"

        # 兼容 LLM 偶尔传单字符串
        if isinstance(memo_ids, str):
            target_ids = [memo_ids.strip()]
        else:
            target_ids = [str(item).strip() for item in memo_ids if str(item).strip()]

        if not target_ids:
            return False, "memo_ids 解析后为空"

        stream_id = self.chat_stream.stream_id if self.chat_stream else ""
        store = _resolve_session_store(self.plugin)
        if store is None:
            logger.warning("[KFC-Memo] 无法访问 session_store，删除失败")
            return False, "插件未就绪，无法删除备忘"

        async with store.lock(stream_id):
            session = await store.get_or_create(stream_id)
            deleted = session.delete_memos(target_ids)
            for memo in deleted:
                session.add_memo_event(KFCEventType.MEMO_DELETED, memo)
            await store.save(session)

        if not deleted:
            logger.debug(
                f"[KFC-Memo] 未找到匹配的备忘 ids={target_ids}"
            )
            return True, f"未找到匹配的备忘（请求 ids={target_ids}）"

        logger.info(
            f"[KFC-Memo] 删除了 {len(deleted)} 条备忘 ids="
            f"{[m.memo_id for m in deleted]}"
        )
        return True, f"已删除 {len(deleted)} 条备忘"
