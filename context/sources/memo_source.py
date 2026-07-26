"""KFC 备忘录上下文 source。

把 ``KFCSession.memos`` 渲染成一条 turn 级上下文贡献，注入本轮用户
提示词末尾——让模型"看到自己脑门上贴的便签"，同时不污染系统前缀缓存，
也不进入持久化对话链。
"""

from __future__ import annotations

import datetime
import time

from ...models import Memo
from ..types import ContextContribution

_MEMO_SOURCE = "kfc.memo"

_MEMO_PRIORITY = 80
"""高于一般 notice，使备忘录紧贴提示词末尾。"""

_MEMO_GUIDANCE = (
    "## 关于这些备忘\n"
    "这些是你给自己留下的备忘录，记着接下来一段时间需要意识到的事。"
    "**不需要时刻提起或反复念叨**，只在恰当的时机自然地用上：\n"
    "- 对方提到的话题刚好和某条备忘相关时，你心里能想起这事；\n"
    "- 某件被记录的事到了该兑现的时间，你能主动行动；\n"
    "- 某件事已经做了或不再相关时，主动调用 `action-kfc_memo_delete` 清理它，"
    "避免备忘录和实际状态对不上。\n\n"
    "**写入时机：** 你觉得「过几个小时或几天后回看时还想知道这件事」，就可以记。"
    "不必拘泥于「该不该记」，宽一点没关系。\n\n"
    "**删除时机：** 看到某条已经做了 / 兑现了 / 不再相关，主动删除它。"
    "过期时间只是兜底，不要依赖它。"
)


def _format_datetime(timestamp: float) -> str:
    """把时间戳格式化为「年-月-日 时:分」。"""
    try:
        return datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
    except (OSError, ValueError, OverflowError):
        return "未知时间"


def _format_remaining(memo: Memo, now: float) -> str:
    """把剩余时间格式化为人类可读文本。

    按小时取整，避免每次渲染都产生细微差异而干扰调试与缓存。
    """
    remaining_seconds = memo.remaining_seconds(now)
    if remaining_seconds <= 0:
        return "已过期"

    remaining_hours = remaining_seconds / 3600.0
    if remaining_hours < 1:
        return "剩余不到 1 小时"
    if remaining_hours < 48:
        return f"剩余约 {int(remaining_hours)} 小时"
    return f"剩余约 {int(remaining_hours / 24)} 天"


def _format_memo_block(memos: list[Memo], now: float) -> str:
    """渲染完整备忘录文本块；无有效备忘时返回空串。

    条目字段拆成独立行，在窄面板中也能保持可读，同时便于模型抓取 id。
    """
    valid_memos = [memo for memo in memos if not memo.is_expired(now)]
    if not valid_memos:
        return ""

    lines: list[str] = ["## 我的备忘录", _MEMO_GUIDANCE, "", "### 当前条目"]
    # 按创建时间升序展示，让模型读到的时序自然
    for index, memo in enumerate(
        sorted(valid_memos, key=lambda item: item.created_at), start=1
    ):
        entry_lines = [
            f"#{index}",
            f"- id: {memo.memo_id}",
            f"- 内容: {memo.content}",
        ]
        if memo.intent.strip():
            entry_lines.append(f"- 动机: {memo.intent.strip()}")
        entry_lines.append(f"- 创建时间: {_format_datetime(memo.created_at)}")
        entry_lines.append(
            f"- 过期时间: {_format_datetime(memo.expires_at)}"
            f"（{_format_remaining(memo, now)}）"
        )
        lines.append("\n".join(entry_lines))

    return "\n".join(lines)


def build_memo_contribution(memos: list[Memo]) -> ContextContribution | None:
    """把备忘列表打包为一条 turn 级上下文贡献。

    Args:
        memos: 会话中的全部备忘。

    Returns:
        ContextContribution | None: 渲染结果；无有效备忘时返回 ``None``。
    """
    text = _format_memo_block(memos, time.time())
    if not text:
        return None
    return ContextContribution(
        source=_MEMO_SOURCE,
        owner="notice",
        priority=_MEMO_PRIORITY,
        content=text,
    )
