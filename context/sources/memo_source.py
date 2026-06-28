"""KFC 备忘录上下文 source。

把 ``KFCSession.memos`` 渲染成一个 turn 级 ContextContribution，
由 ContextRenderer 自动注入到本轮的 extra_payload（临时 USER payload，
发送后撤销，不进入持久化对话链）。

定位：让备忘录在每次发请求时都出现在用户提示词末尾，模型可以
"看到自己脑门上贴着的便签"，但不污染前缀缓存基础。
"""

from __future__ import annotations

import datetime
import time
from typing import TYPE_CHECKING

from ...models import Memo
from ..types import ContextContribution

if TYPE_CHECKING:
    pass


# 备忘录引导段（写死，不暴露为配置）
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


def _format_datetime(ts: float) -> str:
    """把时间戳格式化为人类可读字符串（年-月-日 时:分）。"""
    try:
        return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except (OSError, ValueError, OverflowError):
        return "未知时间"


def _format_remaining_text(memo: Memo, now: float) -> str:
    """把备忘剩余时间格式化为人类可读文本。

    粒度按小时取整（短于 1 小时显示 "剩余不到 1 小时"），
    避免每秒变化导致 prompt 不断变动、影响调试可读性。
    """
    remaining_sec = memo.remaining_seconds(now)
    if remaining_sec <= 0:
        return "已过期"
    remaining_hours = remaining_sec / 3600.0
    if remaining_hours < 1:
        return "剩余不到 1 小时"
    if remaining_hours < 24:
        return f"剩余约 {int(remaining_hours)} 小时"
    days = remaining_hours / 24.0
    if days < 2:
        return f"剩余约 {int(days * 24)} 小时"
    return f"剩余约 {int(days)} 天"


def _format_memo_block(memos: list[Memo], now: float | None = None) -> str:
    """把备忘列表渲染成完整文本块（含引导段 + 条目列表）。

    无有效备忘时返回空串。条目格式把关键字段拆成独立行，
    在窄面板/小宽度终端中也能保持可读，同时方便 LLM 抓取字段。
    """
    current = now if now is not None else time.time()
    valid_memos = [memo for memo in memos if not memo.is_expired(current)]
    if not valid_memos:
        return ""

    # 按 created_at 升序展示（先记的在前），让模型读起来时序自然
    sorted_memos = sorted(valid_memos, key=lambda memo: memo.created_at)

    lines: list[str] = ["## 我的备忘录", _MEMO_GUIDANCE, "", "### 当前条目"]
    for index, memo in enumerate(sorted_memos, start=1):
        created_str = _format_datetime(memo.created_at)
        expires_str = _format_datetime(memo.expires_at)
        remaining_str = _format_remaining_text(memo, current)

        entry_lines = [
            f"#{index}",
            f"- id: {memo.memo_id}",
            f"- 内容: {memo.content}",
        ]
        if memo.intent.strip():
            entry_lines.append(f"- 动机: {memo.intent.strip()}")
        entry_lines.extend(
            [
                f"- 创建时间: {created_str}",
                f"- 过期时间: {expires_str}（{remaining_str}）",
            ]
        )
        lines.append("\n".join(entry_lines))

    return "\n".join(lines)


def build_memo_contribution(memos: list[Memo]) -> ContextContribution | None:
    """把备忘列表打包为一个 turn 级 ContextContribution。

    返回值为 None 时表示无内容可注入（无有效备忘）。
    """
    text = _format_memo_block(memos)
    if not text:
        return None
    return ContextContribution(
        source="kfc.memo",
        owner="notice",
        scope="turn",
        priority=80,  # 高于一般 notice，紧贴最末尾
        ttl_turns=1,
        content=text,
    )
