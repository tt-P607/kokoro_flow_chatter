"""KFC 快照批量裁剪的定向回归测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plugins.kokoro_flow_chatter.config import KFCConfig  # noqa: E402
from plugins.kokoro_flow_chatter.services.summary_service import (  # noqa: E402
    SummaryService,
)
from plugins.kokoro_flow_chatter.session import KFCSession  # noqa: E402
from plugins.kokoro_flow_chatter.snapshot import (  # noqa: E402
    _batch_trim_to_recent_boundary,
    deserialize_snapshot,
    trim_snapshot,
)
from src.app.plugin_system.types import LLMPayload, ROLE, Text  # noqa: E402


def _entry(role: ROLE, text: str) -> dict[str, object]:
    return {"role": role.value, "content": [{"type": "text", "text": text}]}


def _payload(text: str) -> LLMPayload:
    return LLMPayload(ROLE.USER, Text(text))


def test_default_snapshot_capacity_is_100() -> None:
    """默认快照容量应为 100，旧滑动窗口语义同步更新。"""
    assert KFCConfig().prompt.max_context_payloads == 100


def test_snapshot_appends_without_trimming_below_threshold() -> None:
    """低于 80% 阈值时保持 append-only。"""
    entries = [_entry(ROLE.USER, "u1"), _entry(ROLE.ASSISTANT, "a1")]
    assert trim_snapshot(entries, 10) == entries


def test_snapshot_batch_trims_at_threshold_and_keeps_recent_boundary() -> None:
    """达到 80% 后一次性截掉旧前缀，实际边界可略多于 20% 目标。"""
    entries = []
    for index in range(80):
        entries.append(_entry(ROLE.USER, f"u{index}"))
        entries.append(_entry(ROLE.ASSISTANT, f"a{index}"))

    trimmed = trim_snapshot(entries, 100)

    assert 20 <= len(trimmed) <= 25
    assert trimmed[0]["role"] == ROLE.USER.value
    texts = [part["text"] for entry in trimmed for part in entry["content"]]
    assert texts[0] == "u70"
    restored = deserialize_snapshot(
        [{"role": item["role"], "content": item["content"]} for item in trimmed]
    )
    assert restored is not None


def test_custom_snapshot_capacity_uses_configured_ratios() -> None:
    """自定义容量按同一个 80%/20% 策略计算。"""
    entries = []
    for index in range(8):
        entries.append(_entry(ROLE.USER, f"u{index}"))
        entries.append(_entry(ROLE.ASSISTANT, f"a{index}"))

    trimmed = trim_snapshot(entries, 10)

    assert [item.get("content", [{}])[0].get("text") for item in trimmed] == [
        "u7",
        "a7",
    ]


def test_batch_cut_expands_to_complete_tool_segment() -> None:
    """机械切点落在工具段内时，向前扩展到最近的 USER 边界。"""
    entries = [
        _entry(ROLE.USER, "first"),
        _entry(ROLE.ASSISTANT, "answer-first"),
        _entry(ROLE.USER, "second"),
        _entry(ROLE.ASSISTANT, "call-second"),
        {"role": ROLE.TOOL_RESULT.value, "content": []},
        _entry(ROLE.ASSISTANT, "after-tool"),
        _entry(ROLE.USER, "third"),
    ]

    trimmed = _batch_trim_to_recent_boundary(entries, keep_count=4)

    assert [entry["role"] for entry in trimmed] == [
        ROLE.USER.value,
        ROLE.ASSISTANT.value,
        ROLE.TOOL_RESULT.value,
        ROLE.ASSISTANT.value,
        ROLE.USER.value,
    ]
    assert trimmed[0]["content"][0]["text"] == "second"


def test_batch_trim_then_append_preserves_new_prefix_and_skips_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """裁剪后可继续追加；裁剪路径不得触发近期记忆压缩。"""

    def fail_summary(*args: object, **kwargs: object):
        raise AssertionError("snapshot trimming must not schedule history summary")

    session = KFCSession(user_id="u1", stream_id="stream-trim")
    session.context_snapshot = [
        _entry(ROLE.USER if index % 2 == 0 else ROLE.ASSISTANT, f"old-{index}")
        for index in range(79)
    ]
    monkeypatch.setattr(
        SummaryService, "maybe_schedule_compression", staticmethod(fail_summary)
    )

    changed = session.append_context_entries([_payload("new-user-1")], 100)
    assert changed
    assert len(session.context_snapshot) == 20

    before = list(session.context_snapshot)
    assert session.append_context_entries([_payload("new-user-2")], 100)
    assert len(session.context_snapshot) == 21
    assert session.context_snapshot[0] in before
    assert "new-user-2" not in str(before)
