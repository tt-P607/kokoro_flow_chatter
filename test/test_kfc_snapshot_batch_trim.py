"""KFC 快照无损累积的定向回归测试。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plugins.kokoro_flow_chatter.session import KFCSession  # noqa: E402
from plugins.kokoro_flow_chatter.snapshot import (  # noqa: E402
    capture_snapshot,
    deserialize_snapshot,
)
from src.app.plugin_system.types import LLMPayload, ROLE, Text  # noqa: E402


def _entry(role: ROLE, text: str) -> dict[str, object]:
    return {"role": role.value, "content": [{"type": "text", "text": text}]}


def _payload(text: str) -> LLMPayload:
    return LLMPayload(ROLE.USER, Text(text))


def test_snapshot_retains_all_entries_above_old_threshold() -> None:
    """超过旧条数阈值仍完整保留原话。"""
    payloads = []
    for index in range(80):
        payloads.append(LLMPayload(ROLE.USER, Text(f"u{index}")))
        payloads.append(LLMPayload(ROLE.ASSISTANT, Text(f"a{index}")))

    snapshot = capture_snapshot(payloads)

    assert snapshot is not None
    assert len(snapshot) == len(payloads)
    assert snapshot[0]["role"] == ROLE.USER.value
    texts = [part["text"] for entry in snapshot for part in entry["content"]]
    assert texts[0] == "u0"
    restored = deserialize_snapshot(snapshot)
    assert restored is not None


def test_append_retains_old_history_across_limit() -> None:
    """连续追加不会因经过原条数阈值而丢掉旧回合。"""
    session = KFCSession(user_id="u1", stream_id="stream-trim")
    session.context_snapshot = [
        _entry(ROLE.USER if index % 2 == 0 else ROLE.ASSISTANT, f"old-{index}")
        for index in range(79)
    ]
    changed = session.append_context_entries([_payload("new-user-1")])
    assert changed
    assert len(session.context_snapshot) == 80

    before = list(session.context_snapshot)
    assert session.append_context_entries([_payload("new-user-2")])
    assert len(session.context_snapshot) == 81
    assert session.context_snapshot[0] in before
    assert "new-user-2" not in str(before)
