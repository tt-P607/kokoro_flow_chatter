"""KFC 上下文生命周期的定向回归测试。"""

from __future__ import annotations

import importlib.util
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plugins.kokoro_flow_chatter.context.planner import (  # noqa: E402
    build_last_mile_payload,
)
from plugins.kokoro_flow_chatter.context.renderer import (  # noqa: E402
    render_initial_context,
)
from plugins.kokoro_flow_chatter.context.types import InitialContextPlan  # noqa: E402
from plugins.kokoro_flow_chatter.config import KFCConfig  # noqa: E402
from plugins.kokoro_flow_chatter.handlers.voice_call_history_handler import (  # noqa: E402
    VoiceCallHistoryHandler,
)
from plugins.kokoro_flow_chatter.plugin import KFCPlugin  # noqa: E402
from plugins.kokoro_flow_chatter.session import KFCSession  # noqa: E402
from plugins.kokoro_flow_chatter.snapshot import (  # noqa: E402
    DYNAMIC_BACKGROUND_MARKER,
    capture_snapshot,
    deserialize_snapshot,
)
from plugins.kokoro_flow_chatter.runtime.request_view import (  # noqa: E402
    _without_transient_payloads,
)
from src.app.plugin_system.types import (  # noqa: E402
    LLMPayload,
    ROLE,
    Text,
    ToolCall,
    ToolResult,
)


_MAX_PAYLOADS = 30


def _text(payload: LLMPayload) -> str:
    content = payload.content
    if not isinstance(content, list):
        content = [content]
    return "".join(part.text for part in content if isinstance(part, Text))


def test_context_snapshot_is_single_truth_and_ignores_legacy_chain() -> None:
    """旧 chain 字段直接忽略；新 USER/ASSISTANT 只进入并从快照恢复。"""
    legacy_data = {
        "user_id": "u1",
        "stream_id": "stream-1",
        "chain_payloads": [{"role": "user", "text": "旧链内容"}],
        "context_snapshot": [
            {"role": "user", "content": [{"type": "text", "text": "旧快照"}]}
        ],
    }
    loaded = KFCSession.from_dict(legacy_data)
    assert not hasattr(loaded, "chain_payloads")
    assert _deserialize_user_text(loaded.context_snapshot) == "旧快照"

    loaded.append_context_entries(
        [
            LLMPayload(ROLE.USER, Text("你好")),
            LLMPayload(ROLE.ASSISTANT, Text("你好呀")),
        ],
        _MAX_PAYLOADS,
    )
    reloaded = KFCSession.from_dict(loaded.to_dict())
    restored = deserialize_snapshot(reloaded.context_snapshot)
    assert restored is not None
    assert [_text(payload) for payload in restored] == ["旧快照", "你好", "你好呀"]
    assert "chain_payloads" not in reloaded.to_dict()
    assert importlib.util.find_spec(
        "plugins.kokoro_flow_chatter.domain.chain_entry"
    ) is None


def test_dynamic_background_and_transients_do_not_enter_snapshot() -> None:
    """通道、摘要、叙事、last-mile 和 retry 都不能被捕获为历史。"""
    background_text = (
        f"{DYNAMIC_BACKGROUND_MARKER}\n\n"
        "[当前通道参数]\n\n---\n\n【近期记忆】长期摘要\n\n---\n\n融合叙事"
    )
    payloads = [
        LLMPayload(ROLE.SYSTEM, Text("稳定系统规则")),
        LLMPayload(ROLE.USER, Text(background_text)),
        LLMPayload(ROLE.USER, Text("真实用户输入")),
        LLMPayload(ROLE.ASSISTANT, Text("真实回复")),
    ]
    snapshot = capture_snapshot(payloads, _MAX_PAYLOADS)
    assert snapshot is not None
    text = str(snapshot)
    assert "稳定系统规则" not in text
    for forbidden in ("当前通道参数", "长期摘要", "融合叙事"):
        assert forbidden not in text

    source = [LLMPayload(ROLE.USER, Text("真实用户输入"))]
    transients = [
        build_last_mile_payload(),
        LLMPayload(ROLE.USER, Text("重试提醒")),
    ]
    result = [
        LLMPayload(ROLE.USER, Text("<system_reminder>提醒</system_reminder>真实用户输入")),
        *transients,
        LLMPayload(ROLE.ASSISTANT, Text("真实回复")),
    ]
    persistent = _without_transient_payloads(
        result,
        source_payloads=source,
        transient_count=len(transients),
    )
    assert [_text(payload) for payload in persistent] == ["真实用户输入", "真实回复"]
    assert "请务必使用工具" not in str(persistent)
    assert "重试提醒" not in str(persistent)


def test_tool_continuation_shape_survives_snapshot_reload() -> None:
    """闭合工具链的调用与回执按原顺序恢复。"""
    payloads = [
        LLMPayload(ROLE.USER, Text("查一下")),
        LLMPayload(
            ROLE.ASSISTANT,
            [ToolCall(id="call-1", name="action-kfc_reply", args={"q": "天气"})],
        ),
        LLMPayload(
            ROLE.TOOL_RESULT,
            ToolResult(value="晴", call_id="call-1", name="action-kfc_reply"),
        ),
        LLMPayload(ROLE.ASSISTANT, Text("今天是晴天")),
    ]
    restored = deserialize_snapshot(capture_snapshot(payloads, _MAX_PAYLOADS))
    assert restored is not None
    assert [payload.role for payload in restored] == [
        ROLE.USER,
        ROLE.ASSISTANT,
        ROLE.TOOL_RESULT,
        ROLE.ASSISTANT,
    ]


@pytest.mark.asyncio
async def test_voice_call_backfills_then_reloads_from_snapshot() -> None:
    """voice call 直接写入快照；重启重建动态背景后仍能读到该记录。"""
    session = KFCSession(user_id="u1", stream_id="stream-voice")
    store = _FakeStore(session)
    plugin = KFCPlugin.__new__(KFCPlugin)
    plugin.config = KFCConfig()
    plugin.session_store = store
    handler = VoiceCallHistoryHandler.__new__(VoiceCallHistoryHandler)
    handler.plugin = plugin

    _, params = await handler.execute(
        "voice_call.ended",
        {
            "previous_chatter_signature": (
                "kokoro_flow_chatter:chatter:kokoro_flow_chatter"
            ),
            "caller_stream_id": "stream-voice",
            "duration_seconds": 61,
            "messages_in_call": [
                {"role": "user", "text": "通话里说什么？", "ts": 20},
                {"role": "assistant", "text": "说了测试。"},
            ],
        },
    )
    assert params is not None
    assert store.saved == [session]
    text = str(session.context_snapshot)
    assert "通话里说什么？" in text
    assert "说了测试。" in text

    chat_stream = SimpleNamespace(
        platform="qq",
        chat_type="private",
        bot_id="bot",
        bot_nickname="Bot",
        stream_name="对方",
        context=SimpleNamespace(history_messages=[]),
    )

    async def system_prompt(_stream: object, _extra: dict[str, str] | None) -> str:
        return "SYSTEM"

    _, history, _ = await render_initial_context(
        chat_stream=chat_stream,
        plan=InitialContextPlan(history_summary="动态摘要"),
        mental_log=None,
        serialized_context_snapshot=session.context_snapshot,
        build_system_prompt_fn=system_prompt,
        build_fused_narrative_fn=lambda _stream, _log: "动态叙事",
    )
    payload_text = "\n".join(_text(payload) for payload in history)
    assert "通话里说什么？" in payload_text
    assert "说了测试。" in payload_text


def _deserialize_user_text(snapshot: list[dict[str, object]] | None) -> str:
    if not snapshot:
        return ""
    content = snapshot[0].get("content")
    if not isinstance(content, list) or not content:
        return ""
    first = content[0]
    if not isinstance(first, dict):
        return ""
    return str(first.get("text", ""))


class _FakeStore:
    def __init__(self, session: KFCSession) -> None:
        self.session = session
        self.saved: list[KFCSession] = []

    @asynccontextmanager
    async def lock(self, _stream_id: str):
        yield

    async def get_or_create(self, _stream_id: str) -> KFCSession:
        return self.session

    async def save(self, session: KFCSession) -> None:
        self.saved.append(session)
