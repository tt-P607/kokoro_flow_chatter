"""KFC 完整上下文快照序列化与恢复测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.app.plugin_system.types import (  # noqa: E402
    LLMPayload,
    ROLE,
    Text,
    ToolCall,
    ToolResult,
)

from plugins.kokoro_flow_chatter.context.renderer import (  # noqa: E402
    render_initial_context,
)
from plugins.kokoro_flow_chatter.context.types import InitialContextPlan  # noqa: E402
from plugins.kokoro_flow_chatter.session import KFCSession  # noqa: E402
from plugins.kokoro_flow_chatter.snapshot import (  # noqa: E402
    ContextSnapshot,
    SnapshotEntry,
    capture_snapshot_from_payloads,
    deserialize_snapshot,
    log_snapshot_restored,
    serialize_payloads,
)


def test_serialize_payloads_round_trip() -> None:
    """payload 链应序列化为快照条目，Text/ToolCall/ToolResult 被正确归类。"""
    payloads = [
        LLMPayload(ROLE.SYSTEM, Text("你是助手")),
        LLMPayload(ROLE.USER, Text("你好")),
        LLMPayload(ROLE.ASSISTANT, ToolCall("call-1", "kfc_reply", {"content": "hi"})),
        LLMPayload(ROLE.TOOL_RESULT, ToolResult("已发送", call_id="call-1")),
    ]

    entries = serialize_payloads(payloads)

    roles = [e.role for e in entries]
    assert "system" in roles
    assert "user" in roles
    assert "assistant" in roles
    assert "tool_result" in roles

    assistant = next(e for e in entries if e.role == "assistant")
    assert assistant.tool_calls[0]["name"] == "kfc_reply"
    assert assistant.tool_calls[0]["args"]["content"] == "hi"

    tool_result = next(e for e in entries if e.role == "tool_result")
    assert tool_result.call_id == "call-1"
    assert tool_result.content == "已发送"


def test_serialize_payloads_strips_system_reminder_from_user() -> None:
    """user 文本中的 system_reminder 块应被剥离，保留真正的对话内容。"""
    payloads = [
        LLMPayload(
            ROLE.USER,
            Text(
                "<system_reminder>\n[kfc_rule]\n【规则】...\n</system_reminder>\n"
                "# 新收到的消息\n@K 你好"
            ),
        ),
    ]

    entries = serialize_payloads(payloads)

    user_entry = next(e for e in entries if e.role == "user")
    assert "<system_reminder>" not in user_entry.content
    assert "[kfc_rule]" not in user_entry.content
    assert "# 新收到的消息" in user_entry.content
    assert "@K 你好" in user_entry.content


def test_capture_snapshot_and_round_trip() -> None:
    """快照应能序列化、反序列化并保留结构。"""
    payloads = [
        LLMPayload(ROLE.SYSTEM, Text("你是助手")),
        LLMPayload(ROLE.USER, Text("你好")),
    ]

    snapshot = capture_snapshot_from_payloads("stream-1", payloads)
    assert snapshot is not None
    assert snapshot.stream_id == "stream-1"
    assert len(snapshot.entries) == 2

    data = snapshot.to_dict()
    loaded = ContextSnapshot.from_dict(data)

    assert loaded.stream_id == "stream-1"
    assert loaded.version == snapshot.version
    assert [e.role for e in loaded.entries] == ["system", "user"]
    assert loaded.entries[0].content == "你是助手"


def test_capture_snapshot_none_on_empty() -> None:
    """无 stream_id 或空 payload 时不应生成快照。"""
    assert capture_snapshot_from_payloads("", []) is None
    assert capture_snapshot_from_payloads("stream-1", []) is None


def test_serialize_payloads_skips_tool_declaration() -> None:
    """ROLE.TOOL（工具 schema 声明）不应进入快照。"""
    tool_payload = LLMPayload(ROLE.TOOL, [])
    payloads = [
        LLMPayload(ROLE.SYSTEM, Text("你是助手")),
        tool_payload,
        LLMPayload(ROLE.USER, Text("你好")),
    ]

    entries = serialize_payloads(payloads)

    roles = [e.role for e in entries]
    assert "tool" not in roles
    assert roles == ["system", "user"]


def test_snapshot_restored_log_uses_panel() -> None:
    """恢复日志应调用 print_panel 输出醒目面板。"""
    snapshot = ContextSnapshot(
        stream_id="stream-1",
        entries=[SnapshotEntry(role="user", content="你好")],
    )

    with patch("plugins.kokoro_flow_chatter.snapshot.logger.print_panel") as mock_panel:
        log_snapshot_restored(snapshot, payload_count=2)
        mock_panel.assert_called_once()
        args, kwargs = mock_panel.call_args
        assert "上下文快照已恢复" in args[0]
        assert kwargs["border_style"] == "bright_cyan"


def test_deserialize_snapshot_skips_system_and_restores_history() -> None:
    """反序列化应跳过 system，恢复 user/assistant/tool_result 历史链。"""
    snapshot = ContextSnapshot(
        stream_id="stream-1",
        entries=[
            SnapshotEntry(role="system", content="你是助手"),
            SnapshotEntry(role="user", content="你好"),
            SnapshotEntry(
                role="assistant",
                tool_calls=[{"id": "call-1", "name": "kfc_reply", "args": {"content": "hi"}}],
            ),
            SnapshotEntry(role="tool_result", content="已发送", call_id="call-1"),
        ],
    )

    payloads = deserialize_snapshot(snapshot)

    roles = [p.role for p in payloads]
    assert roles == [ROLE.USER, ROLE.ASSISTANT, ROLE.TOOL_RESULT]

    assert payloads[0].content[0].text == "你好"
    assert payloads[1].content[0].name == "kfc_reply"
    assert payloads[2].content[0].call_id == "call-1"
    assert payloads[2].content[0].value == "已发送"


def test_deserialize_snapshot_drops_unpaired_tool_calls_tail() -> None:
    """反序列化应丢弃尾部悬挂的未配对 tool_calls。"""
    snapshot = ContextSnapshot(
        stream_id="stream-1",
        entries=[
            SnapshotEntry(role="user", content="你好"),
            SnapshotEntry(
                role="assistant",
                tool_calls=[{"id": "call-1", "name": "kfc_reply", "args": {}}],
            ),
            # 悬挂：call-1 后无 tool_result
        ],
    )

    payloads = deserialize_snapshot(snapshot)

    roles = [p.role for p in payloads]
    assert roles == [ROLE.USER]
    assert payloads[0].content[0].text == "你好"


def test_deserialize_snapshot_drops_unpaired_tool_calls_in_middle() -> None:
    """反序列化应丢弃中间位置的未配对 tool_calls（其后是 user 而非 tool_result）。"""
    snapshot = ContextSnapshot(
        stream_id="stream-1",
        entries=[
            SnapshotEntry(role="user", content="第一轮"),
            SnapshotEntry(
                role="assistant",
                tool_calls=[{"id": "call-ok", "name": "kfc_reply", "args": {}}],
            ),
            SnapshotEntry(role="tool_result", content="已执行", call_id="call-ok"),
            SnapshotEntry(role="assistant", content="好的"),
            # 中断：此 assistant 发出未配对的 tool_call，后面直接是 user
            SnapshotEntry(
                role="assistant",
                tool_calls=[{"id": "call-orphan", "name": "kfc_memo", "args": {}}],
            ),
            SnapshotEntry(role="user", content="第二轮"),
        ],
    )

    payloads = deserialize_snapshot(snapshot)

    roles = [p.role for p in payloads]
    # call-orphan 段被丢弃，保留 call-ok 完成的一轮
    assert roles == [ROLE.USER, ROLE.ASSISTANT, ROLE.TOOL_RESULT, ROLE.ASSISTANT, ROLE.USER]
    assert payloads[0].content[0].text == "第一轮"
    assert payloads[4].content[0].text == "第二轮"
    orphan_names = [
        c.name
        for p in payloads
        if p.role == ROLE.ASSISTANT
        for c in getattr(p, "content", [])
        if hasattr(c, "name")
    ]
    assert "kfc_memo" not in orphan_names


def test_deserialize_snapshot_drops_leading_orphan_assistant() -> None:
    """反序列化应丢弃起始处孤立的 assistant（无 user 起始）。"""
    snapshot = ContextSnapshot(
        stream_id="stream-1",
        entries=[
            SnapshotEntry(role="assistant", content="孤立的助手发言"),
            SnapshotEntry(role="user", content="你好"),
        ],
    )

    payloads = deserialize_snapshot(snapshot)

    roles = [p.role for p in payloads]
    assert roles == [ROLE.USER]


def test_session_round_trip_preserves_snapshot() -> None:
    """KFCSession 序列化往返应保留 context_snapshot。"""
    session = KFCSession(user_id="u1", stream_id="stream-1")
    session.context_snapshot = capture_snapshot_from_payloads(
        "stream-1",
        [
            LLMPayload(ROLE.USER, Text("你好")),
            LLMPayload(ROLE.ASSISTANT, Text("嗨")),
        ],
    )

    data = session.to_dict()
    loaded = KFCSession.from_dict(data, max_log_entries=50)

    assert loaded.context_snapshot is not None
    assert loaded.context_snapshot.stream_id == "stream-1"
    assert [e.role for e in loaded.context_snapshot.entries] == ["user", "assistant"]
    assert loaded.context_snapshot.entries[1].content == "嗨"


def test_session_round_trip_without_snapshot() -> None:
    """无快照的会话往返后 context_snapshot 应为 None。"""
    session = KFCSession(user_id="u1", stream_id="stream-1")

    loaded = KFCSession.from_dict(session.to_dict(), max_log_entries=50)

    assert loaded.context_snapshot is None


@pytest.mark.asyncio
async def test_render_initial_context_skip_narrative_keeps_time_anchor() -> None:
    """skip_narrative=True 时应跳过融合叙事，但仍保留当前时间锚点。"""
    from types import SimpleNamespace
    from typing import cast

    chat_stream = cast(
        Any,
        SimpleNamespace(
            stream_name="言柒",
            platform="qq",
            chat_type="private",
            bot_id="42",
            bot_nickname="狐狐",
            context=SimpleNamespace(history_messages=[]),
        ),
    )
    plan = InitialContextPlan(history_summary="近期摘要")

    async def _build_system_prompt(
        _chat_stream: Any,
        _extra_vars: dict[str, str] | None,
    ) -> str:
        return "稳定系统提示词"

    system_payloads, chain_payloads, has_history = await render_initial_context(
        chat_stream=chat_stream,
        plan=plan,
        mental_log=None,
        serialized_chain_payloads=[],
        skip_narrative=True,
        build_system_prompt_fn=_build_system_prompt,
        build_fused_narrative_fn=lambda _stream, _log, _before_ts: "融合叙事（不应出现）",
    )

    assert [payload.role for payload in system_payloads] == [ROLE.SYSTEM]
    assert [payload.role for payload in chain_payloads] == [ROLE.USER]
    dynamic_text = _text_of(chain_payloads[0])
    assert "近期摘要" in dynamic_text
    assert "融合叙事（不应出现）" not in dynamic_text
    assert "当前时间：" in dynamic_text
    assert has_history is False


def _text_of(payload: Any) -> str:
    """提取 payload 的文本内容。"""
    content = payload.content
    if not isinstance(content, list):
        content = [content]
    return "".join(item.text for item in content if isinstance(item, Text))
