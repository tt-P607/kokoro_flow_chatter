"""KFC 上下文快照序列化与恢复测试。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plugins.kokoro_flow_chatter.snapshot import (  # noqa: E402
    capture_snapshot,
    deserialize_snapshot,
    serialize_payloads,
    trim_snapshot,
)
from src.app.plugin_system.types import (  # noqa: E402
    Audio,
    Image,
    LLMPayload,
    ROLE,
    Text,
    ToolCall,
    ToolResult,
)
from src.kernel.llm.payload import ReasoningText  # noqa: E402


def _text_of(payload: LLMPayload) -> str:
    """提取 payload 内全部文本片段。"""
    content = payload.content
    if not isinstance(content, list):
        content = [content]
    return "".join(
        part.text for part in content if isinstance(part, (Text, ReasoningText))
    )


def test_serialize_round_trip_preserves_all_parts() -> None:
    """round-trip 应保留全部内容类型与字段。"""
    payloads = [
        LLMPayload(
            ROLE.USER,
            Text("你好"),
        ),
        LLMPayload(
            ROLE.ASSISTANT,
            [
                ReasoningText("内心推理", signature="sig-1"),
                Text("回复文本"),
                ToolCall(id="c1", name="action-kfc_reply", args={"content": ["hi"]}),
            ],
        ),
        LLMPayload(
            ROLE.TOOL_RESULT,
            ToolResult(value={"ok": True}, call_id="c1", name="action-kfc_reply"),
        ),
        LLMPayload(
            ROLE.USER,
            [Text("新消息"), Image("aGVsbG8=")],
        ),
    ]
    entries = serialize_payloads(payloads)
    restored = deserialize_snapshot(entries)
    assert restored is not None
    assert len(restored) == 4
    assert [payload.role for payload in restored] == [
        ROLE.USER,
        ROLE.ASSISTANT,
        ROLE.TOOL_RESULT,
        ROLE.USER,
    ]
    assert _text_of(restored[0]) == "你好"
    assert _text_of(restored[1]) == "内心推理回复文本"
    assert isinstance(restored[1].content, list)
    tool_calls = [
        part for part in restored[1].content if isinstance(part, ToolCall)
    ]
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "c1"
    assert tool_calls[0].args == {"content": ["hi"]}
    tool_results = [
        part for part in restored[2].content if isinstance(part, ToolResult)
    ]
    assert len(tool_results) == 1
    # dict 值在快照中被 JSON 序列化为字符串，恢复后保持文本语义
    assert '"ok": true' in str(tool_results[0].value)
    assert tool_results[0].call_id == "c1"
    assert any(isinstance(part, Image) for part in restored[3].content)


def test_serialize_skips_system_and_tool_roles() -> None:
    """SYSTEM / TOOL 声明不应进入快照。"""
    entries = serialize_payloads(
        [
            LLMPayload(ROLE.SYSTEM, Text("系统提示")),
            LLMPayload(ROLE.TOOL, Text("工具声明")),
            LLMPayload(ROLE.USER, Text("真实消息")),
        ]
    )
    assert len(entries) == 1
    assert entries[0]["role"] == ROLE.USER.value


def test_serialize_strips_system_reminder_from_user() -> None:
    """user 文本中的 system_reminder 块应被剥离。"""
    entries = serialize_payloads(
        [
            LLMPayload(
                ROLE.USER,
                Text("<system_reminder>\n提示内容\n</system_reminder>\n真实内容"),
            )
        ]
    )
    parts = entries[0]["content"]
    assert all("system_reminder" not in str(part.get("text", "")) for part in parts)
    text = "".join(str(part.get("text", "")) for part in parts)
    assert "真实内容" in text


def test_capture_snapshot_none_on_empty() -> None:
    """无可捕获内容时返回 None。"""
    assert capture_snapshot([], 30) is None


def test_trim_snapshot_keeps_user_head() -> None:
    """裁剪后链头必须是 USER，且条数不超过上限。"""
    entries = []
    for index in range(10):
        entries.append({"role": "user", "content": [{"type": "text", "text": f"u{index}"}]})
        entries.append({"role": "assistant", "content": [{"type": "text", "text": f"a{index}"}]})
    trimmed = trim_snapshot(entries, 5)
    assert len(trimmed) <= 5
    assert trimmed[0]["role"] == "user"


def test_trim_snapshot_drops_leading_non_user() -> None:
    """头部孤立 assistant 应被丢弃。"""
    trimmed = trim_snapshot(
        [
            {"role": "assistant", "content": [{"type": "text", "text": "孤立"}]},
            {"role": "user", "content": [{"type": "text", "text": "u"}]},
        ],
        30,
    )
    assert trimmed[0]["role"] == "user"


def test_deserialize_drops_unpaired_tail_tool_calls() -> None:
    """尾部未配对的 assistant(tool_calls) 悬挂段应被丢弃，避免发送时校验失败。"""
    entries = [
        {"role": "user", "content": [{"type": "text", "text": "u"}]},
        {"role": "assistant", "content": [{"type": "tool_call", "id": "c1", "name": "x", "args": {}}]},
    ]
    restored = deserialize_snapshot(entries)
    assert restored is not None
    assert len(restored) == 1
    assert restored[0].role == ROLE.USER


def test_deserialize_returns_none_on_empty_or_dirty() -> None:
    """空快照或无法还原为合法链时返回 None。"""
    assert deserialize_snapshot(None) is None
    assert deserialize_snapshot([]) is None
    assert deserialize_snapshot([{"role": "assistant", "content": []}]) is None


def test_deserialize_restores_audio_and_reasoning_redacted() -> None:
    """音频与带 redacted_data 的推理应完整还原。"""
    entries = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "带音频"},
                {"type": "audio", "data": "YXVkaW8="},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "reasoning", "text": "", "redacted_data": "REDACTED"}],
        },
    ]
    restored = deserialize_snapshot(entries)
    assert restored is not None
    assert any(isinstance(part, Audio) for part in restored[0].content)
    reasoning = [
        part for part in restored[1].content if isinstance(part, ReasoningText)
    ]
    assert reasoning and reasoning[0].redacted_data == "REDACTED"


def test_capture_snapshot_and_deserialize_round_trip() -> None:
    """capture + deserialize 整链 round-trip 合法。"""
    payloads = [
        LLMPayload(ROLE.USER, Text("第一轮")),
        LLMPayload(ROLE.ASSISTANT, [ReasoningText("想了一下"), Text("回复1")]),
        LLMPayload(
            ROLE.USER,
            Text("<system_reminder>\n动态注入\n</system_reminder>\n第二轮"),
        ),
        LLMPayload(
            ROLE.ASSISTANT,
            [
                ToolCall(id="c2", name="action-do_nothing", args={}),
            ],
        ),
        LLMPayload(
            ROLE.TOOL_RESULT,
            ToolResult(value="已选择不回复", call_id="c2", name="action-do_nothing"),
        ),
        LLMPayload(ROLE.ASSISTANT, Text("好的")),
    ]
    snapshot = capture_snapshot(payloads, 30)
    assert snapshot is not None
    restored = deserialize_snapshot(snapshot)
    assert restored is not None
    roles = [payload.role for payload in restored]
    assert roles[0] == ROLE.USER
    assert roles[-1] == ROLE.ASSISTANT
    # reminder 已剥离
    assert "system_reminder" not in _text_of(restored[1])


def test_restore_chain_keeps_dynamic_user_and_snapshot_history() -> None:
    """快照恢复拼接：最新动态 USER 为链头，快照完整历史接其后。

    这等价于 ``context_builder`` 的拼接语义（动态 USER 含最新日记总结，
    快照历史含真实用户消息与 bot 回复），确保两者都不丢失。
    """
    snapshot = capture_snapshot(
        [
            LLMPayload(ROLE.USER, Text("[新消息]\n你好")),
            LLMPayload(ROLE.ASSISTANT, Text("回复")),
        ],
        30,
    )
    restored = deserialize_snapshot(snapshot)
    assert restored is not None

    dynamic_user = LLMPayload(
        ROLE.USER,
        Text("[通道]\n\n---\n\n【日记总结】今天很开心。\n\n---\n\n[历史叙事]"),
    )
    final_chain = [dynamic_user, *restored]

    joined = "\n".join(_text_of(p) for p in final_chain)
    # 最新日记总结在链中（不再被快照覆盖）
    assert "【日记总结】今天很开心。" in joined
    # 快照历史完整保留
    assert "你好" in joined
    assert "回复" in joined
    # 首条为 USER，快照首条 USER 也保留（不丢首轮真实消息）
    assert final_chain[0].role == ROLE.USER
    assert final_chain[1].role == ROLE.USER
