"""kokoro_flow_chatter 重构后核心协议测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plugins.kokoro_flow_chatter.context.sources.history_source import restore_chain_payloads
from plugins.kokoro_flow_chatter.domain.chain_entry import ChainEntry
from plugins.kokoro_flow_chatter.domain.decision import Decision
from plugins.kokoro_flow_chatter.domain.turn_trigger import TurnTrigger, classify_turn_trigger
from plugins.kokoro_flow_chatter.execution.decision_executor import execute_decision_draft
from plugins.kokoro_flow_chatter.models import DO_NOTHING, KFC_REPLY, ToolCallResult
from plugins.kokoro_flow_chatter.protocol.decision_parser import build_decision
from plugins.kokoro_flow_chatter.protocol.tool_call_adapter import (
    build_decision_draft,
    extract_call_args,
    is_kfc_control_call,
    normalize_call_name,
)
from plugins.kokoro_flow_chatter.runtime.phase_machine import (
    ConversationPhase,
    can_accept_user_payload,
    has_tool_result_tail,
    phase_for_model_result,
    phase_for_turn_start,
)
from plugins.kokoro_flow_chatter.runtime.request_view import (
    _without_transient_payloads,
    build_request_view,
)
from plugins.kokoro_flow_chatter.runtime.turn_controller import build_chain_assistant_entry
from plugins.kokoro_flow_chatter.services.context_bridge import (
    ensure_tool_chain_closed,
    heal_orphan_tool_results,
    safe_add_payload,
)
from src.app.plugin_system.types import LLMPayload, ROLE, Text, ToolCall, ToolResult


class _FakeSession:
    """最小 session 替身。"""

    def __init__(self, waiting: bool = False) -> None:
        self._waiting = waiting

    def is_waiting(self) -> bool:
        """返回等待状态。"""
        return self._waiting


class _FakeResponse:
    """最小 response 替身。"""

    def __init__(self, payloads: list[LLMPayload] | None = None) -> None:
        self.payloads = payloads or []
        self.call_list: list[Any] = []

    def add_payload(self, payload: LLMPayload) -> None:
        """按框架语义追加 payload。"""
        self.payloads.append(payload)


class _FakeDebugConfig:
    """最小 debug 配置。"""

    show_prompt = False


class _FakeReplyConfig:
    """最小 reply 配置。"""

    typing_chars_per_sec = 0.0
    typing_delay_min = 0.0
    typing_delay_max = 0.0


class _FakeConfig:
    """最小 KFCConfig 替身。"""

    debug = _FakeDebugConfig()
    reply = _FakeReplyConfig()


class _CollectingResponse:
    """收集 payload 的最小 response。"""

    def __init__(self) -> None:
        self.payloads: list[LLMPayload] = []

    def add_payload(self, payload: LLMPayload) -> None:
        """追加 payload。"""
        self.payloads.append(payload)


def _text_of(payload: LLMPayload) -> str:
    """提取测试 payload 的首个文本片段。"""
    part = payload.content[0]
    assert isinstance(part, Text)
    return part.text


def test_phase_machine_covers_all_role_phase_branches() -> None:
    """role-phase 状态机应覆盖等待、续轮、工具执行和提交路径。"""
    empty_response = _FakeResponse()
    tool_tail_response = _FakeResponse([LLMPayload(ROLE.TOOL_RESULT, Text("done"))])
    model_response = _FakeResponse()
    model_response.call_list = [ToolCall(name="action-kfc_reply", args={}, id="c1")]

    assert has_tool_result_tail(empty_response) is False
    assert has_tool_result_tail(tool_tail_response) is True
    assert phase_for_turn_start(empty_response, has_pending_tool_results=False) is ConversationPhase.WAIT_INPUT
    assert phase_for_turn_start(empty_response, has_pending_tool_results=True) is ConversationPhase.FOLLOW_UP
    assert phase_for_turn_start(tool_tail_response, has_pending_tool_results=False) is ConversationPhase.FOLLOW_UP
    assert phase_for_model_result(model_response) is ConversationPhase.TOOL_EXEC
    assert phase_for_model_result(empty_response) is ConversationPhase.COMMIT
    assert can_accept_user_payload(ConversationPhase.WAIT_INPUT) is True
    assert can_accept_user_payload(ConversationPhase.COMMIT) is True
    assert can_accept_user_payload(ConversationPhase.FOLLOW_UP) is False


def test_turn_trigger_priority_and_idle_branch() -> None:
    """触发原因优先级应固定，避免 tool follow-up 被误当新输入。"""
    waiting_session = cast(Any, _FakeSession(waiting=True))
    idle_session = cast(Any, _FakeSession(waiting=False))

    assert classify_turn_trigger(
        has_unread=True,
        has_pending_tool_results=True,
        session=waiting_session,
        is_timeout=True,
    ) is TurnTrigger.NEW_MESSAGES
    assert classify_turn_trigger(
        has_unread=False,
        has_pending_tool_results=True,
        session=waiting_session,
        is_timeout=True,
    ) is TurnTrigger.FOLLOWUP_TOOL_RESULT
    assert classify_turn_trigger(
        has_unread=False,
        has_pending_tool_results=False,
        session=waiting_session,
        is_timeout=True,
    ) is TurnTrigger.TIMEOUT_EXPIRED
    assert classify_turn_trigger(
        has_unread=False,
        has_pending_tool_results=False,
        session=idle_session,
        is_timeout=True,
    ) is TurnTrigger.IDLE_WAIT


def test_chain_entry_schema_filters_dirty_data_and_serializes_cleanly() -> None:
    """ChainEntry 应集中约束 chain_payloads 的可用 schema。"""
    user_entry = ChainEntry.user("你好", ts=1.5)
    assistant_entry = ChainEntry.assistant("", [{"name": "action-kfc_reply", "args": {}}])

    assert user_entry.is_user is True
    assert user_entry.is_assistant is False
    assert user_entry.to_dict() == {"role": "user", "text": "你好", "ts": 1.5}
    assert assistant_entry.is_assistant is True
    assert assistant_entry.has_tool_calls is True
    assert assistant_entry.text == "好的。"
    assert assistant_entry.to_dict()["tool_calls"] == [{"name": "action-kfc_reply", "args": {}}]

    assert ChainEntry.from_dict({"role": "bad", "text": "x"}) is None
    assert ChainEntry.from_dict({"role": "user", "text": ""}) is None
    assert ChainEntry.from_dict({"role": "assistant", "text": "", "tool_calls": []}) is None
    restored = ChainEntry.from_dict(
        {
            "role": "assistant",
            "text": "",
            "tool_calls": [{"name": "action-kfc_reply"}, {"args": {"ignored": True}}],
            "ts": -1,
        }
    )
    assert restored is not None
    assert restored.text == "好的。"
    assert restored.ts is None
    assert restored.tool_calls == [{"name": "action-kfc_reply"}]


def test_restore_chain_payloads_keeps_only_readable_history() -> None:
    """历史读取只恢复用户可读文本，不把审计 tool_calls 再喂给模型。"""
    payloads = restore_chain_payloads(
        [
            {"role": "assistant", "text": "孤立开头会被丢弃"},
            {"role": "user", "text": "你好", "ts": 1.0},
            {
                "role": "assistant",
                "text": "你好呀~",
                "tool_calls": [
                    {
                        "name": "action-kfc_reply",
                        "id": "reply-1",
                        "args": {"content": ["你好呀~"]},
                    }
                ],
            },
            {"role": "bad", "text": "忽略"},
        ]
    )

    assert [payload.role for payload in payloads] == [ROLE.USER, ROLE.ASSISTANT]
    assert _text_of(payloads[0]) == "你好"
    assert _text_of(payloads[1]) == "你好呀~"


def test_tool_call_adapter_normalizes_and_extracts_all_branches() -> None:
    """tool call adapter 应只做无副作用规范化。"""
    calls = [
        ToolCall(name="action-kfc_reply", args={"content": ["你好"], "thought": "想回复"}, id="c1"),
        ToolCall(name="tool-weather", args='{"city": "上海"}', id="c2"),
        ToolCall(name="agent:planner", args="[]", id="c3"),
        ToolCall(name="raw", args="{bad json", id="c4"),
    ]

    draft = build_decision_draft(calls)

    assert normalize_call_name("") == ""
    assert normalize_call_name("agent:planner") == "planner"
    assert normalize_call_name("tool-weather") == "weather"
    assert normalize_call_name("raw") == "raw"
    assert extract_call_args({"a": 1}) == {"a": 1}
    assert extract_call_args("[]") == {}
    assert extract_call_args(1) == {}
    assert draft.has_calls is True
    assert [call.normalized_name for call in draft.calls] == ["kfc_reply", "weather", "planner", "raw"]
    assert draft.calls[1].args == {"city": "上海"}
    assert draft.calls[2].args == {}
    assert is_kfc_control_call(draft.calls[0]) is True
    assert is_kfc_control_call(draft.calls[1]) is False
    assert build_decision_draft(None).has_calls is False


def test_context_bridge_closes_user_after_tool_result_and_heals_orphans() -> None:
    """上下文桥接应闭合合法 tool 链，并移除孤立 TOOL_RESULT。"""
    call = ToolCall(name="action-kfc_reply", args={}, id="c1")
    response = _FakeResponse(
        [
            LLMPayload(ROLE.SYSTEM, Text("sys")),
            LLMPayload(ROLE.ASSISTANT, [call]),
            LLMPayload(ROLE.TOOL_RESULT, ToolResult(value="ok", call_id="c1", name="action-kfc_reply")),
        ]
    )

    assert ensure_tool_chain_closed(response, reason="unit") is True
    assert response.payloads[-1].role == ROLE.ASSISTANT
    assert _text_of(response.payloads[-1]) == "好的。"

    safe_add_payload(response, LLMPayload(ROLE.USER, Text("新输入")), reason="unit")
    assert response.payloads[-1].role == ROLE.USER
    assert ensure_tool_chain_closed(_FakeResponse(), reason="empty") is False
    assert ensure_tool_chain_closed(_FakeResponse([LLMPayload(ROLE.USER, Text("x"))]), reason="user") is False

    orphan_response = _FakeResponse(
        [
            LLMPayload(ROLE.USER, Text("u")),
            LLMPayload(ROLE.TOOL_RESULT, ToolResult(value="bad", call_id="bad", name="bad")),
            LLMPayload(ROLE.TOOL_RESULT, ToolResult(value="bad2", call_id="bad2", name="bad2")),
        ]
    )
    assert heal_orphan_tool_results(orphan_response, where="unit") == 2
    assert [payload.role for payload in orphan_response.payloads] == [ROLE.USER]

    valid_response = _FakeResponse(
        [
            LLMPayload(ROLE.ASSISTANT, [call]),
            LLMPayload(ROLE.TOOL_RESULT, ToolResult(value="ok", call_id="c1", name="action-kfc_reply")),
        ]
    )
    assert heal_orphan_tool_results(valid_response, where="valid") == 0


def test_request_view_keeps_transient_payload_out_of_source() -> None:
    """构造发送视图不应把 transient payload 写入原始 response。"""
    base_payload = LLMPayload(ROLE.USER, Text("主输入"))
    extra_payload = LLMPayload(ROLE.USER, Text("临时上下文"))
    response = SimpleNamespace(payloads=[base_payload])

    view = build_request_view(response, [extra_payload])

    assert response.payloads == [base_payload]
    assert view.payloads == [base_payload, extra_payload]


def test_without_transient_payloads_strips_extra_and_restores_user_payloads() -> None:
    """RequestView 裁剪辅助应同时去掉 extra 并还原被 reminder 修改的 USER。"""
    source_user = LLMPayload(ROLE.USER, Text("原始"))
    source_assistant = LLMPayload(ROLE.ASSISTANT, Text("旧回复"))
    injected_user = LLMPayload(ROLE.USER, [Text("注入"), Text("原始")])
    transient_user = LLMPayload(ROLE.USER, Text("临时"))
    new_assistant = LLMPayload(ROLE.ASSISTANT, Text("新回复"))

    stripped = _without_transient_payloads(
        [injected_user, source_assistant, transient_user, new_assistant],
        source_payloads=[source_user, source_assistant],
        transient_count=1,
    )
    unchanged = _without_transient_payloads(
        [injected_user, source_assistant],
        source_payloads=[source_user, source_assistant],
        transient_count=0,
    )

    assert stripped == [source_user, source_assistant, new_assistant]
    assert unchanged == [source_user, source_assistant]


@pytest.mark.asyncio
async def test_request_view_returns_raw_result_for_request_like_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """source 不是 LLMResponse 时，RequestView 应返回新结果而非回写字段。"""
    from src.kernel.llm.response import LLMResponse

    source = SimpleNamespace(
        payloads=[LLMPayload(ROLE.USER, Text("主输入"))],
        model_set=cast(Any, [{}]),
        context_manager=None,
        request_name="kfc_test",
        meta_data={},
    )

    async def _fake_send(self: Any, auto_append_response: bool = True, stream: bool = False) -> LLMResponse:
        _ = (self, auto_append_response, stream)
        return LLMResponse(
            _stream=None,
            _upper=cast(Any, SimpleNamespace(request_name="kfc_test", meta_data={})),
            _auto_append_response=False,
            payloads=list(self.payloads),
            model_set=cast(Any, [{}]),
            context_manager=None,
            message="ok",
            call_list=[],
        )

    monkeypatch.setattr("src.kernel.llm.request.LLMRequest.send", _fake_send)

    result = await build_request_view(source, []).send(auto_append_response=False, stream=False)

    assert result is not source
    assert result.message == "ok"
    assert getattr(result, "_consumed") is True


@pytest.mark.asyncio
async def test_request_view_syncs_appended_state_after_send(monkeypatch: pytest.MonkeyPatch) -> None:
    """RequestView 回写后再写 TOOL_RESULT 不应重复追加 ASSISTANT。"""
    from src.kernel.llm.response import LLMResponse

    base_payload = LLMPayload(ROLE.USER, Text("主输入"))
    assistant_call = ToolCall(name="action-kfc_reply", args={"content": ["你好"]}, id="c1")
    source = LLMResponse(
        _stream=None,
        _upper=cast(Any, SimpleNamespace(request_name="kfc_test", meta_data={})),
        _auto_append_response=True,
        payloads=[base_payload],
        model_set=cast(Any, [{}]),
        context_manager=None,
        message="",
        call_list=[],
    )

    async def _fake_send(self: Any, auto_append_response: bool = True, stream: bool = False) -> LLMResponse:
        _ = (self, stream)
        result = LLMResponse(
            _stream=None,
            _upper=cast(Any, SimpleNamespace(request_name="kfc_test", meta_data={})),
            _auto_append_response=auto_append_response,
            payloads=[base_payload],
            model_set=cast(Any, [{}]),
            context_manager=None,
            message="",
            call_list=[assistant_call],
        )
        await result
        return result

    monkeypatch.setattr("src.kernel.llm.request.LLMRequest.send", _fake_send)

    response = await build_request_view(source, []).send(auto_append_response=True, stream=False)
    response.add_payload(
        LLMPayload(
            ROLE.TOOL_RESULT,
            ToolResult(value="ok", call_id="c1", name="action-kfc_reply"),
        )
    )

    assistant_payloads = [payload for payload in response.payloads if payload.role == ROLE.ASSISTANT]
    assert len(assistant_payloads) == 1
    assert response.payloads[-1].role == ROLE.TOOL_RESULT


@pytest.mark.asyncio
async def test_request_view_strips_reminder_from_persistent_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """RequestView 应确保 transient payload 触发的 reminder 不被回写到 source。"""
    from src.kernel.llm.context import LLMContextManager
    from src.kernel.llm.response import LLMResponse

    source_user = LLMPayload(ROLE.USER, Text("原始输入"))
    source = LLMResponse(
        _stream=None,
        _upper=cast(Any, SimpleNamespace(request_name="kfc_test", meta_data={})),
        _auto_append_response=True,
        payloads=[source_user],
        model_set=cast(Any, [{}]),
        context_manager=LLMContextManager(),
        message="",
        call_list=[],
    )
    extra_payload = LLMPayload(ROLE.USER, Text("临时注入"))

    async def _fake_send(self: Any, auto_append_response: bool = True, stream: bool = False) -> LLMResponse:
        _ = stream
        injected_user = LLMPayload(ROLE.USER, [Text("【系统提示：我是注入的】"), Text("原始输入")])
        injected_extra = LLMPayload(ROLE.USER, [Text("【系统提示：我是注入的】"), Text("临时注入")])
        result = LLMResponse(
            _stream=None,
            _upper=cast(Any, SimpleNamespace(request_name="kfc_test", meta_data={})),
            _auto_append_response=auto_append_response,
            payloads=[injected_user, injected_extra],
            model_set=cast(Any, [{}]),
            context_manager=self.context_manager,
            message="回复内容",
            call_list=[],
        )
        await result
        return result

    monkeypatch.setattr("src.kernel.llm.request.LLMRequest.send", _fake_send)

    await build_request_view(source, [extra_payload]).send(auto_append_response=True, stream=False)

    assert len(source.payloads) == 2
    assert source.payloads[0].role == ROLE.USER
    assert _text_of(source.payloads[0]) == "原始输入"
    assert source.payloads[1].role == ROLE.ASSISTANT
    assert _text_of(source.payloads[1]) == "回复内容"


@pytest.mark.asyncio
async def test_executor_routes_regular_action_to_framework_but_keeps_kfc_reply_special() -> None:
    """普通 action 走框架执行，kfc_reply 仍由 KFC 特殊处理。"""
    calls = [
        ToolCall(name="action-draw_image", args={"prompt": "fox"}, id="draw1"),
        ToolCall(name="action-kfc_reply", args={"content": ["画好了"]}, id="reply1"),
    ]
    draft = build_decision_draft(calls)
    response = _CollectingResponse()
    framework_call_names: list[str] = []
    sent_segments: list[str] = []

    async def _run_tool_call(
        pending_calls: list[Any],
        _response: Any,
        _usable_map: Any,
        _trigger_msg: Any | None,
    ) -> list[tuple[bool, bool]]:
        framework_call_names.extend(call.name for call in pending_calls)
        for call in pending_calls:
            _response.add_payload(
                LLMPayload(
                    ROLE.TOOL_RESULT,
                    ToolResult(value="ok", call_id=call.id, name=call.name),
                )
            )
        return [(True, True) for _call in pending_calls]

    async def _execute_reply(
        content: str,
        _config: Any,
        _trigger_msg: Any | None,
        _reply_to: str,
    ) -> bool:
        sent_segments.append(content)
        return True

    result = await execute_decision_draft(
        draft,
        response,
        usable_map=cast(Any, {}),
        trigger_msg=object(),
        config=cast(Any, _FakeConfig()),
        execute_reply_fn=_execute_reply,
        run_tool_call_fn=_run_tool_call,
    )

    assert framework_call_names == ["action-draw_image"]
    assert sent_segments == ["画好了"]
    assert result.has_reply is True
    assert result.has_third_party is True
    assert [payload.role for payload in response.payloads] == [ROLE.TOOL_RESULT, ROLE.TOOL_RESULT]


@pytest.mark.asyncio
async def test_executor_handles_reply_failure_do_nothing_info_tools_and_hook() -> None:
    """执行层应覆盖回复失败、沉默、信息工具和 hook 分支。"""
    calls = [
        ToolCall(name="tool-search", args={"query": "q", "reason": "查资料"}, id="tool1"),
        ToolCall(name="action-kfc_reply", args={"content": ["第一句", "第二句"], "reply_to": "m1"}, id="reply1"),
        ToolCall(name="action-do_nothing", args={"max_wait_seconds": 3, "expected_reaction": "继续"}, id="none1"),
    ]
    draft = build_decision_draft(calls)
    response = _CollectingResponse()
    sent_segments: list[tuple[str, str]] = []
    hook_seen: list[ToolCallResult] = []

    async def _run_tool_call(
        pending_calls: list[Any],
        _response: Any,
        _usable_map: Any,
        _trigger_msg: Any | None,
    ) -> list[tuple[bool, bool]]:
        for call in pending_calls:
            _response.add_payload(LLMPayload(ROLE.TOOL_RESULT, ToolResult(value="ok", call_id=call.id, name=call.name)))
        return [(True, False) for _call in pending_calls]

    async def _execute_reply(content: str, _config: Any, _trigger_msg: Any | None, reply_to: str) -> bool:
        sent_segments.append((content, reply_to))
        return False

    result = await execute_decision_draft(
        draft,
        response,
        usable_map=cast(Any, {}),
        trigger_msg=None,
        config=cast(Any, _FakeConfig()),
        execute_reply_fn=_execute_reply,
        run_tool_call_fn=_run_tool_call,
        pre_execute_hook=hook_seen.append,
    )

    assert result.has_info_tool is True
    assert result.has_third_party is True
    assert result.has_reply is True
    assert result.has_do_nothing is True
    assert result.max_wait_seconds == 3
    assert sent_segments == [("第一句", "m1")]
    assert hook_seen == [result]
    assert [payload.role for payload in response.payloads] == [ROLE.TOOL_RESULT, ROLE.TOOL_RESULT, ROLE.TOOL_RESULT]


def test_decision_parser_builds_unified_decision_and_schedule() -> None:
    """build_decision 应统一提取可见回复、第三方调用和主动计划。"""
    result = ToolCallResult(
        thought="想法",
        mood="开心",
        expected_reaction="回应",
        max_wait_seconds=5,
        actions=[
            {"type": KFC_REPLY, "content": [" 你好 ", ""]},
            {"type": KFC_REPLY, "content": " 再见 "},
            {"type": "draw_image", "content": "ignored"},
        ],
        has_reply=True,
        has_third_party=True,
        has_info_tool=True,
    )
    response = SimpleNamespace(
        call_list=[
            ToolCall(name="action-kfc_reply", args={}, id="reply"),
            ToolCall(name="tool-search", args='{"query":"q"}', id="search"),
            ToolCall(name="action-schedule_proactive", args={"delay_minutes": "bad", "reason": "想你"}, id="schedule"),
            ToolCall(name="agent:planner", args="[]", id="planner"),
        ]
    )

    decision = build_decision(result, response)

    assert decision.thought == "想法"
    assert decision.reply_text == "你好\n再见"
    assert decision.should_reply is True
    assert decision.should_wait is True
    assert decision.should_end_turn is False
    assert decision.has_third_party_calls is True
    assert [(call.name, call.call_id, call.args) for call in decision.third_party_calls] == [
        ("search", "search", {"query": "q"}),
        ("schedule_proactive", "schedule", {"delay_minutes": "bad", "reason": "想你"}),
        ("planner", "planner", {}),
    ]
    assert decision.proactive_schedule is not None
    assert decision.proactive_schedule.delay_minutes == 30.0
    assert decision.proactive_schedule.reason == "想你"


def test_decision_properties_and_chain_assistant_entry() -> None:
    """Decision 属性和 chain assistant 构造应保持单一语义。"""
    info_decision = Decision(has_meaningful_action=True, has_info_tool_calls=True)
    final_decision = Decision(has_meaningful_action=True, visible_reply_segments=["A", "B"], has_reply_action=True)

    assert info_decision.should_end_turn is False
    assert final_decision.should_end_turn is True
    assert final_decision.reply_text == "A\nB"
    assert build_chain_assistant_entry("", [{"name": "action-kfc_reply"}])["text"] == "好的。"


def test_control_constants_keep_expected_values() -> None:
    """控制工具名称常量应保持与 prompt/tool schema 一致。"""
    assert KFC_REPLY == "kfc_reply"
    assert DO_NOTHING == "do_nothing"


def test_history_source_payload_builders_and_fused_narrative() -> None:
    """history source 的摘要、时间和融合叙事分支应可预测。"""
    import datetime
    from plugins.kokoro_flow_chatter.context.sources.history_source import (
        build_current_time_payload,
        build_fused_narrative,
        build_history_summary_payload,
    )
    from plugins.kokoro_flow_chatter.models import KFCEventType

    named_stream = SimpleNamespace(partner_name="言柒", group_name="群", context=SimpleNamespace(history_messages=[]))
    group_stream = SimpleNamespace(partner_name="", group_name="群聊", context=SimpleNamespace(history_messages=[]))
    unknown_stream = SimpleNamespace(partner_name="", group_name="", context=SimpleNamespace(history_messages=[]))

    assert build_history_summary_payload(named_stream, "") is None
    assert _text_of(build_history_summary_payload(named_stream, "记忆") or cast(Any, None)) == "【你对言柒的近期记忆】\n记忆"
    assert _text_of(build_history_summary_payload(group_stream, "记忆") or cast(Any, None)) == "【你对群聊的近期记忆】\n记忆"
    assert _text_of(build_history_summary_payload(unknown_stream, "记忆") or cast(Any, None)) == "【你对对方的近期记忆】\n记忆"
    assert _text_of(build_current_time_payload(datetime.datetime(2026, 5, 9, 22, 0))) == "当前时间：2026-05-09 22:00"

    messages = [
        SimpleNamespace(time="bad", sender_name="A", sender_id="u", message_id="m0", processed_plain_text="忽略"),
        SimpleNamespace(time=1.0, sender_name="A", sender_id="u", message_id="m1", processed_plain_text="早期"),
        SimpleNamespace(time=2.0, sender_name="Bot", sender_id="bot", message_id="m2", processed_plain_text="机器人"),
        SimpleNamespace(time=3.0, sender_name="Bot", sender_id="other", message_id="action_kfc_reply_1", processed_plain_text="动作回复"),
        SimpleNamespace(time=4.0, sender_name="B", sender_id="u", message_id="", processed_plain_text=""),
    ]
    mental_log = SimpleNamespace(
        entries=[
            SimpleNamespace(timestamp=2.5, event_type=KFCEventType.BOT_PLANNING, thought="想到你"),
            SimpleNamespace(timestamp=3.5, event_type=KFCEventType.USER_MESSAGE, thought="忽略"),
            SimpleNamespace(timestamp="bad", event_type=KFCEventType.BOT_PLANNING, thought="坏时间"),
        ]
    )
    stream = SimpleNamespace(bot_id="bot", context=SimpleNamespace(history_messages=messages))

    narrative = build_fused_narrative(stream, mental_log, before_ts=4.0)
    assert "A [消息id:m1]说：早期" in narrative
    assert "你回复：机器人" in narrative
    assert "你回复：动作回复" in narrative
    assert "（你的内心：想到你）" in narrative
    assert "坏时间" not in narrative
    assert build_fused_narrative(SimpleNamespace(bot_id="bot", context=SimpleNamespace(history_messages=[])), None) == ""


def test_models_waiting_config_and_visible_reply_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """models 中的等待配置和可见回复提取应覆盖所有分支。"""
    from plugins.kokoro_flow_chatter.models import KFCEventType, WaitingConfig, extract_visible_reply_text

    result = ToolCallResult(
        actions=[
            {"type": KFC_REPLY, "content": [" A ", "", "B"]},
            {"type": KFC_REPLY, "content": " C "},
            {"type": "other", "content": "D"},
        ]
    )
    assert extract_visible_reply_text(result) == "A\nB\nC"
    assert str(KFCEventType.USER_MESSAGE) == "user_message"

    inactive = WaitingConfig()
    assert inactive.is_active() is False
    assert inactive.get_elapsed_seconds() == 0.0
    assert inactive.is_timeout() is False
    assert inactive.get_progress() == 0.0

    monkeypatch.setattr("plugins.kokoro_flow_chatter.models.time.time", lambda: 15.0)
    active = WaitingConfig(expected_reaction="回", max_wait_seconds=10, started_at=5, followup_count=2)
    assert active.is_active() is True
    assert active.get_elapsed_seconds() == 10.0
    assert active.is_timeout() is True
    assert active.get_progress() == 1.0
    assert active.to_dict() == {
        "expected_reaction": "回",
        "max_wait_seconds": 10,
        "started_at": 5,
        "followup_count": 2,
    }
    restored = WaitingConfig.from_dict(active.to_dict())
    assert restored.expected_reaction == "回"
    assert restored.followup_count == 2
    active.reset()
    assert active.to_dict() == {"expected_reaction": "", "max_wait_seconds": 0.0, "started_at": 0.0, "followup_count": 0}


@pytest.mark.asyncio
async def test_parse_response_decision_delegates_execution() -> None:
    """parse_response_decision 应从 response.call_list 到 Decision 完整收敛。"""
    from plugins.kokoro_flow_chatter.protocol.decision_parser import parse_response_decision

    response = SimpleNamespace(payloads=[])
    response.add_payload = lambda payload: response.payloads.append(payload)
    response.call_list = [ToolCall(name="action-kfc_reply", args={"content": ["你好"]}, id="r1")]
    sent: list[str] = []

    async def _execute_reply(content: str, _config: Any, _trigger_msg: Any | None, _reply_to: str) -> bool:
        sent.append(content)
        return True

    async def _run_tool_call(
        _pending_calls: list[Any],
        _response: Any,
        _usable_map: Any,
        _trigger_msg: Any | None,
    ) -> list[tuple[bool, bool]]:
        return []

    decision = await parse_response_decision(
        response,
        usable_map={},
        trigger_msg=None,
        config=cast(Any, _FakeConfig()),
        execute_reply_fn=_execute_reply,
        run_tool_call_fn=_run_tool_call,
    )

    assert sent == ["你好"]
    assert decision.reply_text == "你好"
    assert decision.should_reply is True


def test_decision_parser_helper_fallbacks() -> None:
    """decision parser 私有归一化 helper 的容错分支应稳定。"""
    from plugins.kokoro_flow_chatter.protocol import decision_parser

    assert decision_parser._normalize_call_name("") == ""
    assert decision_parser._normalize_call_name("action-draw") == "draw"
    assert decision_parser._normalize_call_name("agent:plan") == "plan"
    assert decision_parser._extract_args({"x": 1}) == {"x": 1}
    assert decision_parser._extract_args("{bad") == {}
    assert decision_parser._extract_args("[]") == {}
    assert decision_parser._extract_args(1) == {}


@pytest.mark.asyncio
async def test_executor_covers_string_reply_delay_and_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    """执行层应覆盖字符串回复、打字延迟和 debug 日志分支。"""
    calls = [ToolCall(name="action-kfc_reply", args={"content": "单句"}, id="reply1")]
    draft = build_decision_draft(calls)
    response = _CollectingResponse()
    slept: list[float] = []
    sent: list[str] = []

    class _DebugConfig(_FakeConfig):
        """启用 debug 的配置。"""

        debug = SimpleNamespace(show_prompt=True)
        reply = SimpleNamespace(typing_chars_per_sec=1.0, typing_delay_min=0.1, typing_delay_max=1.0)

    monkeypatch.setattr("plugins.kokoro_flow_chatter.execution.decision_executor.asyncio.sleep", lambda delay: slept.append(delay) or _noop_async())

    async def _execute_reply(content: str, _config: Any, _trigger_msg: Any | None, _reply_to: str) -> bool:
        sent.append(content)
        return True

    result = await execute_decision_draft(
        draft,
        response,
        usable_map=cast(Any, {}),
        trigger_msg=None,
        config=cast(Any, _DebugConfig()),
        execute_reply_fn=_execute_reply,
        run_tool_call_fn=_unused_run_tool_call,
    )

    assert result.has_reply is True
    assert sent == ["单句"]
    assert slept == []


async def _noop_async() -> None:
    """供 monkeypatch 替换 async sleep 使用。"""
    return None


async def _unused_run_tool_call(
    _pending_calls: list[Any],
    _response: Any,
    _usable_map: Any,
    _trigger_msg: Any | None,
) -> list[tuple[bool, bool]]:
    """不应被调用的工具执行器。"""
    raise AssertionError("unexpected tool call")


def test_request_view_cutoff_break_and_context_bridge_edge_cases() -> None:
    """覆盖 RequestView 裁剪 break 与 context bridge 低层边界。"""
    from plugins.kokoro_flow_chatter.services import context_bridge

    source_user = LLMPayload(ROLE.USER, Text("原始"))
    assert _without_transient_payloads([], source_payloads=[source_user], transient_count=0) == []
    assert heal_orphan_tool_results(SimpleNamespace(payloads="bad"), where="bad") == 0

    with_pinned = _FakeResponse(
        [
            LLMPayload(ROLE.SYSTEM, Text("sys")),
            LLMPayload(ROLE.ASSISTANT, [ToolCall(name="tool-a", args={}, id="a")]),
            LLMPayload(ROLE.TOOL, Text("schema")),
            LLMPayload(ROLE.TOOL_RESULT, ToolResult(value="ok", call_id="a", name="tool-a")),
        ]
    )
    assert heal_orphan_tool_results(with_pinned, where="pinned") == 0

    assistant_text_only = LLMPayload(ROLE.ASSISTANT, Text("文字"))
    assistant_list_without_call = LLMPayload(ROLE.ASSISTANT, [Text("文字")])
    assert context_bridge._assistant_has_tool_calls(assistant_text_only) is False
    assert context_bridge._assistant_has_tool_calls(assistant_list_without_call) is False
    assert context_bridge._preview_payload(LLMPayload(ROLE.USER, Text("abc"))) == "Text('abc')"
