"""KFC 核心协议测试。

覆盖领域模型、协议层、执行层、上下文层与运行时的关键行为契约。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.app.plugin_system.base import Wait  # noqa: E402

from src.app.plugin_system.types import (  # noqa: E402
    LLMPayload,
    ROLE,
    Text,
    ToolCall,
    ToolResult,
)

from plugins.kokoro_flow_chatter.context.renderer import (  # noqa: E402
    render_initial_context,
    render_turn_contributions,
)
from plugins.kokoro_flow_chatter.context.sources.history_source import (  # noqa: E402
    build_channel_payload,
    build_current_time_payload,
    build_fused_narrative,
    build_history_summary_payload,
)
from plugins.kokoro_flow_chatter.context.sources.memo_source import (  # noqa: E402
    build_memo_contribution,
)
from plugins.kokoro_flow_chatter.context.types import (  # noqa: E402
    ContextContribution,
    InitialContextPlan,
)
from plugins.kokoro_flow_chatter.domain.decision import (  # noqa: E402
    Decision,
    ProactiveSchedule,
    build_experience_snapshot,
)
from plugins.kokoro_flow_chatter.domain.turn_trigger import (  # noqa: E402
    TurnTrigger,
    classify_turn_trigger,
)
from plugins.kokoro_flow_chatter.execution.decision_executor import (  # noqa: E402
    calculate_typing_delay,
    execute_decision_draft,
    parse_content_segments,
)
from plugins.kokoro_flow_chatter.execution.runner import run_decision  # noqa: E402
from plugins.kokoro_flow_chatter.models import (  # noqa: E402
    DO_NOTHING,
    KFC_REPLY,
    Memo,
    ToolCallResult,
    WaitingConfig,
    clamp_expire_hours,
)
from plugins.kokoro_flow_chatter.protocol.decision_parser import (  # noqa: E402
    build_decision,
)
from plugins.kokoro_flow_chatter.protocol.response_normalizer import (  # noqa: E402
    normalize_response,
)
from plugins.kokoro_flow_chatter.protocol.tool_call_adapter import (  # noqa: E402
    build_decision_draft,
    extract_call_args,
    normalize_call_name,
)
from plugins.kokoro_flow_chatter.runtime.payload_hygiene import (  # noqa: E402
    heal_orphan_tool_results,
    strip_stale_reminder_prefixes,
)
from plugins.kokoro_flow_chatter.runtime.phase_machine import (  # noqa: E402
    ConversationPhase,
    can_accept_user_payload,
    has_tool_result_tail,
    phase_for_model_result,
    phase_for_turn_start,
)
from plugins.kokoro_flow_chatter.runtime.request_view import (  # noqa: E402
    _without_transient_payloads,
    build_request_view,
)
from plugins.kokoro_flow_chatter.runtime.turn_controller import (  # noqa: E402
    _final_signal,
)
from plugins.kokoro_flow_chatter.runtime.unread_policy import (  # noqa: E402
    filter_interrupt_messages,
    prefer_real_unreads,
)


# ── 测试替身 ──────────────────────────────────────────────


class _FakeSession:
    """最小会话替身，只实现等待状态查询。"""

    def __init__(self, waiting: bool = False) -> None:
        self._waiting = waiting

    def is_waiting(self) -> bool:
        """返回等待状态。"""
        return self._waiting


class _FakeResponse:
    """最小响应替身。"""

    def __init__(self, payloads: list[LLMPayload] | None = None) -> None:
        self.payloads = payloads or []
        self.call_list: list[Any] = []

    def add_payload(self, payload: LLMPayload) -> None:
        """按框架语义追加 payload。"""
        self.payloads.append(payload)


class _FakeConfig:
    """最小配置替身。"""

    debug = SimpleNamespace(show_prompt=False, show_response=False)
    reply = SimpleNamespace(
        typing_chars_per_sec=0.0,
        typing_delay_min=0.0,
        typing_delay_max=0.0,
    )


def _text_of(payload: LLMPayload) -> str:
    """提取 payload 的首个文本片段。"""
    content = payload.content
    part = content[0] if isinstance(content, list) else content
    assert isinstance(part, Text)
    return part.text


async def _unused_run_tool_call(
    _pending_calls: list[Any],
    _response: Any,
    _usable_map: Any,
    _trigger_msg: Any | None,
) -> list[tuple[bool, bool]]:
    """不应被调用的工具执行器。"""
    raise AssertionError("unexpected tool call")


# ── 领域模型 ──────────────────────────────────────────────


def test_turn_trigger_priority_is_stable() -> None:
    """触发优先级应固定，避免工具续轮被误判为新输入。"""
    waiting_session = cast(Any, _FakeSession(waiting=True))
    idle_session = cast(Any, _FakeSession(waiting=False))

    assert (
        classify_turn_trigger(
            has_unread=True,
            has_pending_tool_results=True,
            session=waiting_session,
            is_timeout=True,
        )
        is TurnTrigger.NEW_MESSAGES
    )
    assert (
        classify_turn_trigger(
            has_unread=False,
            has_pending_tool_results=True,
            session=waiting_session,
            is_timeout=True,
        )
        is TurnTrigger.FOLLOWUP_TOOL_RESULT
    )
    assert (
        classify_turn_trigger(
            has_unread=False,
            has_pending_tool_results=False,
            session=waiting_session,
            is_timeout=True,
        )
        is TurnTrigger.TIMEOUT_EXPIRED
    )
    assert (
        classify_turn_trigger(
            has_unread=False,
            has_pending_tool_results=False,
            session=idle_session,
            is_timeout=True,
        )
        is TurnTrigger.IDLE_WAIT
    )


def test_decision_exposes_semantic_properties() -> None:
    """Decision 的语义属性应保持单一含义。"""
    decision = Decision(
        has_meaningful_action=True,
        visible_reply_segments=["A", "B"],
        has_reply_action=True,
    )

    assert decision.should_reply is True
    assert decision.reply_text == "A\nB"
    assert decision.has_third_party_calls is False


def test_waiting_config_and_memo_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """等待配置与备忘边界值应可预测。"""
    inactive = WaitingConfig()
    assert inactive.is_active() is False
    assert inactive.get_elapsed_seconds() == 0.0
    assert inactive.is_timeout() is False

    monkeypatch.setattr("plugins.kokoro_flow_chatter.models.time.time", lambda: 15.0)
    active = WaitingConfig(expected_reaction="回", max_wait_seconds=10, started_at=5)
    assert active.is_active() is True
    assert active.get_elapsed_seconds() == 10.0
    assert active.is_timeout() is True
    assert WaitingConfig.from_dict(active.to_dict()).expected_reaction == "回"

    active.reset()
    assert active.to_dict() == {
        "expected_reaction": "",
        "max_wait_seconds": 0.0,
        "started_at": 0.0,
    }

    assert clamp_expire_hours(0) == 24.0
    assert clamp_expire_hours("bad") == 24.0  # type: ignore[arg-type]
    assert clamp_expire_hours(0.5) == 1.0
    assert clamp_expire_hours(9999) == 14 * 24.0


# ── 协议层 ────────────────────────────────────────────────


def test_tool_call_adapter_normalizes_without_side_effects() -> None:
    """工具调用适配层应只做无副作用的归一化。"""
    calls = [
        ToolCall(name="action-kfc_reply", args={"content": ["你好"]}, id="c1"),
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
    assert [call.normalized_name for call in draft.calls] == [
        "kfc_reply",
        "weather",
        "planner",
        "raw",
    ]
    assert draft.calls[1].args == {"city": "上海"}
    assert draft.calls[2].args == {}
    assert draft.calls[0].is_kfc_control is True
    assert draft.calls[1].is_kfc_control is False
    assert draft.calls[1].is_info_tool is True
    assert draft.calls[0].is_info_tool is False
    assert build_decision_draft(None).has_calls is False


def test_normalize_response_parses_compat_tool_calls() -> None:
    """正文内嵌的 compat 工具调用应被就地转成标准形态。"""
    response = SimpleNamespace(
        message=(
            '{"message":"", "tool_calls": ['
            '{"id":"reply1", "name":"action-kfc_reply", '
            '"args":{"content":["你好"]}}]}'
        ),
        call_list=[],
        payloads=[],
        reasoning_content=None,
    )

    assert normalize_response(response) is True
    assert response.call_list[0].name == "action-kfc_reply"
    assert response.call_list[0].args == {"content": ["你好"]}
    assert response.payloads[-1].role == ROLE.ASSISTANT
    # 已有标准调用时不应重复解析
    assert normalize_response(response) is False


def test_build_decision_extracts_replies_and_schedule() -> None:
    """决策收敛应提取可见回复、第三方调用与主动预约。"""
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
    call_list = [
        ToolCall(name="action-kfc_reply", args={}, id="reply"),
        ToolCall(name="tool-search", args='{"query":"q"}', id="search"),
        ToolCall(
            name="action-schedule_proactive",
            args={"delay_minutes": "bad", "reason": "想你"},
            id="schedule",
        ),
    ]

    decision = build_decision(result, call_list)

    assert decision.thought == "想法"
    assert decision.reply_text == "你好\n再见"
    assert decision.should_reply is True
    assert decision.has_third_party_calls is True
    assert [(call.name, call.call_id) for call in decision.third_party_calls] == [
        ("search", "search"),
        ("schedule_proactive", "schedule"),
    ]
    assert decision.proactive_schedule is not None
    assert decision.proactive_schedule.delay_minutes == 30.0
    assert decision.proactive_schedule.reason == "想你"


def test_experience_snapshot_is_closed_structural_metadata_only() -> None:
    """闭合回合快照只输出固定结构状态，pending 时不提前输出。"""
    decision = Decision(
        thought="internal thought",
        actions=[
            {"type": "draw_image", "prompt": "raw prompt"},
            {"type": "kfc_reply", "content": ["raw reply"]},
        ],
        has_reply_action=True,
        has_meaningful_action=True,
        has_failed_tool=True,
        proactive_schedule=ProactiveSchedule(delay_minutes=3.0),
    )

    snapshot = build_experience_snapshot(
        decision,
        result_signal="wait",
        wait_seconds=12.0,
    )

    assert snapshot is not None
    assert set(snapshot) == {
        "schema_version",
        "actor_round_closed",
        "result_signal",
        "has_meaningful_action",
        "reply_attempted",
        "chose_silence",
        "waiting_after_round",
        "wait_seconds",
        "attention_hint",
        "proactive_plan_exists",
        "attempted_action_names",
        "has_failed_action",
        "pending_tool_results",
    }
    assert snapshot["attempted_action_names"] == ["draw_image", "kfc_reply"]
    serialized = str(snapshot)
    assert "raw prompt" not in serialized
    assert "raw reply" not in serialized
    assert "internal thought" not in serialized
    assert build_experience_snapshot(
        decision,
        result_signal="stop",
        pending_tool_results=True,
    ) is None
    signal = _final_signal(
        Wait(0),
        decision,
        result_signal="wait",
        wait_seconds=12.0,
    )
    assert signal.step_data is not None
    assert signal.step_data["experience_snapshot"] == snapshot


# ── 执行层 ────────────────────────────────────────────────


def test_parse_content_segments_handles_all_shapes() -> None:
    """content 参数的三种形态都应正确解析。"""
    assert parse_content_segments(["A", " ", "B"]) == ["A", "B"]
    assert parse_content_segments('["A", "B"]') == ["A", "B"]
    assert parse_content_segments("单句") == ["单句"]
    assert parse_content_segments("") == []
    assert parse_content_segments(None) == []


def test_calculate_typing_delay_respects_bounds() -> None:
    """打字延迟应被夹到配置区间内。"""
    config = cast(
        Any,
        SimpleNamespace(
            reply=SimpleNamespace(
                typing_chars_per_sec=10.0,
                typing_delay_min=0.5,
                typing_delay_max=2.0,
            )
        ),
    )
    assert calculate_typing_delay("ab", config) == 0.5
    assert calculate_typing_delay("a" * 100, config) == 2.0
    assert calculate_typing_delay("a" * 10, config) == 1.0

    zero_speed = cast(
        Any,
        SimpleNamespace(
            reply=SimpleNamespace(
                typing_chars_per_sec=0.0,
                typing_delay_min=0.5,
                typing_delay_max=2.0,
            )
        ),
    )
    assert calculate_typing_delay("abc", zero_speed) == 0.0


@pytest.mark.asyncio
async def test_executor_routes_regular_action_to_framework() -> None:
    """普通动作交给框架，kfc_reply 仍由 KFC 特殊处理。"""
    draft = build_decision_draft(
        [
            ToolCall(name="action-draw_image", args={"prompt": "fox"}, id="draw1"),
            ToolCall(name="action-kfc_reply", args={"content": ["画好了"]}, id="reply1"),
        ]
    )
    response = _FakeResponse()
    framework_call_names: list[str] = []
    sent_segments: list[str] = []

    async def _run_tool_call(
        pending_calls: list[Any],
        target_response: Any,
        _usable_map: Any,
        _trigger_msg: Any | None,
    ) -> list[tuple[bool, bool]]:
        framework_call_names.extend(call.name for call in pending_calls)
        for call in pending_calls:
            target_response.add_payload(
                LLMPayload(
                    ROLE.TOOL_RESULT,
                    ToolResult(value="ok", call_id=call.id, name=call.name),
                )
            )
        return [(True, True) for _ in pending_calls]

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

    # 第三方工具必须先于回复执行，模型才能基于结果回话
    assert framework_call_names == ["action-draw_image"]
    assert sent_segments == ["画好了"]
    assert result.has_reply is True
    assert result.has_third_party is True
    assert [payload.role for payload in response.payloads] == [
        ROLE.TOOL_RESULT,
        ROLE.TOOL_RESULT,
    ]


@pytest.mark.asyncio
async def test_executor_only_first_segment_carries_reply_to() -> None:
    """分段回复时只有首段携带引用，避免重复引用块。"""
    draft = build_decision_draft(
        [
            ToolCall(
                name="action-kfc_reply",
                args={"content": ["第一句", "第二句"], "reply_to": "m1"},
                id="reply1",
            )
        ]
    )
    sent: list[tuple[str, str]] = []

    async def _execute_reply(
        content: str,
        _config: Any,
        _trigger_msg: Any | None,
        reply_to: str,
    ) -> bool:
        sent.append((content, reply_to))
        return True

    result = await execute_decision_draft(
        draft,
        _FakeResponse(),
        usable_map=cast(Any, {}),
        trigger_msg=None,
        config=cast(Any, _FakeConfig()),
        execute_reply_fn=_execute_reply,
        run_tool_call_fn=_unused_run_tool_call,
    )

    assert sent == [("第一句", "m1"), ("第二句", "")]
    assert result.has_failed_tool is False


@pytest.mark.asyncio
async def test_executor_marks_empty_and_duplicated_reply_as_failure() -> None:
    """空内容视为失败，重复回复被忽略。"""
    draft = build_decision_draft(
        [
            ToolCall(name="action-kfc_reply", args={"content": []}, id="reply1"),
            ToolCall(name="action-kfc_reply", args={"content": ["补发"]}, id="reply2"),
        ]
    )
    response = _FakeResponse()

    async def _execute_reply(
        _content: str,
        _config: Any,
        _trigger_msg: Any | None,
        _reply_to: str,
    ) -> bool:
        raise AssertionError("空内容不应触发发送")

    result = await execute_decision_draft(
        draft,
        response,
        usable_map=cast(Any, {}),
        trigger_msg=None,
        config=cast(Any, _FakeConfig()),
        execute_reply_fn=_execute_reply,
        run_tool_call_fn=_unused_run_tool_call,
    )

    assert result.has_failed_tool is True
    assert len(response.payloads) == 2


@pytest.mark.asyncio
async def test_run_decision_converges_call_list_to_decision() -> None:
    """执行入口应把 call_list 完整收敛为 Decision。"""
    response = _FakeResponse()
    response.call_list = [
        ToolCall(name="action-kfc_reply", args={"content": ["你好"]}, id="r1")
    ]
    sent: list[str] = []

    async def _execute_reply(
        content: str,
        _config: Any,
        _trigger_msg: Any | None,
        _reply_to: str,
    ) -> bool:
        sent.append(content)
        return True

    decision = await run_decision(
        response,
        usable_map=cast(Any, {}),
        trigger_msg=None,
        config=cast(Any, _FakeConfig()),
        execute_reply_fn=_execute_reply,
        run_tool_call_fn=_unused_run_tool_call,
    )

    assert sent == ["你好"]
    assert decision.reply_text == "你好"
    assert decision.should_reply is True


# ── 上下文层 ──────────────────────────────────────────────


def test_history_source_payload_builders() -> None:
    """摘要、时间与通道 payload 的分支应可预测。"""
    import datetime

    named_stream = cast(
        Any, SimpleNamespace(stream_name="言柒", context=SimpleNamespace(history_messages=[]))
    )
    unknown_stream = cast(
        Any, SimpleNamespace(stream_name="", context=SimpleNamespace(history_messages=[]))
    )

    assert build_history_summary_payload(named_stream, "") is None
    assert (
        _text_of(build_history_summary_payload(named_stream, "记忆") or cast(Any, None))
        == "【你对言柒的近期记忆】\n记忆"
    )
    assert (
        _text_of(build_history_summary_payload(unknown_stream, "记忆") or cast(Any, None))
        == "【你对对方的近期记忆】\n记忆"
    )

    time_payload = build_current_time_payload(datetime.datetime(2026, 5, 9, 22, 0))
    assert time_payload.role == ROLE.USER
    assert _text_of(time_payload) == "当前时间：2026-05-09 22:00"

    channel_payload = build_channel_payload(
        cast(
            Any,
            SimpleNamespace(
                platform="qq",
                chat_type="private",
                bot_id="42",
                bot_nickname="狐狐",
                stream_name="言柒",
            ),
        )
    )
    assert channel_payload.role == ROLE.USER
    assert "聊天平台：qq" in _text_of(channel_payload)
    assert "ID 42" in _text_of(channel_payload)


def test_build_fused_narrative_interleaves_messages_and_thoughts() -> None:
    """融合叙事应把消息与内心活动按时间线交织。"""
    from plugins.kokoro_flow_chatter.models import KFCEventType

    messages = [
        SimpleNamespace(
            time="bad", sender_name="A", sender_id="u", message_id="m0",
            processed_plain_text="忽略",
        ),
        SimpleNamespace(
            time=1.0, sender_name="A", sender_id="u", message_id="m1",
            processed_plain_text="早期",
        ),
        SimpleNamespace(
            time=2.0, sender_name="Bot", sender_id="bot", message_id="m2",
            processed_plain_text="机器人",
        ),
        SimpleNamespace(
            time=3.0, sender_name="Bot", sender_id="other",
            message_id="action_kfc_reply_1", processed_plain_text="动作回复",
        ),
        SimpleNamespace(
            time=4.0, sender_name="B", sender_id="u", message_id="",
            processed_plain_text="",
        ),
    ]
    mental_log = SimpleNamespace(
        entries=[
            SimpleNamespace(
                timestamp=2.5, event_type=KFCEventType.BOT_PLANNING, thought="想到你"
            ),
            SimpleNamespace(
                timestamp=3.5, event_type=KFCEventType.USER_MESSAGE, thought="忽略"
            ),
            SimpleNamespace(
                timestamp="bad", event_type=KFCEventType.BOT_PLANNING, thought="坏时间"
            ),
        ]
    )
    stream = cast(
        Any,
        SimpleNamespace(bot_id="bot", context=SimpleNamespace(history_messages=messages)),
    )

    narrative = build_fused_narrative(stream, cast(Any, mental_log), before_ts=4.0)

    assert "A [消息id:m1]说：早期" in narrative
    assert "你回复：机器人" in narrative
    assert "你回复：动作回复" in narrative
    assert "（你的内心：想到你）" in narrative
    assert "坏时间" not in narrative

    empty_stream = cast(
        Any, SimpleNamespace(bot_id="bot", context=SimpleNamespace(history_messages=[]))
    )
    assert build_fused_narrative(empty_stream, None) == ""


@pytest.mark.asyncio
async def test_initial_context_keeps_dynamic_content_out_of_system() -> None:
    """初始上下文只保留稳定 SYSTEM；动态背景不进入持久快照。"""
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

    snapshot = [
        {"role": "user", "content": [{"type": "text", "text": "旧用户输入"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "旧回复"}]},
    ]
    system_payloads, history_payloads, has_history = await render_initial_context(
        chat_stream=chat_stream,
        plan=plan,
        mental_log=None,
        serialized_context_snapshot=snapshot,
        build_system_prompt_fn=_build_system_prompt,
        build_fused_narrative_fn=lambda _stream, _log: "融合叙事",
    )

    assert [payload.role for payload in system_payloads] == [ROLE.SYSTEM]
    assert _text_of(system_payloads[0]) == "稳定系统提示词"
    assert [payload.role for payload in history_payloads] == [
        ROLE.USER,
        ROLE.USER,
        ROLE.ASSISTANT,
    ]
    dynamic_text = _text_of(history_payloads[0])
    assert "近期摘要" in dynamic_text
    assert "融合叙事" in dynamic_text
    assert _text_of(history_payloads[1]) == "旧用户输入"
    assert has_history is True

    from plugins.kokoro_flow_chatter.snapshot import capture_snapshot

    captured = capture_snapshot(history_payloads + history_payloads[:0], 30)
    assert captured is not None
    assert [entry["role"] for entry in captured] == ["user", "assistant"]
    assert all("近期摘要" not in str(entry) for entry in captured)


def test_render_turn_contributions_orders_by_owner_and_priority() -> None:
    """上下文贡献应按 owner 分区并在区内按优先级排序。"""
    assert render_turn_contributions([]) is None
    assert (
        render_turn_contributions(
            [ContextContribution(source="s", owner="notice", priority=0, content="  ")]
        )
        is None
    )

    payload = render_turn_contributions(
        [
            ContextContribution(source="a", owner="notice", priority=1, content="通知低"),
            ContextContribution(source="b", owner="notice", priority=9, content="通知高"),
            ContextContribution(source="c", owner="policy", priority=0, content="策略"),
            ContextContribution(
                source="legacy.on_prompt_build.extra",
                owner="notice",
                priority=5,
                content="遗留注入",
            ),
        ]
    )

    assert payload is not None
    text = _text_of(payload)
    assert text.index("[策略约束]") < text.index("通知高")
    assert text.index("通知高") < text.index("遗留注入")
    assert text.index("遗留注入") < text.index("通知低")
    assert "[SYSTEM REMINDER]\n遗留注入" in text


def test_memo_contribution_skips_expired_entries() -> None:
    """备忘渲染应跳过已过期条目。"""
    import time

    now = time.time()
    assert build_memo_contribution([]) is None
    assert (
        build_memo_contribution(
            [Memo(content="已过期", created_at=now - 100, expires_at=now - 1)]
        )
        is None
    )

    contribution = build_memo_contribution(
        [
            Memo(content="有效备忘", intent="记得问", created_at=now, expires_at=now + 7200),
            Memo(content="已过期", created_at=now - 100, expires_at=now - 1),
        ]
    )
    assert contribution is not None
    assert contribution.owner == "notice"
    assert "有效备忘" in contribution.content
    assert "已过期" not in contribution.content
    assert "剩余约" in contribution.content
    assert "小时" in contribution.content


# ── 运行时 ────────────────────────────────────────────────


def test_phase_machine_covers_all_branches() -> None:
    """相位状态机应覆盖等待、续轮、工具执行与提交路径。"""
    empty_response = _FakeResponse()
    tool_tail_response = _FakeResponse([LLMPayload(ROLE.TOOL_RESULT, Text("done"))])
    model_response = _FakeResponse()
    model_response.call_list = [ToolCall(name="action-kfc_reply", args={}, id="c1")]

    assert has_tool_result_tail(empty_response) is False
    assert has_tool_result_tail(tool_tail_response) is True
    assert (
        phase_for_turn_start(empty_response, has_pending_tool_results=False)
        is ConversationPhase.WAIT_INPUT
    )
    assert (
        phase_for_turn_start(empty_response, has_pending_tool_results=True)
        is ConversationPhase.FOLLOW_UP
    )
    assert (
        phase_for_turn_start(tool_tail_response, has_pending_tool_results=False)
        is ConversationPhase.FOLLOW_UP
    )
    assert phase_for_model_result(model_response) is ConversationPhase.TOOL_EXEC
    assert phase_for_model_result(empty_response) is ConversationPhase.COMMIT
    assert can_accept_user_payload(ConversationPhase.WAIT_INPUT) is True
    assert can_accept_user_payload(ConversationPhase.FOLLOW_UP) is False


def test_heal_orphan_tool_results_keeps_valid_chain() -> None:
    """孤立 TOOL_RESULT 应被移除，合法链路保持不变。"""
    orphan_response = _FakeResponse(
        [
            LLMPayload(ROLE.USER, Text("u")),
            LLMPayload(ROLE.TOOL_RESULT, ToolResult(value="bad", call_id="b1", name="b")),
            LLMPayload(ROLE.TOOL_RESULT, ToolResult(value="bad2", call_id="b2", name="b")),
        ]
    )
    assert heal_orphan_tool_results(orphan_response, where="unit") == 2
    assert [payload.role for payload in orphan_response.payloads] == [ROLE.USER]

    valid_response = _FakeResponse(
        [
            LLMPayload(ROLE.ASSISTANT, [ToolCall(name="action-kfc_reply", args={}, id="c1")]),
            LLMPayload(
                ROLE.TOOL_RESULT,
                ToolResult(value="ok", call_id="c1", name="action-kfc_reply"),
            ),
        ]
    )
    assert heal_orphan_tool_results(valid_response, where="valid") == 0


def test_strip_stale_reminder_prefixes_only_touches_middle_users() -> None:
    """只有中间 USER 的残留 reminder 前缀会被剥离。"""
    reminder = Text("<system_reminder>旧提醒</system_reminder>")
    response = _FakeResponse(
        [
            LLMPayload(ROLE.USER, [reminder, Text("首条")]),
            LLMPayload(ROLE.USER, [reminder, Text("中间")]),
            LLMPayload(ROLE.ASSISTANT, [Text("回复")]),
            LLMPayload(ROLE.USER, [reminder, Text("末条")]),
        ]
    )

    strip_stale_reminder_prefixes(response)

    assert len(cast(list[Any], response.payloads[0].content)) == 2
    assert len(cast(list[Any], response.payloads[1].content)) == 1
    assert _text_of(response.payloads[1]) == "中间"
    assert len(cast(list[Any], response.payloads[3].content)) == 2


def test_request_view_keeps_transient_payload_out_of_source() -> None:
    """构造发送视图不应污染原始主链。"""
    base_payload = LLMPayload(ROLE.USER, Text("主输入"))
    extra_payload = LLMPayload(ROLE.USER, Text("临时上下文"))
    response = cast(Any, SimpleNamespace(payloads=[base_payload]))

    view = build_request_view(response, [extra_payload])

    assert response.payloads == [base_payload]
    assert view.payloads == [base_payload, extra_payload]


def test_without_transient_payloads_restores_source_users() -> None:
    """裁剪辅助应同时去掉临时项并还原被 reminder 修改的 USER。"""
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
async def test_unread_policy_prefers_real_messages() -> None:
    """真实消息与主动触发撞车时保留真实消息。"""

    class _FakeUnreadChatter:
        """最小未读 IO 替身。"""

        def __init__(self) -> None:
            self.flushed: list[list[str]] = []

        async def flush_unreads(self, unread_messages: list[Any]) -> int:
            """记录被 flush 的消息 ID。"""
            self.flushed.append([msg.message_id for msg in unread_messages])
            return len(unread_messages)

        @staticmethod
        def format_message_line(msg: Any, time_format: str = "") -> str:
            """测试替身的渲染实现。"""
            _ = time_format
            return str(msg.message_id)

    proactive = SimpleNamespace(message_id="proactive_1")
    real = SimpleNamespace(message_id="u_1")
    chatter = _FakeUnreadChatter()

    kept = await prefer_real_unreads(cast(Any, chatter), [proactive, real])

    assert kept == [real]
    assert chatter.flushed == [["proactive_1"]]

    # 只有主动触发时不应丢弃
    only_proactive = await prefer_real_unreads(cast(Any, chatter), [proactive])
    assert only_proactive == [proactive]
    assert len(chatter.flushed) == 1


def test_filter_interrupt_messages_ignores_known_and_proactive() -> None:
    """已知消息与主动触发都不应打断生成。"""
    inputs = [
        SimpleNamespace(message_id="known_1"),
        SimpleNamespace(message_id="proactive_2"),
        SimpleNamespace(message_id="u_2"),
        SimpleNamespace(message_id=None),
    ]

    filtered = filter_interrupt_messages(inputs, frozenset({"known_1"}))

    assert [msg.message_id for msg in filtered] == ["u_2", None]


def test_control_constants_keep_expected_values() -> None:
    """控制动作名常量必须与提示词、schema 保持一致。"""
    assert KFC_REPLY == "kfc_reply"
    assert DO_NOTHING == "do_nothing"


# ── 执行层写回原子性 ──────────────────────────────────────


def test_reconcile_backfills_missing_tool_results() -> None:
    """已声明的工具调用若缺少 TOOL_RESULT，应补写占位回执。"""
    from plugins.kokoro_flow_chatter.execution.decision_executor import (
        _reconcile_missing_tool_results,
    )
    from plugins.kokoro_flow_chatter.protocol.tool_call_adapter import (
        build_decision_draft,
    )

    draft = build_decision_draft(
        [
            ToolCall(name="action-draw_image", args={"prompt": "fox"}, id="draw1"),
            ToolCall(name="action-kfc_reply", args={"content": ["画好了"]}, id="reply1"),
        ]
    )
    # 模拟 auto_append：链里已有 ASSISTANT(tool_calls)，但 draw_image 的
    # TOOL_RESULT 未写回（kfc_reply 已写回）
    response = _FakeResponse(
        [
            LLMPayload(
                ROLE.ASSISTANT,
                [
                    ToolCall(name="action-draw_image", args={}, id="draw1"),
                    ToolCall(name="action-kfc_reply", args={}, id="reply1"),
                ],
            ),
            LLMPayload(
                ROLE.TOOL_RESULT,
                ToolResult(value="已发送", call_id="reply1", name="action-kfc_reply"),
            ),
        ]
    )

    healed = _reconcile_missing_tool_results(response, draft)

    assert healed == 1
    results = [
        part.call_id
        for payload in response.payloads
        if payload.role == ROLE.TOOL_RESULT
        for part in payload.content
        if isinstance(part, ToolResult)
    ]
    assert set(results) == {"reply1", "draw1"}
    # 幂等：再次调用不再补写
    assert _reconcile_missing_tool_results(response, draft) == 0


@pytest.mark.asyncio
async def test_executor_reconciles_on_cancelled_framework_call() -> None:
    """第三方工具等待被取消时，应补写缺失 TOOL_RESULT 并让异常传播。"""
    draft = build_decision_draft(
        [
            ToolCall(name="action-draw_image", args={"prompt": "fox"}, id="draw1"),
            ToolCall(name="action-kfc_reply", args={"content": ["画好了"]}, id="reply1"),
        ]
    )
    response = _FakeResponse()
    # 模拟 auto_append：链里已有 ASSISTANT(tool_calls) 声明两个工具
    response.payloads.append(
        LLMPayload(
            ROLE.ASSISTANT,
            [
                ToolCall(name="action-draw_image", args={}, id="draw1"),
                ToolCall(name="action-kfc_reply", args={}, id="reply1"),
            ],
        )
    )

    async def _execute_reply(
        content: str,
        _config: Any,
        _trigger_msg: Any | None,
        _reply_to: str,
    ) -> bool:
        return True

    async def _cancel_run_tool_call(
        _pending_calls: list[Any],
        _target_response: Any,
        _usable_map: Any,
        _trigger_msg: Any | None,
    ) -> list[tuple[bool, bool]]:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await execute_decision_draft(
            draft,
            response,
            usable_map=cast(Any, {}),
            trigger_msg=object(),
            config=cast(Any, _FakeConfig()),
            execute_reply_fn=_execute_reply,
            run_tool_call_fn=_cancel_run_tool_call,
        )

    # 取消后 finally 已补写缺失的两个 TOOL_RESULT
    results = {
        part.call_id
        for payload in response.payloads
        if payload.role == ROLE.TOOL_RESULT
        for part in payload.content
        if isinstance(part, ToolResult)
    }
    assert results == {"draw1", "reply1"}
