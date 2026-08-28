"""KFC 通用 external resume 协议测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.app.plugin_system.base import Stop, Wait, WaitResumeEvent  # noqa: E402
from src.app.plugin_system.types import LLMPayload, Text  # noqa: E402

from plugins.kokoro_flow_chatter.chatter import (  # noqa: E402
    KokoroFlowChatter,
)
from plugins.kokoro_flow_chatter.config import KFCConfig  # noqa: E402
from plugins.kokoro_flow_chatter.context.types import (  # noqa: E402
    ContextContribution,
    ContextPlan,
)
from plugins.kokoro_flow_chatter.runtime.orchestrator import (  # noqa: E402
    _clear_external_resume_metadata,
    _set_external_resume_metadata,
)
from plugins.kokoro_flow_chatter.runtime.turn_controller import (  # noqa: E402
    prepare_turn_input,
)
from plugins.kokoro_flow_chatter.services.timeout_service import (  # noqa: E402
    TimeoutService,
)
from plugins.kokoro_flow_chatter.session import KFCSession  # noqa: E402


class _Response:
    """最小可追加 payload 的响应链替身。"""

    def __init__(self) -> None:
        self.payloads: list[LLMPayload] = []
        self.meta_data: dict[str, Any] = {}

    def add_payload(self, payload: LLMPayload) -> None:
        """追加一条持久 payload。"""
        self.payloads.append(payload)


class _TimeoutService:
    """永不触发超时的替身。"""

    @staticmethod
    def check_timeout(_session: KFCSession) -> bool:
        """返回未超时。"""
        return False


class _Chatter:
    """prepare_turn_input 所需的最小 Chatter 替身。"""

    def __init__(
        self,
        event: WaitResumeEvent | None,
        unread_msgs: list[Any] | None = None,
    ) -> None:
        self.stream_id = "target"
        self._event = event
        self._unread_msgs = list(unread_msgs or [])
        self.take_calls = 0

    async def fetch_unreads(self, time_format: str = "") -> tuple[str, list[Any]]:
        """返回固定未读快照。"""
        _ = time_format
        return "", list(self._unread_msgs)

    async def flush_unreads(self, unread_messages: list[Any]) -> int:
        """从替身未读中移除指定消息。"""
        ids = {id(message) for message in unread_messages}
        self._unread_msgs = [
            message for message in self._unread_msgs if id(message) not in ids
        ]
        return len(unread_messages)

    def take_external_resume(self) -> WaitResumeEvent | None:
        """一次性取得恢复事件。"""
        self.take_calls += 1
        event = self._event
        self._event = None
        return event

    @staticmethod
    def format_message_line(message: Any, time_format: str = "") -> str:
        """渲染测试消息。"""
        _ = time_format
        return str(message.processed_plain_text)

    @staticmethod
    def extract_timestamp(message: Any) -> float:
        """返回测试消息时间。"""
        return float(message.time)

    @staticmethod
    def record_reply_timing(_session: KFCSession) -> None:
        """测试中无需记录时效。"""


def _text(payload: LLMPayload | None) -> str:
    """提取 payload 中的文本。"""
    if payload is None:
        return ""
    content = payload.content
    parts = content if isinstance(content, list) else [content]
    return "".join(part.text for part in parts if isinstance(part, Text))


@pytest.mark.asyncio
async def test_execute_receives_core_style_asend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """门面必须接住框架通过 asend 送入的 WaitResumeEvent。"""
    received: list[WaitResumeEvent | None] = []

    async def fake_orchestrator(chatter: KokoroFlowChatter):
        yield Wait(0)
        received.append(chatter.take_external_resume())
        yield Stop(0)

    monkeypatch.setattr(
        "plugins.kokoro_flow_chatter.chatter.execute_orchestrator",
        fake_orchestrator,
    )
    chatter = KokoroFlowChatter("target", cast(Any, object()))
    generator = chatter.execute()
    assert isinstance(await anext(generator), Wait)

    event = WaitResumeEvent(source="external", extra={"resume_prompt": "继续"})
    assert isinstance(await generator.asend(event), Stop)
    assert received == [event]
    await generator.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["message", "timer"])
async def test_execute_ignores_core_wake_only_resume(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    """Core 的消息/定时唤醒不得占用 external resume 单槽。"""
    received: list[WaitResumeEvent | None] = []

    async def fake_orchestrator(chatter: KokoroFlowChatter):
        yield Wait(0)
        received.append(chatter.take_external_resume())
        yield Stop(0)

    monkeypatch.setattr(
        "plugins.kokoro_flow_chatter.chatter.execute_orchestrator",
        fake_orchestrator,
    )
    chatter = KokoroFlowChatter("target", cast(Any, object()))
    generator = chatter.execute()
    assert isinstance(await anext(generator), Wait)

    assert isinstance(
        await generator.asend(WaitResumeEvent(source=source)),
        Stop,
    )
    assert received == [None]
    await generator.aclose()


@pytest.mark.asyncio
async def test_message_wake_processes_second_unread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wait 后的 message 唤醒必须让下一条真实消息进入正常 USER 回合。"""

    async def fake_fetch_unreads(
        time_format: str = "",
    ) -> tuple[str, list[Any]]:
        _ = time_format
        return "", [message]

    async def fake_plan(**_kwargs: Any) -> ContextPlan:
        return ContextPlan(user_text="[新消息]\nsecond")

    message = SimpleNamespace(
        sender_id="user",
        sender_name="User",
        sender_cardname="",
        sender_role=None,
        processed_plain_text="second",
        content="second",
        message_id="m2",
        time=123.0,
    )
    monkeypatch.setattr(
        "plugins.kokoro_flow_chatter.runtime.turn_controller.plan_user_turn",
        fake_plan,
    )
    chatter = KokoroFlowChatter("target", cast(Any, object()))
    chatter.stage_external_resume(WaitResumeEvent(source="message", unread_count=1))
    monkeypatch.setattr(chatter, "fetch_unreads", fake_fetch_unreads)
    response = _Response()
    session = KFCSession(user_id="user", stream_id="target")

    result = await prepare_turn_input(
        chatter,
        response,
        SimpleNamespace(platform="qq"),
        KFCConfig(),
        session,
        cast(Any, _TimeoutService()),
        False,
    )

    assert result.unread_msgs == [message]
    assert result.persistent_user_payload is not None
    assert result.external_resume_request_marker == ""
    assert session.last_user_message_at == 123.0


@pytest.mark.asyncio
async def test_timer_wake_keeps_timeout_path_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wait 后的 timer 唤醒必须继续检查 KFC 自己的等待超时。"""

    async def fake_fetch_unreads(
        time_format: str = "",
    ) -> tuple[str, list[Any]]:
        _ = time_format
        return "", []

    async def empty_followup_payload(_stream_id: str) -> None:
        return None

    monkeypatch.setattr(
        "plugins.kokoro_flow_chatter.runtime.turn_controller._collect_followup_payload",
        empty_followup_payload,
    )
    config = KFCConfig()
    chatter = KokoroFlowChatter("target", cast(Any, object()))
    chatter.stage_external_resume(WaitResumeEvent(source="timer", wait_time=0))
    monkeypatch.setattr(chatter, "fetch_unreads", fake_fetch_unreads)
    session = KFCSession(user_id="user", stream_id="target")
    session.waiting_config.expected_reaction = "reply"
    session.waiting_config.max_wait_seconds = 1.0
    session.waiting_config.started_at = 1.0

    result = await prepare_turn_input(
        chatter,
        _Response(),
        SimpleNamespace(platform="qq"),
        config,
        session,
        TimeoutService(config),
        False,
    )

    assert result.request_only_payload is not None
    assert result.external_resume_request_marker == ""
    assert session.consecutive_timeout_count == 1
    assert session.is_waiting() is False


@pytest.mark.asyncio
async def test_cold_zero_unread_resume_uses_only_transient_contribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """0 unread resume 可形成回合，且不写 USER 活动与持久链。"""

    async def fake_plan(*_args: Any, **kwargs: Any) -> ContextPlan:
        assert kwargs["external_resume"].source == "external"
        assert kwargs["request_marker"]
        return ContextPlan(
            user_text="",
            contributions=[
                ContextContribution(
                    source="plugin.resume",
                    owner="notice",
                    priority=10,
                    content="需要处理的内部行动",
                )
            ],
        )

    monkeypatch.setattr(
        "plugins.kokoro_flow_chatter.runtime.turn_controller.plan_followup_contributions",
        fake_plan,
    )
    event = WaitResumeEvent(source="external")
    chatter = _Chatter(event)
    response = _Response()
    session = KFCSession(user_id="user", stream_id="target")
    initial_activity = session.last_activity_at

    result = await prepare_turn_input(
        cast(Any, chatter),
        response,
        SimpleNamespace(platform="qq"),
        KFCConfig(),
        session,
        cast(Any, _TimeoutService()),
        False,
    )

    assert result.continue_loop is False
    assert result.persistent_user_payload is None
    assert result.request_only_payload is None
    assert "内部行动" in _text(result.extra_payload)
    assert result.external_resume_request_marker
    assert response.payloads == []
    assert session.context_snapshot is None
    assert session.last_user_message_at is None
    assert session.last_activity_at == initial_activity
    assert len(session.mental_log) == 0


@pytest.mark.asyncio
async def test_external_resume_takes_precedence_over_real_unread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """external resume 优先于真实未读：独立行动轮，未读保留不消费。"""
    captured: dict[str, Any] = {}

    async def fake_plan(*_args: Any, **kwargs: Any) -> ContextPlan:
        captured.update(kwargs)
        return ContextPlan(
            user_text="",
            contributions=[
                ContextContribution(
                    source="plugin.resume",
                    owner="notice",
                    priority=10,
                    content="跨流行动指令",
                )
            ],
        )

    monkeypatch.setattr(
        "plugins.kokoro_flow_chatter.runtime.turn_controller.plan_followup_contributions",
        fake_plan,
    )
    message = SimpleNamespace(
        sender_id="user",
        sender_name="User",
        processed_plain_text="hello",
        content="hello",
        message_id="m1",
        time=123.0,
    )
    event = WaitResumeEvent(source="external")
    chatter = _Chatter(event, [message])
    response = _Response()
    session = KFCSession(user_id="user", stream_id="target")
    initial_activity = session.last_activity_at

    result = await prepare_turn_input(
        cast(Any, chatter),
        response,
        SimpleNamespace(platform="qq"),
        KFCConfig(),
        session,
        cast(Any, _TimeoutService()),
        False,
    )

    # 独立行动轮：resume 生效，真实未读不被写入/消费。
    assert "跨流行动指令" in _text(result.extra_payload)
    assert result.persistent_user_payload is None
    assert result.unread_msgs == []
    assert result.external_resume_request_marker
    assert captured["external_resume"] == event
    assert session.last_user_message_at is None
    assert session.last_activity_at == initial_activity
    assert len(session.mental_log) == 0
    # 未读保留在队列中。
    assert chatter._unread_msgs == [message]  # noqa: SLF001


@pytest.mark.asyncio
async def test_real_unread_processed_normally_after_resume_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resume 行动轮结束后，下一轮真实未读正常处理。"""
    captured: dict[str, Any] = {}

    async def fake_plan(**kwargs: Any) -> ContextPlan:
        captured.update(kwargs)
        return ContextPlan(user_text="[新消息]\nhello")

    monkeypatch.setattr(
        "plugins.kokoro_flow_chatter.runtime.turn_controller.plan_user_turn",
        fake_plan,
    )
    message = SimpleNamespace(
        sender_id="user",
        sender_name="User",
        processed_plain_text="hello",
        content="hello",
        message_id="m1",
        time=123.0,
    )
    # resume 槽位已空：行动轮已完成。
    chatter = _Chatter(None, [message])
    response = _Response()
    session = KFCSession(user_id="user", stream_id="target")

    result = await prepare_turn_input(
        cast(Any, chatter),
        response,
        SimpleNamespace(platform="qq"),
        KFCConfig(),
        session,
        cast(Any, _TimeoutService()),
        False,
    )

    assert result.unread_msgs == [message]
    assert result.persistent_user_payload is not None
    assert result.external_resume_request_marker == ""
    assert session.last_user_message_at == 123.0
    assert len(session.mental_log) == 1


@pytest.mark.asyncio
async def test_tool_continuation_defers_external_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未闭合工具链期间不得消费 external resume。"""

    async def empty_plan(*_args: Any, **_kwargs: Any) -> ContextPlan:
        return ContextPlan(user_text="")

    monkeypatch.setattr(
        "plugins.kokoro_flow_chatter.runtime.turn_controller.plan_followup_contributions",
        empty_plan,
    )
    chatter = _Chatter(WaitResumeEvent(source="external"))
    result = await prepare_turn_input(
        cast(Any, chatter),
        _Response(),
        SimpleNamespace(platform="qq"),
        KFCConfig(),
        KFCSession(user_id="user", stream_id="target"),
        cast(Any, _TimeoutService()),
        True,
    )

    assert chatter.take_calls == 0
    assert chatter._event is not None
    assert result.has_pending_tool_results is False
    assert result.external_resume_request_marker == ""


@pytest.mark.asyncio
async def test_empty_external_resume_does_not_create_fake_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """空 prompt 且无 contribution 时安全回 Wait。"""

    async def empty_plan(*_args: Any, **_kwargs: Any) -> ContextPlan:
        return ContextPlan(user_text="")

    monkeypatch.setattr(
        "plugins.kokoro_flow_chatter.runtime.turn_controller.plan_followup_contributions",
        empty_plan,
    )
    response = _Response()
    result = await prepare_turn_input(
        cast(Any, _Chatter(WaitResumeEvent(source="empty"))),
        response,
        SimpleNamespace(platform="qq"),
        KFCConfig(),
        KFCSession(user_id="user", stream_id="target"),
        cast(Any, _TimeoutService()),
        False,
    )

    assert isinstance(result.next_signal, Wait)
    assert result.continue_loop is True
    assert result.persistent_user_payload is None
    assert response.payloads == []


@pytest.mark.asyncio
async def test_explicit_resume_prompt_is_request_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """显式 resume_prompt 只能进入本次 RequestView。"""

    async def empty_plan(*_args: Any, **_kwargs: Any) -> ContextPlan:
        return ContextPlan(user_text="")

    monkeypatch.setattr(
        "plugins.kokoro_flow_chatter.runtime.turn_controller.plan_followup_contributions",
        empty_plan,
    )
    event = WaitResumeEvent(
        source="external",
        extra={"resume_prompt": "请处理内部事件"},
    )
    response = _Response()
    result = await prepare_turn_input(
        cast(Any, _Chatter(event)),
        response,
        SimpleNamespace(platform="qq"),
        KFCConfig(),
        KFCSession(user_id="user", stream_id="target"),
        cast(Any, _TimeoutService()),
        False,
    )

    assert _text(result.request_only_payload) == "请处理内部事件"
    assert result.persistent_user_payload is None
    assert response.payloads == []


def test_external_resume_metadata_is_transient() -> None:
    """external resume marker 只在当前 provider request 前后存在。"""
    response = _Response()
    _set_external_resume_metadata(
        response,
        request_marker="request-1",
        source="external",
    )
    assert response.meta_data == {
        "kfc_external_resume_request_marker": "request-1",
        "kfc_external_resume_source": "external",
    }

    _clear_external_resume_metadata(response)
    assert response.meta_data == {}
