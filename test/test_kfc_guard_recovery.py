"""模型层拒答的恢复流程测试。

覆盖两类失败的分流、一次性 transient reminder 的生命周期，以及
被拦截内容不进入持久上下文的各种保证。

概念链路：

    响应 ──► Response Guard ──┬─ MODEL_REFUSAL ──► Guard recovery
                              └─ PASS ──────────► 格式纠正 / 正常决策
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plugins.kokoro_flow_chatter.config import KFCConfig  # noqa: E402
from plugins.kokoro_flow_chatter.runtime.guard_hook import (  # noqa: E402
    GuardCheck,
    check_response_guard,
)
from plugins.kokoro_flow_chatter.runtime.orchestrator import (  # noqa: E402
    _GUARD_RETRY_REMINDER,
    _PLAIN_TEXT_RETRY_REMINDER,
    _LoopState,
    _consume_guard_reminder,
    _handle_guard_refusal,
    _handle_plain_text_violation,
)
from plugins.kokoro_flow_chatter.runtime.request_view import (  # noqa: E402
    _without_transient_payloads,
    build_request_view,
)
from plugins.kokoro_flow_chatter.runtime.turn_controller import (  # noqa: E402
    TurnInputResult,
)
from plugins.kokoro_flow_chatter.snapshot import capture_snapshot  # noqa: E402
from src.app.plugin_system.base import Stop  # noqa: E402
from src.app.plugin_system.types import (  # noqa: E402
    LLMPayload,
    ROLE,
    Text,
    ToolCall,
)

_EVIDENCE = ("reasoning_explicit_safety_refusal", "reply_platform_refusal")
_PLAIN_REFUSAL = (
    "I cannot comply with this request because it violates safety guidelines."
)
_NORMAL_REPLY = "她轻轻笑了一下，伸手拍了拍你的肩膀。"


class _FakeResponse:
    """最小化模拟带 payloads/message/call_list 的响应对象。"""

    def __init__(
        self,
        payloads: list[LLMPayload],
        message: str = "",
        call_list: list[ToolCall] | None = None,
        reasoning: str = "",
    ) -> None:
        """构造响应替身。"""
        self.payloads = payloads
        self.message = message
        self.reasoning_content = reasoning
        self.reasoning_parts: list[Any] = []
        self.call_list: list[ToolCall] = call_list or []
        self.tool_call_compat = False


class _FakeGuardService:
    """response_guard 服务替身，按序返回预设判定。"""

    def __init__(self, blocks: bool = True) -> None:
        """构造服务替身。"""
        self.blocks = blocks
        self.calls: list[dict[str, Any]] = []

    async def inspect(self, **kwargs: Any) -> Any:
        """记录调用并返回预设判定。"""
        self.calls.append(kwargs)
        return SimpleNamespace(
            blocks=self.blocks,
            evidence=_EVIDENCE if self.blocks else (),
        )


def _user(text: str) -> LLMPayload:
    """构造 USER payload。"""
    return LLMPayload(ROLE.USER, [Text(text)])


def _assistant(text: str) -> LLMPayload:
    """构造 ASSISTANT payload。"""
    return LLMPayload(ROLE.ASSISTANT, [Text(text)])


def _state() -> _LoopState:
    """构造一个未绑定摘要同步器的循环状态。"""
    return _LoopState(summary=cast(Any, None))


def _config(max_retries: int = 1) -> KFCConfig:
    """构造只指定守卫重试上限的 KFC 配置。"""
    return KFCConfig(
        general=KFCConfig.GeneralSection(guard_max_retries=max_retries)
    )


def _plain_refusal_response() -> _FakeResponse:
    """构造纯文本安全拒答响应（无工具调用）。"""
    return _FakeResponse(
        payloads=[_user("[新消息]"), _assistant(_PLAIN_REFUSAL)],
        message=_PLAIN_REFUSAL,
        reasoning="安全政策不允许此类内容，我必须拒绝这个请求。",
    )


def _normal_plain_response() -> _FakeResponse:
    """构造守卫放行的普通纯文本响应（无工具调用）。"""
    return _FakeResponse(
        payloads=[_user("[新消息]"), _assistant(_NORMAL_REPLY)],
        message=_NORMAL_REPLY,
    )


def _tool_refusal_response() -> _FakeResponse:
    """构造包装在合法工具调用里的安全拒答响应。"""
    call = ToolCall(
        id="call-1", name="kfc_reply", args={"content": [_PLAIN_REFUSAL]}
    )
    return _FakeResponse(
        payloads=[_user("[新消息]"), LLMPayload(ROLE.ASSISTANT, [call])],
        message="",
        call_list=[call],
        reasoning="安全政策不允许此类内容，我必须拒绝这个请求。",
    )


def _reminder_text(payload: LLMPayload) -> str:
    """取出 payload 中的纯文本内容。"""
    return "".join(
        item.text for item in payload.content if isinstance(item, Text)
    )


# ── 一次性 reminder 的生命周期 ──────────────────────────────────────────


def test_guard_block_creates_single_reminder() -> None:
    """命中时创建一条 Guard 提示。"""
    state = _state()
    _handle_guard_refusal(
        _tool_refusal_response(),
        1,
        state,
        _config(),
        _EVIDENCE,
        from_tool_call=True,
    )

    assert state.guard_reminder is not None
    assert _GUARD_RETRY_REMINDER in _reminder_text(state.guard_reminder)


def test_reminder_is_one_shot() -> None:
    """提示只被消费一次，消费后槽位立即清空。"""
    state = _state()
    _handle_guard_refusal(
        _plain_refusal_response(),
        1,
        state,
        _config(),
        _EVIDENCE,
        from_tool_call=False,
    )

    assert len(_consume_guard_reminder(state)) == 1
    assert _consume_guard_reminder(state) == []
    assert state.guard_reminder is None


def test_consecutive_blocks_do_not_accumulate_reminders() -> None:
    """连续命中不会累积：每次请求最多一条提示。"""
    state = _state()
    for _ in range(3):
        response = _tool_refusal_response()
        _handle_guard_refusal(
            response, 1, state, _config(5), _EVIDENCE, from_tool_call=True
        )
        consumed = _consume_guard_reminder(state)
        assert len(consumed) == 1
        # 消费后槽位为空，下一次命中重新创建而不是复用同一条
        assert state.guard_reminder is None


def test_no_reminder_without_block() -> None:
    """未命中时槽位保持为空，不会凭空注入提示。"""
    state = _state()
    assert _consume_guard_reminder(state) == []


# ── 两类失败的分流 ──────────────────────────────────────────────────────


def test_plain_text_refusal_uses_guard_retry() -> None:
    """纯文本安全拒答走 Guard 重试，不消耗纯文本重试额度。"""
    state = _state()
    retry = _handle_guard_refusal(
        _plain_refusal_response(),
        1,
        state,
        _config(),
        _EVIDENCE,
        from_tool_call=False,
    )

    assert retry is True
    assert state.guard_retry_count == 1
    assert state.plain_text_retry_count == 0
    assert state.plain_text_reminders == []
    assert state.guard_reminder is not None


def test_plain_text_pass_uses_plain_text_retry() -> None:
    """守卫放行的普通纯文本走原有格式纠正，不消耗 Guard 额度。"""
    state = _state()
    retry = _handle_plain_text_violation(
        _normal_plain_response(), 1, state, _config(), [{}]
    )

    assert retry is True
    assert state.plain_text_retry_count == 1
    assert state.guard_retry_count == 0
    assert state.guard_reminder is None
    assert len(state.plain_text_reminders) == 1
    assert _PLAIN_TEXT_RETRY_REMINDER in _reminder_text(
        state.plain_text_reminders[0]
    )


def test_guard_path_clears_plain_text_reminders() -> None:
    """走 Guard 重试时清掉格式纠错提示，避免两种提示同时注入。"""
    state = _state()
    state.plain_text_reminders.append(LLMPayload(ROLE.USER, Text("旧提醒")))

    _handle_guard_refusal(
        _plain_refusal_response(),
        1,
        state,
        _config(),
        _EVIDENCE,
        from_tool_call=False,
    )

    assert state.plain_text_reminders == []


def test_plain_text_path_clears_guard_reminder() -> None:
    """走格式纠正时清掉上一轮的 Guard 提示。"""
    state = _state()
    state.guard_reminder = LLMPayload(ROLE.USER, Text(_GUARD_RETRY_REMINDER))

    _handle_plain_text_violation(
        _normal_plain_response(), 1, state, _config(), [{}]
    )

    assert state.guard_reminder is None
    assert len(state.plain_text_reminders) == 1


def test_guard_retry_then_normal_plain_text_falls_back_to_format_retry() -> None:
    """Guard 重试后返回普通纯文本：重新按当前响应判定为格式问题。"""
    state = _state()
    # 第 1 次：工具调用拒答
    _handle_guard_refusal(
        _tool_refusal_response(),
        1,
        state,
        _config(),
        _EVIDENCE,
        from_tool_call=True,
    )
    _consume_guard_reminder(state)

    # 第 2 次：普通纯文本角色回复（Guard PASS）
    retry = _handle_plain_text_violation(
        _normal_plain_response(), 1, state, _config(), [{}]
    )

    assert retry is True
    assert state.plain_text_retry_count == 1
    assert state.guard_retry_count == 1
    assert state.guard_reminder is None


def test_guard_retry_then_plain_refusal_stops_without_format_retries() -> None:
    """Guard 重试后再次纯文本拒答：直接收口，不落入三次格式重试。"""
    state = _state()
    config = _config(1)

    assert (
        _handle_guard_refusal(
            _tool_refusal_response(),
            1,
            state,
            config,
            _EVIDENCE,
            from_tool_call=True,
        )
        is True
    )
    _consume_guard_reminder(state)

    response = _plain_refusal_response()
    assert (
        _handle_guard_refusal(
            response, 1, state, config, _EVIDENCE, from_tool_call=False
        )
        is False
    )
    # 收口路径同样完成回滚，且未产生新的格式纠正提示
    assert response.message == ""
    assert response.call_list == []
    assert state.guard_reminder is None
    assert state.plain_text_reminders == []
    assert state.plain_text_retry_count == 0


def test_counter_isolation_between_two_retry_kinds() -> None:
    """两类重试的计数完全独立，互不挤占额度。"""
    state = _state()
    config = _config(5)

    # 1) 普通格式错误
    _handle_plain_text_violation(_normal_plain_response(), 1, state, config, [{}])
    assert state.plain_text_retry_count == 1
    assert state.guard_retry_count == 0

    # 2) 工具调用拒答
    _handle_guard_refusal(
        _tool_refusal_response(), 1, state, config, _EVIDENCE, from_tool_call=True
    )
    assert state.plain_text_retry_count == 1
    assert state.guard_retry_count == 1

    # 3) 又是普通格式错误
    _handle_plain_text_violation(_normal_plain_response(), 1, state, config, [{}])
    assert state.plain_text_retry_count == 2
    assert state.guard_retry_count == 1


def test_default_budget_allows_three_retries() -> None:
    """默认预算为 3 次 Guard retry，初始响应不计入其中。

    三次重试分别以工具调用形态与纯文本形态出现，证明两种拒答共用同一个
    计数器；每次命中都重新创建一条提示，消费后立即消失。
    """
    config = KFCConfig(general=KFCConfig.GeneralSection())
    assert config.general.guard_max_retries == 3

    state = _state()
    for expected, from_tool_call in enumerate((True, False, True), start=1):
        response = (
            _tool_refusal_response()
            if from_tool_call
            else _plain_refusal_response()
        )
        assert (
            _handle_guard_refusal(
                response,
                1,
                state,
                config,
                _EVIDENCE,
                from_tool_call=from_tool_call,
            )
            is True
        )
        assert state.guard_retry_count == expected
        assert len(_consume_guard_reminder(state)) == 1
        assert state.guard_reminder is None

    # 第 4 次命中时预算已耗尽：直接收口，不再产生提示
    response = _plain_refusal_response()
    assert (
        _handle_guard_refusal(
            response, 1, state, config, _EVIDENCE, from_tool_call=False
        )
        is False
    )
    assert state.guard_retry_count == 3
    assert state.guard_reminder is None
    assert response.message == ""


# ── 纯文本响应必须真的经过 Guard ────────────────────────────────────────


async def test_guard_receives_plain_text_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无工具调用时 Guard 仍需拿到正文与推理，否则纯文本拒答会漏判。"""
    from plugins.kokoro_flow_chatter.runtime import guard_hook

    service = _FakeGuardService()
    monkeypatch.setattr(guard_hook, "get_service_class", lambda signature: object)
    monkeypatch.setattr(guard_hook, "get_service", lambda signature: service)

    result = await check_response_guard(
        _plain_refusal_response(), request_name="kokoro_flow_chatter"
    )

    assert result.blocked is True
    assert service.calls[0]["reply_text"] == _PLAIN_REFUSAL
    assert service.calls[0]["tool_calls"] == ()
    assert service.calls[0]["reasoning_text"] == (
        "安全政策不允许此类内容，我必须拒绝这个请求。"
    )


# ── 持久链与快照的洁净性 ────────────────────────────────────────────────


def _persistent_chain_after_send(
    source: _FakeResponse, transient: list[LLMPayload], assistant: LLMPayload
) -> list[LLMPayload]:
    """模拟框架发送并回写后的持久主链。"""
    view = build_request_view(source, transient)
    sent = list(view.payloads) + [assistant]
    return _without_transient_payloads(
        sent,
        source_payloads=list(source.payloads),
        transient_count=len(view.payloads) - len(source.payloads),
    )


def test_request_view_sends_reminder_but_chain_keeps_it_out() -> None:
    """提示进入本次发送视图，但不会写回持久主链。"""
    source = _FakeResponse(payloads=[_user("用户真实消息")])
    reminder = LLMPayload(ROLE.USER, Text(_GUARD_RETRY_REMINDER))

    view = build_request_view(source, [reminder])
    assert len(view.payloads) == 2
    assert _GUARD_RETRY_REMINDER in _reminder_text(view.payloads[1])

    persistent = _persistent_chain_after_send(
        source, [reminder], _assistant("正常角色回复")
    )

    assert len(persistent) == 2
    assert persistent[-1].role == ROLE.ASSISTANT
    assert all(
        _GUARD_RETRY_REMINDER not in _reminder_text(payload)
        for payload in persistent
    )


def test_guard_reminder_absent_from_context_snapshot() -> None:
    """被拦截拒答与 Guard 提示都不会进入上下文快照。"""
    source = _FakeResponse(payloads=[_user("用户真实消息")])
    reminder = LLMPayload(ROLE.USER, Text(_GUARD_RETRY_REMINDER))

    persistent = _persistent_chain_after_send(
        source, [reminder], _assistant("正常角色回复")
    )
    snapshot = capture_snapshot(persistent, 20)

    assert snapshot is not None
    blob = json.dumps(snapshot, ensure_ascii=False, default=str)
    # 正向对照：正常内容确实会被序列化，断言才有意义
    assert "正常角色回复" in blob
    assert _GUARD_RETRY_REMINDER not in blob
    assert _PLAIN_REFUSAL not in blob


def test_rollback_leaves_nothing_for_mental_log() -> None:
    """回滚后 message 为空，提交阶段不可能把拒答写进 mental_log。"""
    response = _plain_refusal_response()
    _handle_guard_refusal(
        response,
        1,
        _state(),
        _config(),
        _EVIDENCE,
        from_tool_call=False,
    )

    assert response.message == ""
    assert response.reasoning_content == ""
    assert response.reasoning_parts == []
    assert response.call_list == []
    assert len(response.payloads) == 1
    assert _PLAIN_REFUSAL not in _reminder_text(response.payloads[0])


# ── 主循环：守卫预算耗尽必须显式收口 ────────────────────────────────────


class _ScriptExhausted(Exception):
    """测试脚本用尽，用于中断主循环驱动。"""


class _ChainResponse:
    """可追加 payload 的响应链替身。"""

    def __init__(self, payloads: list[LLMPayload]) -> None:
        """构造响应链。"""
        self.payloads = list(payloads)
        self.message = ""
        self.reasoning_content = ""
        self.reasoning_parts: list[Any] = []
        self.call_list: list[ToolCall] = []
        self.meta_data: dict[str, Any] = {}


class _HarnessSession:
    """记录会话副作用的替身。"""

    def __init__(self) -> None:
        """初始化记录容器。"""
        self.history_summary = ""
        self.context_snapshot: Any = None
        self.bot_planning: list[dict[str, Any]] = []
        self.user_id = "user-1"

    def append_context_entries(self, payloads: list[Any], max_payloads: int) -> bool:
        """不实际改动快照，避免测试依赖会话序列化。"""
        return False

    def add_bot_planning(self, **kwargs: Any) -> None:
        """记录一次 bot planning 写入。"""
        self.bot_planning.append(kwargs)


class _HarnessChatter:
    """execute_orchestrator 所需的最小 Chatter 替身。"""

    def __init__(self, config: KFCConfig, session: _HarnessSession) -> None:
        """初始化替身。"""
        self.stream_id = "guard-stream"
        self._config = config
        self._session = session

    def get_config(self) -> KFCConfig:
        """返回注入的配置。"""
        return self._config

    async def load_session(self) -> _HarnessSession:
        """返回注入的会话。"""
        return self._session

    async def save_session(self, session: Any) -> None:
        """记录保存动作。"""
        _ = session

    async def fetch_unreads(self, time_format: str = "") -> tuple[str, list[Any]]:
        """返回空未读快照。"""
        _ = time_format
        return "", []

    async def flush_unreads(self, unread_messages: list[Any]) -> int:
        """记录消费条数。"""
        return len(unread_messages)

    async def build_virtual_trigger_message(self) -> Any:
        """返回一个占位的触发消息。"""
        return SimpleNamespace(message_id="virtual-1", content="[虚拟触发]")

    async def send_reply(self, *args: Any, **kwargs: Any) -> None:
        """占位回复发送，仅供决策层取用引用。"""

    async def run_tool_call(self, *args: Any, **kwargs: Any) -> None:
        """占位工具执行，仅供决策层取用引用。"""


def _payloads_text(payloads: list[LLMPayload]) -> str:
    """拼接一组 payload 的全部文本。"""
    chunks: list[str] = []
    for payload in payloads:
        content = payload.content
        parts = content if isinstance(content, list) else [content]
        chunks.extend(part.text for part in parts if isinstance(part, Text))
    return "\n".join(chunks)


@pytest.mark.parametrize("max_retries", [1, 3])
async def test_guard_budget_exhausted_stops_before_decision(
    monkeypatch: pytest.MonkeyPatch, max_retries: int
) -> None:
    """守卫重试预算耗尽时必须显式结束本轮，不进入决策层。

    驱动真实 ``execute_orchestrator``：初始响应与每一次重试都被守卫判定
    为模型层拒答，第 ``max_retries + 1`` 次命中时直接收口。初始请求不计入
    重试次数，因此预算 3 对应四次模型生成。
    """
    from plugins.kokoro_flow_chatter.runtime import orchestrator as module

    config = KFCConfig(
        general=KFCConfig.GeneralSection(
            guard_enabled=True, guard_max_retries=max_retries
        )
    )
    session = _HarnessSession()
    chatter = _HarnessChatter(config, session)
    chain = _ChainResponse([_user("用户真实消息")])

    sent_views: list[list[LLMPayload]] = []
    guard_retries: list[int] = []
    decision_calls: list[Any] = []
    committed: list[Any] = []

    turn_inputs = [
        TurnInputResult(
            response=chain, persistent_user_payload=_user("用户真实消息")
        ),
        # 守卫重试：带 has_pending_tool_results 的续轮，无新用户输入，
        # 因此 guard_retry_count 不会被重置。
        *(
            TurnInputResult(response=chain, has_pending_tool_results=False)
            for _ in range(max_retries)
        ),
    ]

    async def _fake_prepare_turn_input(*args: Any, **kwargs: Any) -> Any:
        if not turn_inputs:
            raise _ScriptExhausted
        return turn_inputs.pop(0)

    async def _fake_send_llm_request(
        chatter_arg: Any,
        send_target: Any,
        config_arg: Any,
        known_ids: Any,
        state: Any,
    ) -> tuple[Any, list[Any]]:
        _ = (chatter_arg, config_arg, known_ids, state)
        sent_views.append(list(send_target.payloads))
        call = ToolCall(id="call-1", name="kfc_reply", args={"content": [_PLAIN_REFUSAL]})
        chain.payloads.append(LLMPayload(ROLE.ASSISTANT, [call]))
        chain.call_list = [call]
        chain.reasoning_content = "安全政策不允许此类内容，我必须拒绝这个请求。"
        chain.reasoning_parts = [SimpleNamespace(text="安全政策不允许此类内容")]
        return chain, []

    async def _fake_guard(
        response: Any, *, request_name: str, retry_index: int = 0
    ) -> GuardCheck:
        _ = (response, request_name)
        guard_retries.append(retry_index)
        return GuardCheck(blocked=True, evidence=_EVIDENCE)

    async def _fake_run_decision(*args: Any, **kwargs: Any) -> Any:
        decision_calls.append((args, kwargs))
        raise AssertionError("守卫预算耗尽后不得进入 run_decision")

    async def _fake_commit_turn_decision(*args: Any, **kwargs: Any) -> Any:
        committed.append((args, kwargs))
        return SimpleNamespace(
            next_signal=None,
            return_after_yield=False,
            has_pending_tool_results=False,
            is_final_timeout=False,
        )

    class _FakeSummary:
        def __init__(self, *args: Any) -> None:
            """忽略参数。"""

        def sync_if_changed(self, *args: Any) -> None:
            """测试中无需同步摘要。"""

    class _FakeTimeoutService:
        def __init__(self, *args: Any) -> None:
            """忽略参数。"""

    async def _fake_activate_stream(stream_id: str) -> Any:
        return SimpleNamespace(stream_id=stream_id, platform="qq")

    async def _fake_build_initial_request(*args: Any, **kwargs: Any) -> Any:
        return chain, {}

    monkeypatch.setattr(module, "activate_stream", _fake_activate_stream)
    monkeypatch.setattr(module, "resolve_model_set", lambda cfg: [{"m": 1}])
    monkeypatch.setattr(module, "build_initial_request", _fake_build_initial_request)
    monkeypatch.setattr(module, "SummarySynchronizer", _FakeSummary)
    monkeypatch.setattr(module, "TimeoutService", _FakeTimeoutService)
    monkeypatch.setattr(module, "prepare_turn_input", _fake_prepare_turn_input)
    monkeypatch.setattr(
        module, "build_last_mile_payload", lambda: _user("[last-mile]")
    )
    monkeypatch.setattr(module, "heal_orphan_tool_results", lambda *a, **k: None)
    monkeypatch.setattr(module, "_send_llm_request", _fake_send_llm_request)
    monkeypatch.setattr(module, "check_response_guard", _fake_guard)
    monkeypatch.setattr(module, "run_decision", _fake_run_decision)
    monkeypatch.setattr(module, "commit_turn_decision", _fake_commit_turn_decision)

    signals: list[Any] = []
    try:
        async for signal in module.execute_orchestrator(chatter):
            signals.append(signal)
    except _ScriptExhausted:
        pytest.fail("守卫收口后主循环仍在继续，未显式结束本轮")

    # 收口信号：Stop(0)，且只发出一次
    assert len(signals) == 1
    assert isinstance(signals[0], Stop)
    assert signals[0].time == 0

    # 未进入决策层，未提交任何 planning
    assert decision_calls == []
    assert committed == []
    assert session.bot_planning == []
    assert session.context_snapshot is None

    # 守卫在每次模型生成后各检查一次，重试序号从 0 递增到上限
    assert guard_retries == list(range(max_retries + 1))

    # 被拦响应已完整回滚
    assert chain.message == ""
    assert chain.reasoning_content == ""
    assert chain.reasoning_parts == []
    assert chain.call_list == []
    assert len(chain.payloads) == 1
    assert _PLAIN_REFUSAL not in _payloads_text(chain.payloads)

    # 提示只注入一次：首次请求无提示，每次重试恰好一条 Guard 提示
    assert len(sent_views) == max_retries + 1
    assert _GUARD_RETRY_REMINDER not in _payloads_text(sent_views[0])
    for view in sent_views[1:]:
        assert _payloads_text(view).count(_GUARD_RETRY_REMINDER) == 1
    # 收口时提示不残留，也不会误用纯文本格式提示
    assert _PLAIN_TEXT_RETRY_REMINDER not in _payloads_text(sent_views[-1])


async def test_guard_recovery_succeeds_before_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """重试期间恢复出正常回复时立即进入决策层，不再继续消耗预算。

    预算 3 下的脚本：初始与 Retry #1 均为 MODEL_REFUSAL，Retry #2 返回
    正常工具调用。此时应当正常决策并发送，而不是发出第 3 次重试。
    """
    from plugins.kokoro_flow_chatter.runtime import orchestrator as module

    config = KFCConfig(
        general=KFCConfig.GeneralSection(guard_enabled=True, guard_max_retries=3)
    )
    session = _HarnessSession()
    chatter = _HarnessChatter(config, session)
    chain = _ChainResponse([_user("用户真实消息")])

    sent_views: list[list[LLMPayload]] = []
    guard_retries: list[int] = []
    decisions: list[Any] = []
    committed: list[Any] = []

    turn_inputs = [
        TurnInputResult(
            response=chain, persistent_user_payload=_user("用户真实消息")
        ),
        TurnInputResult(response=chain, has_pending_tool_results=False),
        TurnInputResult(response=chain, has_pending_tool_results=False),
    ]

    async def _fake_prepare_turn_input(*args: Any, **kwargs: Any) -> Any:
        if not turn_inputs:
            raise _ScriptExhausted
        return turn_inputs.pop(0)

    def _refusal_call() -> ToolCall:
        return ToolCall(
            id="call-1", name="kfc_reply", args={"content": [_PLAIN_REFUSAL]}
        )

    def _normal_call() -> ToolCall:
        return ToolCall(
            id="call-2", name="kfc_reply", args={"content": [_NORMAL_REPLY]}
        )

    async def _fake_send_llm_request(
        chatter_arg: Any,
        send_target: Any,
        config_arg: Any,
        known_ids: Any,
        state: Any,
    ) -> tuple[Any, list[Any]]:
        _ = (chatter_arg, config_arg, known_ids, state)
        sent_views.append(list(send_target.payloads))
        if len(sent_views) <= 2:
            call = _refusal_call()
            chain.reasoning_content = "安全政策不允许此类内容，我必须拒绝这个请求。"
            chain.reasoning_parts = [SimpleNamespace(text="安全政策不允许此类内容")]
        else:
            call = _normal_call()
            chain.reasoning_content = ""
            chain.reasoning_parts = []
        chain.payloads.append(LLMPayload(ROLE.ASSISTANT, [call]))
        chain.call_list = [call]
        return chain, []

    async def _fake_guard(
        response: Any, *, request_name: str, retry_index: int = 0
    ) -> GuardCheck:
        _ = (response, request_name)
        guard_retries.append(retry_index)
        blocked = len(guard_retries) <= 2
        return GuardCheck(blocked=blocked, evidence=_EVIDENCE if blocked else ())

    async def _fake_run_decision(*args: Any, **kwargs: Any) -> Any:
        decisions.append((args, kwargs))
        return SimpleNamespace(proactive_schedule=None, has_failed_tool=False)

    async def _fake_commit_turn_decision(*args: Any, **kwargs: Any) -> Any:
        committed.append((args, kwargs))
        return SimpleNamespace(
            next_signal=None,
            return_after_yield=False,
            has_pending_tool_results=False,
            is_final_timeout=False,
        )

    class _FakeSummary:
        def __init__(self, *args: Any) -> None:
            """忽略参数。"""

        def sync_if_changed(self, *args: Any) -> None:
            """测试中无需同步摘要。"""

    class _FakeTimeoutService:
        def __init__(self, *args: Any) -> None:
            """忽略参数。"""

    async def _fake_activate_stream(stream_id: str) -> Any:
        return SimpleNamespace(stream_id=stream_id, platform="qq")

    async def _fake_build_initial_request(*args: Any, **kwargs: Any) -> Any:
        return chain, {}

    monkeypatch.setattr(module, "activate_stream", _fake_activate_stream)
    monkeypatch.setattr(module, "resolve_model_set", lambda cfg: [{"m": 1}])
    monkeypatch.setattr(module, "build_initial_request", _fake_build_initial_request)
    monkeypatch.setattr(module, "SummarySynchronizer", _FakeSummary)
    monkeypatch.setattr(module, "TimeoutService", _FakeTimeoutService)
    monkeypatch.setattr(module, "prepare_turn_input", _fake_prepare_turn_input)
    monkeypatch.setattr(
        module, "build_last_mile_payload", lambda: _user("[last-mile]")
    )
    monkeypatch.setattr(module, "heal_orphan_tool_results", lambda *a, **k: None)
    monkeypatch.setattr(module, "_send_llm_request", _fake_send_llm_request)
    monkeypatch.setattr(module, "check_response_guard", _fake_guard)
    monkeypatch.setattr(module, "run_decision", _fake_run_decision)
    monkeypatch.setattr(module, "commit_turn_decision", _fake_commit_turn_decision)

    signals: list[Any] = []
    try:
        async for signal in module.execute_orchestrator(chatter):
            signals.append(signal)
    except _ScriptExhausted:
        pass

    # 恢复成功：发出了 3 次请求（初始 + 2 次重试），第 3 次不再被拦截
    assert guard_retries == [0, 1, 2]
    assert len(sent_views) == 3
    assert len(decisions) == 1
    assert len(committed) == 1
    # 正常决策路径不向框架发出收口信号
    assert signals == []

    # 每次重试请求恰好带一条提示，恢复成功的请求同样只有一条
    assert _GUARD_RETRY_REMINDER not in _payloads_text(sent_views[0])
    assert _payloads_text(sent_views[1]).count(_GUARD_RETRY_REMINDER) == 1
    assert _payloads_text(sent_views[2]).count(_GUARD_RETRY_REMINDER) == 1

    # 提示不残留在响应链上，正常回复已进入主链
    assert _GUARD_RETRY_REMINDER not in _payloads_text(chain.payloads)
    assert _PLAIN_TEXT_RETRY_REMINDER not in _payloads_text(chain.payloads)
    reply_calls = [
        item
        for payload in chain.payloads
        for item in payload.content
        if isinstance(item, ToolCall)
    ]
    assert reply_calls[-1].args["content"] == [_NORMAL_REPLY]


async def test_new_user_message_resets_guard_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """重试过程中到达的新用户消息把守卫预算重置，不与上一轮共用。

    预算 3 下的脚本：新消息 A 触发 MODEL_REFUSAL 并重试；紧接着新消息 B
    到达，此时重试序号必须回到 0，否则上一轮的失败会持续挤压本轮额度。
    """
    from plugins.kokoro_flow_chatter.runtime import orchestrator as module

    config = KFCConfig(
        general=KFCConfig.GeneralSection(guard_enabled=True, guard_max_retries=3)
    )
    session = _HarnessSession()
    chatter = _HarnessChatter(config, session)
    chain = _ChainResponse([_user("用户真实消息 A")])

    sent_views: list[list[LLMPayload]] = []
    guard_retries: list[int] = []

    turn_inputs = [
        TurnInputResult(
            response=chain, persistent_user_payload=_user("用户真实消息 A")
        ),
        TurnInputResult(
            response=chain, persistent_user_payload=_user("用户真实消息 B")
        ),
    ]

    async def _fake_prepare_turn_input(*args: Any, **kwargs: Any) -> Any:
        if not turn_inputs:
            raise _ScriptExhausted
        return turn_inputs.pop(0)

    async def _fake_send_llm_request(
        chatter_arg: Any,
        send_target: Any,
        config_arg: Any,
        known_ids: Any,
        state: Any,
    ) -> tuple[Any, list[Any]]:
        _ = (chatter_arg, config_arg, known_ids, state)
        sent_views.append(list(send_target.payloads))
        call = ToolCall(
            id="call-1", name="kfc_reply", args={"content": [_PLAIN_REFUSAL]}
        )
        chain.payloads.append(LLMPayload(ROLE.ASSISTANT, [call]))
        chain.call_list = [call]
        chain.reasoning_content = "安全政策不允许此类内容，我必须拒绝这个请求。"
        chain.reasoning_parts = [SimpleNamespace(text="安全政策不允许此类内容")]
        return chain, []

    async def _fake_guard(
        response: Any, *, request_name: str, retry_index: int = 0
    ) -> GuardCheck:
        _ = (response, request_name)
        guard_retries.append(retry_index)
        return GuardCheck(blocked=True, evidence=_EVIDENCE)

    class _FakeSummary:
        def __init__(self, *args: Any) -> None:
            """忽略参数。"""

        def sync_if_changed(self, *args: Any) -> None:
            """测试中无需同步摘要。"""

    class _FakeTimeoutService:
        def __init__(self, *args: Any) -> None:
            """忽略参数。"""

    async def _fake_activate_stream(stream_id: str) -> Any:
        return SimpleNamespace(stream_id=stream_id, platform="qq")

    async def _fake_build_initial_request(*args: Any, **kwargs: Any) -> Any:
        return chain, {}

    monkeypatch.setattr(module, "activate_stream", _fake_activate_stream)
    monkeypatch.setattr(module, "resolve_model_set", lambda cfg: [{"m": 1}])
    monkeypatch.setattr(module, "build_initial_request", _fake_build_initial_request)
    monkeypatch.setattr(module, "SummarySynchronizer", _FakeSummary)
    monkeypatch.setattr(module, "TimeoutService", _FakeTimeoutService)
    monkeypatch.setattr(module, "prepare_turn_input", _fake_prepare_turn_input)
    monkeypatch.setattr(
        module, "build_last_mile_payload", lambda: _user("[last-mile]")
    )
    monkeypatch.setattr(module, "heal_orphan_tool_results", lambda *a, **k: None)
    monkeypatch.setattr(module, "_send_llm_request", _fake_send_llm_request)
    monkeypatch.setattr(module, "check_response_guard", _fake_guard)

    signals: list[Any] = []
    try:
        async for signal in module.execute_orchestrator(chatter):
            signals.append(signal)
    except _ScriptExhausted:
        pass

    # 第二次检查发生在新用户消息到来之后，序号必须归零
    assert guard_retries == [0, 0]
    assert len(sent_views) == 2
    # 两次请求各自最多一条提示，收口前不残留
    assert _GUARD_RETRY_REMINDER not in _payloads_text(sent_views[0])
    assert _payloads_text(sent_views[1]).count(_GUARD_RETRY_REMINDER) == 1
    assert signals == []

