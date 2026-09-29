"""模型层拒答的恢复流程测试。

覆盖两类失败的分流、逐层累积的 Guard Retry 提示，以及被拦截内容与
提示都不进入持久上下文的各种保证。

概念链路：

    响应 ──► Response Guard ──┬─ MODEL_REFUSAL ──► Guard recovery（L1..L3）
                              └─ PASS ──────────► 格式纠正 / 正常决策

Guard Retry 固定三层，提示不驻留在状态里：状态只保存 ``guard_retry_count``，
每次组装发送视图时按层级现场构造 ``L1..LN``。
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
    _GUARD_MAX_RETRIES,
    _GUARD_RETRY_REMINDERS,
    _PLAIN_TEXT_RETRY_REMINDER,
    _LoopState,
    _build_guard_retry_reminders,
    _guard_retry_payloads,
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
from src.kernel.llm import ReasoningText  # noqa: E402

_L1, _L2, _L3 = _GUARD_RETRY_REMINDERS
"""三层提示的具名引用，供断言顺序与内容。"""

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


def _config(**overrides: Any) -> KFCConfig:
    """构造只指定守卫开关的 KFC 配置。

    Guard Retry 层数不接受用户配置，因此这里没有对应入参。
    """
    settings: dict[str, Any] = {"guard_enabled": True}
    settings.update(overrides)
    return KFCConfig(general=KFCConfig.GeneralSection(**settings))


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


def _content_of(payload: LLMPayload) -> list[Any]:
    """取 payload 的内容列表。"""
    content = payload.content
    return content if isinstance(content, list) else [content]


def _reminder_text(payload: LLMPayload) -> str:
    """取出 payload 中的纯文本内容。"""
    return "".join(
        item.text for item in payload.content if isinstance(item, Text)
    )


def _layers_in(payloads: list[LLMPayload]) -> list[str]:
    """按出现顺序提取一组 payload 中的 Guard 层级文本。"""
    layers: list[str] = []
    for payload in payloads:
        text = _reminder_text(payload)
        if text in _GUARD_RETRY_REMINDERS:
            layers.append(text)
    return layers


def _payloads_text(payloads: list[LLMPayload]) -> str:
    """拼接一组 payload 的全部文本。"""
    chunks: list[str] = []
    for payload in payloads:
        chunks.extend(
            item.text for item in _content_of(payload) if isinstance(item, Text)
        )
    return "\n".join(chunks)


def _block(
    state: _LoopState, *, from_tool_call: bool = True, **kwargs: Any
) -> bool:
    """驱动一次守卫命中，返回是否进入重试。"""
    response = (
        _tool_refusal_response() if from_tool_call else _plain_refusal_response()
    )
    return _handle_guard_refusal(
        response,
        list(response.payloads[:1]),
        state,
        _EVIDENCE,
        from_tool_call=from_tool_call,
        **kwargs,
    )


# ── 固定层数与分层构造 ─────────────────────────────────────────────────


def test_retry_limit_is_derived_from_reminder_layers() -> None:
    """重试上限由提示层数推导，避免两者写成不同数字。"""
    assert len(_GUARD_RETRY_REMINDERS) == 3
    assert _GUARD_MAX_RETRIES == len(_GUARD_RETRY_REMINDERS)


def test_reminders_are_distinct_and_ordered() -> None:
    """三层文案互不相同，且第 3 层保留 do_nothing 收口出口。"""
    assert len(set(_GUARD_RETRY_REMINDERS)) == _GUARD_MAX_RETRIES
    assert "action-do_nothing" in _L3


@pytest.mark.parametrize(
    ("retry_count", "expected"),
    [
        (0, []),
        (1, [_L1]),
        (2, [_L1, _L2]),
        (3, [_L1, _L2, _L3]),
    ],
)
def test_build_reminders_by_retry_count(
    retry_count: int, expected: list[str]
) -> None:
    """按重试次数逐层切片，后面的层级包含前面的层级。"""
    payloads = _build_guard_retry_reminders(retry_count)
    assert _layers_in(payloads) == expected
    assert all(payload.role == ROLE.USER for payload in payloads)


def test_build_reminders_ignores_negative_count() -> None:
    """非法计数不产生提示。"""
    assert _build_guard_retry_reminders(-1) == []


# ── 逐层注入 ───────────────────────────────────────────────────────────


def test_first_retry_shows_only_l1() -> None:
    """第 1 次重试只带 L1。"""
    state = _state()
    assert _block(state) is True

    assert state.guard_retry_count == 1
    assert _layers_in(_build_guard_retry_reminders(state.guard_retry_count)) == [
        _L1
    ]


def test_second_retry_shows_l1_then_l2() -> None:
    """第 2 次重试带 L1 + L2，顺序固定。"""
    state = _state()
    _block(state)
    assert _block(state) is True

    assert state.guard_retry_count == 2
    assert _layers_in(_build_guard_retry_reminders(state.guard_retry_count)) == [
        _L1,
        _L2,
    ]


def test_third_retry_shows_all_three_layers() -> None:
    """第 3 次重试带 L1 + L2 + L3。"""
    state = _state()
    for _ in range(3):
        assert _block(state) is True

    assert state.guard_retry_count == 3
    assert _layers_in(_build_guard_retry_reminders(state.guard_retry_count)) == [
        _L1,
        _L2,
        _L3,
    ]


def test_mixed_refusal_shapes_share_one_counter() -> None:
    """工具调用拒答与纯文本拒答共用同一个计数器与同一套层级。"""
    state = _state()

    for index, from_tool_call in enumerate((True, False, True), start=1):
        assert _block(state, from_tool_call=from_tool_call) is True
        assert state.guard_retry_count == index

    # 第 4 次命中时固定预算已耗尽：收口
    assert _block(state, from_tool_call=False) is False
    assert state.guard_retry_count == _GUARD_MAX_RETRIES


def test_no_reminder_without_block() -> None:
    """未命中时不产生任何提示。"""
    state = _state()
    assert _build_guard_retry_reminders(state.guard_retry_count) == []


def test_reminders_are_rebuilt_not_accumulated_in_state() -> None:
    """提示不驻留状态：重复构造得到等值内容，状态里没有提示列表。"""
    state = _state()
    _block(state)

    first = _build_guard_retry_reminders(state.guard_retry_count)
    second = _build_guard_retry_reminders(state.guard_retry_count)
    assert _layers_in(first) == _layers_in(second) == [_L1]

    slots = getattr(type(state), "__slots__", ())
    assert "guard_reminders" not in slots


# ── 回合触发提示的重试沿用 ─────────────────────────────────────────────


def _trigger_payload() -> LLMPayload:
    """构造一条模拟超时提示的回合级触发 payload。"""
    return LLMPayload(ROLE.USER, Text("（超时提示：你等待的时间已到）"))


def test_retry_reuses_turn_trigger_payload() -> None:
    """重试轮沿用被打断回合的触发提示，且可反复取用。"""
    state = _state()
    trigger = _trigger_payload()

    assert _block(state, retry_payload=trigger) is True

    consumed = _guard_retry_payloads(state)
    assert len(consumed) == 1
    assert consumed[0] is trigger
    # 回合级上下文：同一回合的后续重试仍需取到同一条
    assert _guard_retry_payloads(state) == [trigger]


def test_retry_without_trigger_payload_stays_empty() -> None:
    """新消息回合不带触发提示，重试轮也不会凭空造一条。"""
    state = _state()
    _block(state)
    assert _guard_retry_payloads(state) == []


def test_later_retry_keeps_earlier_trigger_payload() -> None:
    """重试轮自身不带触发提示，不得把上一轮暂存的冲掉。"""
    state = _state()
    trigger = _trigger_payload()

    _block(state, retry_payload=trigger)
    # 第二轮重试：仍然是同一回合，因此没有新的触发提示
    _block(state)

    assert _guard_retry_payloads(state) == [trigger]


def test_first_request_of_new_turn_replaces_trigger_payload() -> None:
    """新回合的首次请求会以本回合取值覆写上一回合的残留。"""
    state = _state()
    state.guard_retry_payload = _trigger_payload()

    _block(state, retry_payload=None)

    assert state.guard_retry_payload is None
    assert _guard_retry_payloads(state) == []


def test_trigger_payload_cleared_on_exhaustion() -> None:
    """预算耗尽收口时清除暂存的触发提示，不污染下一回合。"""
    state = _state()

    for _ in range(_GUARD_MAX_RETRIES):
        _block(state, retry_payload=_trigger_payload())
    assert _block(state) is False

    assert _guard_retry_payloads(state) == []
    assert state.guard_retry_payload is None


# ── 两类失败的分流 ──────────────────────────────────────────────────────


def test_plain_text_refusal_uses_guard_retry() -> None:
    """纯文本安全拒答走 Guard 重试，不消耗纯文本重试额度。"""
    state = _state()
    retry = _block(state, from_tool_call=False)

    assert retry is True
    assert state.guard_retry_count == 1
    assert state.plain_text_retry_count == 0
    assert state.plain_text_reminders == []


def test_plain_text_pass_uses_plain_text_retry() -> None:
    """守卫放行的普通纯文本走原有格式纠正，不消耗 Guard 额度。"""
    state = _state()
    retry = _handle_plain_text_violation(
        _normal_plain_response(), [_user("[新消息]")], state, _config(), [{}]
    )

    assert retry is True
    assert state.plain_text_retry_count == 1
    assert state.guard_retry_count == 0
    assert _build_guard_retry_reminders(state.guard_retry_count) == []
    assert len(state.plain_text_reminders) == 1
    assert _PLAIN_TEXT_RETRY_REMINDER in _reminder_text(
        state.plain_text_reminders[0]
    )


def test_guard_path_clears_plain_text_reminders() -> None:
    """走 Guard 重试时清掉格式纠错提示，避免两种提示同时注入。"""
    state = _state()
    state.plain_text_reminders.append(LLMPayload(ROLE.USER, Text("旧提醒")))

    _block(state, from_tool_call=False)

    assert state.plain_text_reminders == []


def test_plain_text_path_drops_guard_layers() -> None:
    """转为格式纠正时清零层数，L1/L2 不会泄漏进格式重试。"""
    state = _state()
    _block(state)
    _block(state)
    assert state.guard_retry_count == 2

    _handle_plain_text_violation(
        _normal_plain_response(), [_user("[新消息]")], state, _config(), [{}]
    )

    assert state.guard_retry_count == 0
    assert _build_guard_retry_reminders(state.guard_retry_count) == []
    assert len(state.plain_text_reminders) == 1


def test_guard_retry_then_normal_plain_text_falls_back_to_format_retry() -> None:
    """Guard 重试后返回普通纯文本：重新按当前响应判定为格式问题。"""
    state = _state()
    _block(state)

    retry = _handle_plain_text_violation(
        _normal_plain_response(), [_user("[新消息]")], state, _config(), [{}]
    )

    assert retry is True
    assert state.plain_text_retry_count == 1
    # 守卫链已结束：层数清零，格式纠正只带纯文本提示
    assert state.guard_retry_count == 0
    assert _layers_in(_build_guard_retry_reminders(state.guard_retry_count)) == []


def test_guard_retry_then_plain_refusal_exhausts_fixed_budget() -> None:
    """固定三次后再次拒答：收口，且不产生格式纠正提示。"""
    state = _state()

    for _ in range(_GUARD_MAX_RETRIES):
        assert _block(state, from_tool_call=True) is True

    response = _plain_refusal_response()
    assert (
        _handle_guard_refusal(
            response, list(response.payloads[:1]), state, _EVIDENCE, from_tool_call=False
        )
        is False
    )
    # 收口路径同样完成回滚，且未产生格式纠正提示
    assert response.message == ""
    assert response.call_list == []
    assert state.plain_text_reminders == []
    assert state.plain_text_retry_count == 0


def test_counter_isolation_between_two_retry_kinds() -> None:
    """两类重试的计数完全独立，互不挤占额度。"""
    state = _state()

    # 1) 普通格式错误
    _handle_plain_text_violation(
        _normal_plain_response(), [_user("[新消息]")], state, _config(), [{}]
    )
    assert state.plain_text_retry_count == 1
    assert state.guard_retry_count == 0

    # 2) 工具调用拒答：只清格式提示，不动格式计数
    _block(state)
    assert state.plain_text_retry_count == 1
    assert state.plain_text_reminders == []
    assert state.guard_retry_count == 1

    # 3) 又是普通格式错误：格式计数继续累加，守卫层数归零
    _handle_plain_text_violation(
        _normal_plain_response(), [_user("[新消息]")], state, _config(), [{}]
    )
    assert state.plain_text_retry_count == 2
    assert state.guard_retry_count == 0


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
        transient_payloads=transient,
    )


def test_request_view_sends_layers_but_chain_keeps_them_out() -> None:
    """三层提示都进入本次发送视图，但不写回持久主链。"""
    source = _FakeResponse(payloads=[_user("用户真实消息")])
    transient = _build_guard_retry_reminders(_GUARD_MAX_RETRIES)

    view = build_request_view(source, transient)
    assert len(view.payloads) == 1 + _GUARD_MAX_RETRIES
    assert _layers_in(view.payloads) == [_L1, _L2, _L3]

    persistent = _persistent_chain_after_send(
        source, transient, _assistant("正常角色回复")
    )

    assert len(persistent) == 2
    assert persistent[-1].role == ROLE.ASSISTANT
    assert _layers_in(persistent) == []


def test_guard_rollback_discards_transient_turn_boundary() -> None:
    """拒绝续轮回复时，连同该回复产生的中性回合边界一起回滚。"""
    source = _FakeResponse(payloads=[_user("原始消息"), _assistant("已有回复")])
    before_send = list(source.payloads)
    persisted = _persistent_chain_after_send(
        source, [_user("仅本轮可见的触发提示")], _assistant(_PLAIN_REFUSAL)
    )
    response = _FakeResponse(payloads=persisted, message=_PLAIN_REFUSAL)

    assert _handle_guard_refusal(
        response, before_send, _state(), _EVIDENCE, from_tool_call=False
    )
    assert response.payloads == source.payloads
    assert response.message == ""


def test_guard_retry_after_assistant_keeps_only_successful_reply() -> None:
    """仅临时触发的续轮经两次拒绝后保留原历史与最终有效回复。"""
    from src.kernel.llm.context_structure import validate_payload_sequence

    source = _FakeResponse(payloads=[_user("原始消息"), _assistant("已有回复")])
    initial_payloads = list(source.payloads)
    trigger = _user("仅本轮可见的触发提示")
    state = _state()

    for retry_index in range(2):
        before_send = list(source.payloads)
        source.payloads = _persistent_chain_after_send(
            source,
            [trigger, *_build_guard_retry_reminders(retry_index)],
            _assistant(_PLAIN_REFUSAL),
        )
        assert _handle_guard_refusal(
            source, before_send, state, _EVIDENCE, from_tool_call=False
        )
        assert source.payloads == initial_payloads

    persistent = _persistent_chain_after_send(
        source, [trigger, *_build_guard_retry_reminders(2)], _assistant(_NORMAL_REPLY)
    )
    validate_payload_sequence(persistent, allow_incomplete_tail=False)
    assert persistent[0:2] == initial_payloads
    assert persistent[-1].content == [Text(_NORMAL_REPLY)]
    assert _PLAIN_REFUSAL not in _payloads_text(persistent)
    assert _layers_in(persistent) == []
    assert "仅本轮可见的触发提示" not in _payloads_text(persistent)


def test_all_layers_absent_from_context_snapshot() -> None:
    """被拦截拒答与三层提示都不会进入上下文快照。"""
    source = _FakeResponse(payloads=[_user("用户真实消息")])
    transient = _build_guard_retry_reminders(_GUARD_MAX_RETRIES)

    persistent = _persistent_chain_after_send(
        source, transient, _assistant("正常角色回复")
    )
    snapshot = capture_snapshot(persistent)

    assert snapshot is not None
    blob = json.dumps(snapshot, ensure_ascii=False, default=str)
    # 正向对照：正常内容确实会被序列化，断言才有意义
    assert "正常角色回复" in blob
    for layer in _GUARD_RETRY_REMINDERS:
        assert layer not in blob
    assert _PLAIN_REFUSAL not in blob


def test_rollback_leaves_nothing_for_mental_log() -> None:
    """回滚后 message 为空，提交阶段不可能把拒答写进 mental_log。"""
    response = _plain_refusal_response()
    _handle_guard_refusal(
        response, list(response.payloads[:1]), _state(), _EVIDENCE, from_tool_call=False
    )

    assert response.message == ""
    assert response.reasoning_content == ""
    assert response.reasoning_parts == []
    assert response.call_list == []
    assert len(response.payloads) == 1
    assert _PLAIN_REFUSAL not in _reminder_text(response.payloads[0])


# ── 主循环驱动 ──────────────────────────────────────────────────────────


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
        self.sealed_segments: list[Any] = []
        self.bot_planning: list[dict[str, Any]] = []
        self.user_id = "user-1"

    def append_context_entries(self, payloads: list[Any]) -> bool:
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
        self.session_store = object()

    def get_config(self) -> KFCConfig:
        """返回注入的配置。"""
        return self._config

    async def load_session(self) -> _HarnessSession:
        """返回注入的会话。"""
        return self._session

    async def save_session(self, session: Any) -> None:
        """记录保存动作。"""
        _ = session

    async def fetch_unreads(
        self, time_format: str = ""
    ) -> tuple[str, list[Any]]:
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


class _OrchestratorHarness:
    """为一条 ``execute_orchestrator`` 脚本装配替身与传感器。"""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        turn_inputs: list[TurnInputResult],
        *,
        blocked: int | None = None,
    ) -> None:
        """记录 monkeypatch、脚本与被拦次数。"""
        self.monkeypatch = monkeypatch
        self.turn_inputs = turn_inputs
        self.blocked = blocked
        self.sent_views: list[list[LLMPayload]] = []
        self.guard_retries: list[int] = []
        self.decisions: list[Any] = []
        self.committed: list[Any] = []
        self.chain = _ChainResponse([_user("用户真实消息")])
        self.session = _HarnessSession()
        self.signals: list[Any] = []

    def response_for_view(self, index: int) -> Any:
        """返回第 ``index`` 次请求应产出的响应。"""
        if self.blocked is None or index <= self.blocked:
            call = ToolCall(
                id="call-1", name="kfc_reply", args={"content": [_PLAIN_REFUSAL]}
            )
            self.chain.reasoning_content = (
                "安全政策不允许此类内容，我必须拒绝这个请求。"
            )
            self.chain.reasoning_parts = [
                SimpleNamespace(text="安全政策不允许此类内容")
            ]
        else:
            call = ToolCall(
                id="call-2", name="kfc_reply", args={"content": [_NORMAL_REPLY]}
            )
            self.chain.reasoning_content = ""
            self.chain.reasoning_parts = []
        self.chain.payloads.append(LLMPayload(ROLE.ASSISTANT, [call]))
        self.chain.call_list = [call]
        return self.chain

    def install(self, module: Any) -> None:
        """把替身接到 orchestrator 模块上。"""
        monkeypatch = self.monkeypatch
        chain = self.chain
        harness = self

        async def _prepare_turn_input(*args: Any, **kwargs: Any) -> Any:
            if not harness.turn_inputs:
                raise _ScriptExhausted
            return harness.turn_inputs.pop(0)

        async def _send_llm_request(
            chatter_arg: Any,
            send_target: Any,
            config_arg: Any,
            known_ids: Any,
            state: Any,
        ) -> tuple[Any, list[Any]]:
            _ = (chatter_arg, config_arg, known_ids, state)
            harness.sent_views.append(list(send_target.payloads))
            return harness.response_for_view(len(harness.sent_views)), []

        async def _guard(
            response: Any, *, request_name: str, retry_index: int = 0
        ) -> GuardCheck:
            _ = (response, request_name)
            harness.guard_retries.append(retry_index)
            blocked = harness.blocked is None or len(harness.guard_retries) <= (
                harness.blocked
            )
            return GuardCheck(
                blocked=blocked, evidence=_EVIDENCE if blocked else ()
            )

        async def _run_decision(*args: Any, **kwargs: Any) -> Any:
            harness.decisions.append((args, kwargs))
            return SimpleNamespace(proactive_schedule=None, has_failed_tool=False)

        async def _commit_turn_decision(*args: Any, **kwargs: Any) -> Any:
            harness.committed.append((args, kwargs))
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
        monkeypatch.setattr(
            module, "build_initial_request", _fake_build_initial_request
        )
        monkeypatch.setattr(module, "SummarySynchronizer", _FakeSummary)
        monkeypatch.setattr(
            module.SummaryService, "maybe_schedule_compression", lambda *a, **k: False
        )
        monkeypatch.setattr(module, "TimeoutService", _FakeTimeoutService)
        async def _no_rotation(_chatter: Any, response: Any, *args: Any, **kwargs: Any) -> tuple[Any, None, bool]:
            return response, None, False

        monkeypatch.setattr(module, "rotate_if_needed", _no_rotation)
        monkeypatch.setattr(module, "prepare_turn_input", _prepare_turn_input)
        monkeypatch.setattr(
            module, "build_last_mile_payload", lambda: _user("[last-mile]")
        )
        monkeypatch.setattr(
            module, "heal_orphan_tool_results", lambda *a, **k: None
        )
        monkeypatch.setattr(module, "_send_llm_request", _send_llm_request)
        monkeypatch.setattr(module, "check_response_guard", _guard)
        monkeypatch.setattr(module, "run_decision", _run_decision)
        monkeypatch.setattr(
            module, "commit_turn_decision", _commit_turn_decision
        )

    async def drive(self, module: Any, config: KFCConfig) -> None:
        """驱动主循环并收集发出的信号。"""
        chatter = _HarnessChatter(config, self.session)
        try:
            async for signal in module.execute_orchestrator(chatter):
                self.signals.append(signal)
        except _ScriptExhausted:
            pass


def _retry_turn_inputs(count: int, chain: Any) -> list[TurnInputResult]:
    """构造「首次请求 + ``count`` 次重试」的脚本。"""
    items = [
        TurnInputResult(response=chain, persistent_user_payload=_user("用户真实消息"))
    ]
    items.extend(
        TurnInputResult(response=chain, has_pending_tool_results=False)
        for _ in range(count)
    )
    return items


async def test_fixed_three_retries_then_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """固定三次重试后仍拒答：完整回滚、显式收口，不进入决策层。

    初始响应与每一次重试都被判为模型层拒答，第 4 次命中时直接收口。
    初始请求不计入重试次数，因此共四次模型生成。
    """
    from plugins.kokoro_flow_chatter.runtime import orchestrator as module

    harness = _OrchestratorHarness(
        monkeypatch,
        _retry_turn_inputs(_GUARD_MAX_RETRIES, None),
        blocked=None,
    )
    for item in harness.turn_inputs:
        item.response = harness.chain
    harness.install(module)

    try:
        await harness.drive(module, _config())
    except _ScriptExhausted:
        pytest.fail("守卫收口后主循环仍在继续，未显式结束本轮")

    # 收口信号：Stop(0)，且只发出一次
    assert len(harness.signals) == 1
    assert isinstance(harness.signals[0], Stop)
    assert harness.signals[0].time == 0

    # 未进入决策层，未提交任何 planning / 快照
    assert harness.decisions == []
    assert harness.committed == []
    assert harness.session.bot_planning == []
    assert harness.session.context_snapshot is None

    # 守卫在每次模型生成后各检查一次，序号从 0 递增到固定上限
    assert harness.guard_retries == list(range(_GUARD_MAX_RETRIES + 1))

    # 被拦响应已完整回滚
    assert harness.chain.message == ""
    assert harness.chain.reasoning_content == ""
    assert harness.chain.reasoning_parts == []
    assert harness.chain.call_list == []
    assert len(harness.chain.payloads) == 1
    assert _PLAIN_REFUSAL not in _payloads_text(harness.chain.payloads)

    # 逐层累积：第 N 次重试的发送视图带 L1..LN，顺序固定
    assert len(harness.sent_views) == _GUARD_MAX_RETRIES + 1
    assert _layers_in(harness.sent_views[0]) == []
    for index, view in enumerate(harness.sent_views[1:], start=1):
        assert _layers_in(view) == list(_GUARD_RETRY_REMINDERS[:index])

    # 收口时不会误用纯文本格式提示
    assert _PLAIN_TEXT_RETRY_REMINDER not in _payloads_text(
        harness.sent_views[-1]
    )


async def test_guard_retry_removes_refusal_after_send_trims_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """发送时裁剪旧回合后，重试请求仍不能携带被拦回复。"""
    from plugins.kokoro_flow_chatter.runtime import orchestrator as module

    harness = _OrchestratorHarness(
        monkeypatch, _retry_turn_inputs(1, None), blocked=1
    )
    harness.chain.payloads = [
        _user("旧消息"), _assistant("旧回复"), _user("用户真实消息")
    ]
    for item in harness.turn_inputs:
        item.response = harness.chain
    harness.install(module)

    async def _send_with_trim(
        _chatter: Any,
        send_target: Any,
        _config: Any,
        _known_ids: Any,
        _state: Any,
    ) -> tuple[Any, list[Any]]:
        harness.sent_views.append(list(send_target.payloads))
        if len(harness.sent_views) == 1:
            harness.chain.payloads = [harness.chain.payloads[-1]]
        result = harness.response_for_view(len(harness.sent_views))
        if len(harness.sent_views) == 1:
            result.payloads[-1].content.insert(0, ReasoningText("被拦推理"))
        return result, []

    monkeypatch.setattr(module, "_send_llm_request", _send_with_trim)
    await harness.drive(module, _config())

    assert len(harness.sent_views) == 2
    assert _L1 in _payloads_text(harness.sent_views[1])
    assert _PLAIN_REFUSAL not in repr(harness.sent_views[1])
    assert "被拦推理" not in repr(harness.sent_views[1])
    assert _PLAIN_REFUSAL not in repr(harness.chain.payloads)


async def test_retry_succeeds_before_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """第 2 次重试恢复出正常回复：立即决策，不再发第 3 次重试。

    脚本：初始与 Retry #1 被判拒答，Retry #2 返回正常工具调用。
    """
    from plugins.kokoro_flow_chatter.runtime import orchestrator as module

    harness = _OrchestratorHarness(
        monkeypatch, _retry_turn_inputs(2, None), blocked=2
    )
    for item in harness.turn_inputs:
        item.response = harness.chain
    harness.install(module)

    await harness.drive(module, _config())

    # 恢复成功：三次请求（初始 + 2 次重试），第 3 次不再被拦截
    assert harness.guard_retries == [0, 1, 2]
    assert len(harness.sent_views) == 3
    assert len(harness.decisions) == 1
    assert len(harness.committed) == 1
    assert harness.signals == []

    # 层级按序推进，恢复成功那次带 L1 + L2；第 3 层未出现
    assert _layers_in(harness.sent_views[0]) == []
    assert _layers_in(harness.sent_views[1]) == [_L1]
    assert _layers_in(harness.sent_views[2]) == [_L1, _L2]

    # 三层提示都不残留在响应链上，正常回复已进入主链
    persistent_text = _payloads_text(harness.chain.payloads)
    for layer in _GUARD_RETRY_REMINDERS:
        assert layer not in persistent_text
    assert _PLAIN_TEXT_RETRY_REMINDER not in persistent_text

    reply_calls = [
        item
        for payload in harness.chain.payloads
        for item in _content_of(payload)
        if isinstance(item, ToolCall)
    ]
    assert reply_calls[-1].args["content"] == [_NORMAL_REPLY]


async def test_new_user_message_resets_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """新用户消息开启新回合：层数归零，上一轮的提示不泄漏。"""
    from plugins.kokoro_flow_chatter.runtime import orchestrator as module

    harness = _OrchestratorHarness(
        monkeypatch,
        [
            TurnInputResult(
                response=None, persistent_user_payload=_user("用户真实消息 A")
            ),
            TurnInputResult(
                response=None, persistent_user_payload=_user("用户真实消息 B")
            ),
        ],
        blocked=None,
    )
    for item in harness.turn_inputs:
        item.response = harness.chain
    harness.install(module)

    await harness.drive(module, _config())

    # 第二次检查发生在新用户消息之后，序号归零
    assert harness.guard_retries == [0, 0]
    assert len(harness.sent_views) == 2
    # 两次都是各回合的首次请求，都不带任何层级
    assert _layers_in(harness.sent_views[0]) == []
    assert _layers_in(harness.sent_views[1]) == []
    assert harness.signals == []


async def test_retry_request_carries_turn_trigger_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """超时回合被拦后，每次重试请求都带上该回合的触发提示。

    超时提示描述的是「本轮为什么开口」。首次请求带上是既有行为；关键在于
    重试轮走工具续轮路径、自身不含任何触发提示，必须沿用过去，否则模型会
    以一次新规划的姿态重新决定动作与话题方向。
    """
    from plugins.kokoro_flow_chatter.runtime import orchestrator as module

    timeout_prompt = "（超时提示：等待时间已到，请决定继续等待或主动开口。）"
    harness = _OrchestratorHarness(
        monkeypatch,
        [
            # 第 1 轮：超时回合，只带仅当前请求可见的触发提示
            TurnInputResult(
                response=None, request_only_payload=_user(timeout_prompt)
            ),
            # 后两轮：守卫重试，走工具续轮路径，本身没有任何触发提示
            TurnInputResult(response=None, has_pending_tool_results=False),
            TurnInputResult(response=None, has_pending_tool_results=False),
        ],
        blocked=2,
    )
    for item in harness.turn_inputs:
        item.response = harness.chain
    harness.install(module)

    await harness.drive(module, _config())

    assert harness.guard_retries == [0, 1, 2]
    assert len(harness.sent_views) == 3
    for index, view in enumerate(harness.sent_views):
        assert timeout_prompt in _payloads_text(view), (
            f"第 {index + 1} 次请求缺少回合触发提示"
        )

    # 层级随重试推进
    assert _layers_in(harness.sent_views[1]) == [_L1]
    assert _layers_in(harness.sent_views[2]) == [_L1, _L2]

    # 触发提示与层级都不写回持久主链
    persistent_text = _payloads_text(harness.chain.payloads)
    assert timeout_prompt not in persistent_text
    for layer in _GUARD_RETRY_REMINDERS:
        assert layer not in persistent_text
