"""KFC 响应守卫接入层与命中处理的回归测试。

覆盖三件事：``response_guard`` 缺失或异常时主循环不受影响；命中时本轮
输出被完整回滚；重试预算独立且最多重试一次。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plugins.kokoro_flow_chatter.config import KFCConfig  # noqa: E402
from plugins.kokoro_flow_chatter.runtime.guard_hook import (  # noqa: E402
    GUARD_SIGNATURE,
    check_response_guard,
    guard_available,
)
from plugins.kokoro_flow_chatter.runtime.orchestrator import (  # noqa: E402
    _LoopState,
    _handle_guard_refusal,
)
from src.app.plugin_system.types import (  # noqa: E402
    LLMPayload,
    ROLE,
    Text,
    ToolCall,
)

_REFUSAL = "抱歉，我无法提供这类内容。"
_EVIDENCE = ("reasoning_explicit_safety_refusal", "reply_platform_refusal")


class _FakeResponse:
    """最小化模拟带 payloads/message/call_list 的响应对象。"""

    def __init__(
        self,
        payloads: list[LLMPayload],
        message: str = "",
        call_list: list[ToolCall] | None = None,
    ) -> None:
        """构造响应替身。"""
        self.payloads = payloads
        self.message = message
        self.reasoning_content = "安全政策不允许此类内容，我必须拒绝。"
        self.reasoning_parts: list[Any] = []
        self.call_list: list[ToolCall] = call_list or []


class _FakeGuardService:
    """response_guard 服务替身。"""

    def __init__(self, *, blocks: bool = False, error: Exception | None = None) -> None:
        """构造服务替身。"""
        self.blocks = blocks
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def inspect(self, **kwargs: Any) -> Any:
        """记录调用并返回预设结果。"""
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return type(
            "_Result",
            (),
            {"blocks": self.blocks, "evidence": _EVIDENCE if self.blocks else ()},
        )()


def _response(*, with_call: bool = True) -> _FakeResponse:
    """构造一条已追加本轮 ASSISTANT 输出的响应链。"""
    payload = LLMPayload(ROLE.USER, [Text("[新消息]")])
    if with_call:
        assistant = LLMPayload(
            ROLE.ASSISTANT,
            [Text(_REFUSAL), ToolCall(id="call-1", name="kfc_reply", args={})],
        )
    else:
        assistant = LLMPayload(ROLE.ASSISTANT, [Text(_REFUSAL)])
    return _FakeResponse(
        payloads=[payload, assistant],
        message=_REFUSAL,
        call_list=[ToolCall(id="call-1", name="kfc_reply", args={})] if with_call else [],
    )


def _state() -> _LoopState:
    """构造一个未绑定摘要同步器的循环状态。"""
    return _LoopState(summary=cast(Any, None))


def _config(max_retries: int = 1) -> KFCConfig:
    """构造只指定守卫重试上限的 KFC 配置。"""
    return KFCConfig(
        general=KFCConfig.GeneralSection(guard_max_retries=max_retries)
    )


# ── 接入层 ──────────────────────────────────────────────────────────────


def test_guard_unavailable_when_signature_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """response_guard 未注册时视为不可用。"""
    module = _guard_module()
    monkeypatch.setattr(module, "get_service_class", lambda signature: None)
    assert guard_available() is False


def test_guard_available_when_signature_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """response_guard 已注册时视为可用。"""
    module = _guard_module()
    captured: list[str] = []

    def _lookup(signature: str) -> type:
        captured.append(signature)
        return object

    monkeypatch.setattr(module, "get_service_class", _lookup)
    assert guard_available() is True
    assert captured == [GUARD_SIGNATURE]


async def test_check_skips_when_guard_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """未安装 response_guard 时不得命中，也不得查询实例。"""
    module = _guard_module()
    monkeypatch.setattr(module, "get_service_class", lambda signature: None)

    def _unexpected(signature: str) -> Any:
        raise AssertionError("服务缺失时不应查询实例")

    monkeypatch.setattr(module, "get_service", _unexpected)
    result = await check_response_guard(_response(), request_name="kokoro_flow_chatter")
    assert result.blocked is False
    assert result.evidence == ()


async def test_check_passes_when_service_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """服务实例创建失败时放行。"""
    module = _guard_module()
    monkeypatch.setattr(module, "get_service_class", lambda signature: object)
    monkeypatch.setattr(module, "get_service", lambda signature: None)
    result = await check_response_guard(_response(), request_name="kokoro_flow_chatter")
    assert result.blocked is False


async def test_check_reports_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """守卫判定放行时返回未命中。"""
    service = _FakeGuardService()
    _install_service(monkeypatch, service)
    result = await check_response_guard(_response(), request_name="kokoro_flow_chatter")
    assert result.blocked is False
    assert service.calls[0]["request_name"] == "kokoro_flow_chatter"
    assert service.calls[0]["reply_text"] == _REFUSAL
    assert service.calls[0]["reasoning_text"] == "安全政策不允许此类内容，我必须拒绝。"
    assert len(service.calls[0]["tool_calls"]) == 1


async def test_check_reports_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """守卫判定命中时返回命中及证据。"""
    service = _FakeGuardService(blocks=True)
    _install_service(monkeypatch, service)
    result = await check_response_guard(_response(), request_name="kokoro_flow_chatter")
    assert result.blocked is True
    assert result.evidence == _EVIDENCE


async def test_check_passes_on_guard_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """守卫自身异常不得打断主循环。"""
    service = _FakeGuardService(error=RuntimeError("guard exploded"))
    _install_service(monkeypatch, service)
    result = await check_response_guard(_response(), request_name="kokoro_flow_chatter")
    assert result.blocked is False
    assert result.evidence == ()


# ── 命中处理 ────────────────────────────────────────────────────────────


def test_first_block_rolls_back_and_retries() -> None:
    """首次命中：回滚到基线、清空输出字段并请求重试。"""
    response = _response()
    state = _state()

    retry = _handle_guard_refusal(
        response, 1, state, _config(), _EVIDENCE, from_tool_call=True
    )

    assert retry is True
    assert state.guard_retry_count == 1
    assert state.has_pending_tool_results is True
    assert len(response.payloads) == 1
    assert response.payloads[-1].role == ROLE.USER
    assert response.message == ""
    assert response.reasoning_content == ""
    assert response.call_list == []


def test_second_block_stops_retrying() -> None:
    """第二次命中：不再重试，计数不再增长。"""
    response = _response()
    state = _state()

    assert (
        _handle_guard_refusal(
            response, 1, state, _config(), _EVIDENCE, from_tool_call=True
        )
        is True
    )
    response = _response()

    assert (
        _handle_guard_refusal(
            response, 1, state, _config(), _EVIDENCE, from_tool_call=True
        )
        is False
    )
    assert state.guard_retry_count == 1
    assert len(response.payloads) == 1
    assert response.call_list == []


def test_zero_budget_stops_immediately() -> None:
    """重试上限为 0 时命中即刻收口。"""
    response = _response()
    state = _state()

    assert (
        _handle_guard_refusal(
            response, 1, state, _config(0), _EVIDENCE, from_tool_call=True
        )
        is False
    )
    assert state.guard_retry_count == 0
    assert response.message == ""
    assert response.call_list == []


def test_block_clears_pending_plain_text_reminders() -> None:
    """命中时一并清掉纯文本纠正提示，避免两种提示同时注入。"""
    response = _response()
    state = _state()
    state.plain_text_reminders.append(LLMPayload(ROLE.USER, Text("提醒")))

    _handle_guard_refusal(
        response, 1, state, _config(), _EVIDENCE, from_tool_call=True
    )

    assert state.plain_text_reminders == []


def test_guard_retry_is_independent_from_plain_text_retry() -> None:
    """守卫重试与纯文本重试使用独立计数。"""
    response = _response()
    state = _state()
    state.plain_text_retry_count = 3

    _handle_guard_refusal(
        response, 1, state, _config(), _EVIDENCE, from_tool_call=True
    )

    assert state.guard_retry_count == 1
    assert state.plain_text_retry_count == 3


def _guard_module() -> Any:
    """返回 KFC 守卫接入层模块。"""
    from plugins.kokoro_flow_chatter.runtime import guard_hook

    return guard_hook


def _install_service(
    monkeypatch: pytest.MonkeyPatch, service: _FakeGuardService
) -> None:
    """把服务替身注入接入层的 Service 查询。"""
    module = _guard_module()
    monkeypatch.setattr(module, "get_service_class", lambda signature: object)
    monkeypatch.setattr(module, "get_service", lambda signature: service)
