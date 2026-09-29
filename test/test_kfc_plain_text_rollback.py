"""纯文本失败输出的上下文回滚测试。

模型未返回工具调用时，本次正文不会真正发出，不应留在主链，
也不应把 ``message`` 留给提交阶段写入持久对话链。
"""

from __future__ import annotations

from typing import Any

import pytest

from plugins.kokoro_flow_chatter.runtime.orchestrator import (
    _rollback_failed_assistant,
)
from src.app.plugin_system.types import LLMPayload, ROLE, Text, ToolCall


class _FakeResponse:
    """最小化模拟带 payloads/message/call_list 的响应对象。"""

    def __init__(
        self,
        payloads: list[LLMPayload],
        message: str = "",
        call_list: list[ToolCall] | None = None,
    ) -> None:
        self.payloads = payloads
        self.message = message
        self.reasoning_content: str = ""
        self.call_list: list[ToolCall] = call_list or []


def _user(text: str) -> LLMPayload:
    return LLMPayload(ROLE.USER, [Text(text)])


def _assistant(text: str) -> LLMPayload:
    return LLMPayload(ROLE.ASSISTANT, [Text(text)])


def test_rollback_drops_trailing_assistant_and_clears_message() -> None:
    """回滚应丢弃本轮 ASSISTANT 并清空输出字段。"""
    response: Any = _FakeResponse(
        payloads=[_user("[新消息]"), _assistant("纯文本输出")],
        message="纯文本输出",
    )
    _rollback_failed_assistant(response, list(response.payloads[:1]))
    assert len(response.payloads) == 1
    assert response.payloads[-1].role == ROLE.USER
    assert response.message == ""


def test_rollback_preserves_prior_tool_call() -> None:
    """发送前已有的工具调用必须在回滚后保留。"""
    tool_call = ToolCall(id="call-1", name="kfc_reply", args={})
    success_assistant = LLMPayload(ROLE.ASSISTANT, [tool_call])
    response: Any = _FakeResponse(
        payloads=[_user("[新消息]"), success_assistant, _assistant("被拦回复")],
        message="被拦回复",
        call_list=[tool_call],
    )
    _rollback_failed_assistant(response, list(response.payloads[:2]))
    assert response.payloads[-1] is success_assistant
    assert response.message == ""


def test_rollback_restores_snapshot_when_send_trimmed_history() -> None:
    """发送后列表比原链更短时仍须剥离无效回复。"""
    history = [_user("历史"), _assistant("旧输出"), _user("当前消息")]
    response: Any = _FakeResponse(
        payloads=[history[-1], _assistant("被拦回复")], message="被拦回复"
    )
    _rollback_failed_assistant(response, history)
    assert response.payloads == history
    assert response.message == ""


def test_rollback_drops_new_user_boundary_with_refusal() -> None:
    """本轮临时触发而生成的 USER 边界也不能留在主链。"""
    response: Any = _FakeResponse(
        payloads=[_user("历史"), _user("新消息"), _assistant("纯文本输出")],
        message="纯文本输出",
    )
    _rollback_failed_assistant(response, list(response.payloads[:1]))
    assert len(response.payloads) == 1
    assert response.message == ""


@pytest.mark.parametrize(
    "before_send",
    [[], [_user("历史")]],
)
def test_rollback_partial_shapes(before_send: list[LLMPayload]) -> None:
    """不同主链快照下均只保留发送前内容。"""
    response: Any = _FakeResponse(
        payloads=[*before_send, _assistant("本轮纯文本")], message="x"
    )
    _rollback_failed_assistant(response, before_send)
    assert response.payloads == before_send
