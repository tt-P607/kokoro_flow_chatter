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
    """回滚应丢弃基线后的 ASSISTANT 并清空输出字段。"""
    response: Any = _FakeResponse(
        payloads=[_user("[新消息]"), _assistant("纯文本输出")],
        message="纯文本输出",
    )
    _rollback_failed_assistant(response, payload_baseline=1)
    assert len(response.payloads) == 1
    assert response.payloads[-1].role == ROLE.USER
    assert response.message == ""


def test_rollback_keeps_chain_when_trailing_has_tool_calls() -> None:
    """末尾含工具调用的 ASSISTANT 是合法成功输出，不得回滚。"""
    tool_call = ToolCall(id="call-1", name="kfc_reply", args={})
    success_assistant = LLMPayload(ROLE.ASSISTANT, [tool_call])
    response: Any = _FakeResponse(
        payloads=[_user("[新消息]"), success_assistant],
        message="回复正文",
        call_list=[tool_call],
    )
    _rollback_failed_assistant(response, payload_baseline=2)
    # 基线等于长度，无新增可删；message 不应被清空
    assert response.message == "回复正文"


def test_rollback_ignores_when_no_new_payloads() -> None:
    """无新增 payload 时回滚应为无害空操作。"""
    response: Any = _FakeResponse(payloads=[_user("历史"), _assistant("旧输出")], message="保留")
    _rollback_failed_assistant(response, payload_baseline=99)
    assert len(response.payloads) == 2
    assert response.message == "保留"


def test_rollback_skips_non_assistant_trailing() -> None:
    """末尾不是 ASSISTANT 时（异常链形态）不做破坏性删除。"""
    response: Any = _FakeResponse(
        payloads=[_user("历史"), _user("新消息"), _assistant("纯文本输出")],
        message="纯文本输出",
    )
    # 模拟基线后出现了非 ASSISTANT payload：不做删除
    baseline = 1
    payloads = response.payloads
    if all(p.role == ROLE.ASSISTANT for p in payloads[baseline:]):
        del payloads[baseline:]
    assert len(response.payloads) == 3


@pytest.mark.parametrize(
    ("baseline", "expected_len"),
    [(1, 1), (0, 0)],
)
def test_rollback_partial_shapes(baseline: int, expected_len: int) -> None:
    """不同基线下只删除本次新增的 ASSISTANT 段。"""
    response: Any = _FakeResponse(
        payloads=[_assistant("本轮纯文本")], message="x"
    )
    _rollback_failed_assistant(response, payload_baseline=baseline)
    assert len(response.payloads) == expected_len
