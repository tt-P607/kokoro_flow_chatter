"""KokoroFlow chatter 发送回执回归测试。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.kokoro_flow_chatter.actions.reply import KFCReplyAction
from src.core.components.types import PlatformSendResult, PlatformSendStatus


def _action_instance(action_type: type) -> object:
    """构造不依赖框架运行时的 Action 实例。"""
    action = object.__new__(action_type)
    action.chat_stream = SimpleNamespace(
        stream_id="stream-1",
        platform="test",
        chat_type="private",
        context=SimpleNamespace(triggering_user_id="user-1"),
    )
    action.plugin = SimpleNamespace(config=None)
    return action


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [PlatformSendStatus.FAILED, PlatformSendStatus.UNKNOWN])
async def test_reply_propagates_detailed_result(
    monkeypatch: pytest.MonkeyPatch,
    status: PlatformSendStatus,
) -> None:
    """KokoroFlow 普通回复将详细发送状态与错误传回调用方。"""
    action = _action_instance(KFCReplyAction)
    result = PlatformSendResult(success=False, status=status, error="bridge rejected")
    monkeypatch.setattr(action, "_send_to_stream_result", AsyncMock(return_value=result))

    response = await action.execute(["hello"])

    assert response[0] is False
    assert "bridge rejected" in response[1]
