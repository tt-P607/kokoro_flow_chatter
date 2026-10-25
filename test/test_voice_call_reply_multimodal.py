"""KFC 语音回复与多模态输入测试。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from plugins.kokoro_flow_chatter.actions import reply as reply_module
from plugins.kokoro_flow_chatter.actions.reply import (
    KFCReplyAction,
    KFCVoiceReplyAction,
    execute_voice_reply,
)
from plugins.kokoro_flow_chatter.multimodal import (
    build_multimodal_content,
    extract_media_from_messages,
)
from plugins.kokoro_flow_chatter.plugin import KFCPlugin
from plugins.kokoro_flow_chatter.runtime.turn_controller import (
    _with_voice_call_context,
)
from src.app.plugin_system.types import Audio, Image, Message, Text
from src.core.components.types import PlatformSendResult, PlatformSendStatus


@pytest.mark.asyncio
@pytest.mark.parametrize("playback_success", [False, True])
@pytest.mark.parametrize("list_content", [False, True])
async def test_complete_voice_reply_uses_one_framework_call_without_typing_delay(
    monkeypatch: pytest.MonkeyPatch,
    playback_success: bool,
    list_content: bool,
) -> None:
    """完整正文只执行一次真实语音 Action，不使用文本分段或打字延迟。"""
    from plugins.kokoro_flow_chatter.chatter import KokoroFlowChatter
    from plugins.kokoro_flow_chatter.config import KFCConfig
    from plugins.kokoro_flow_chatter.execution import decision_executor
    from plugins.kokoro_flow_chatter.protocol.tool_call_adapter import (
        build_decision_draft,
    )
    from src.app.plugin_system.types import ToolCall, ToolResult
    from src.core.utils import llm_tool_call

    stream = SimpleNamespace(stream_id="stream-1", platform="qq", chat_type="private")
    plugin = SimpleNamespace()
    trigger = Message(stream_id="stream-1", content="输入")
    chatter = object.__new__(KokoroFlowChatter)
    chatter.stream_id = "stream-1"
    normalized_results: list[tuple[bool, object]] = []

    async def framework_execute(
        action_type: type,
        message: Message,
        **kwargs: Any,
    ) -> tuple[bool, object]:
        """使用真实框架调度和规范化，不伪造嵌套返回元组。"""
        result = await llm_tool_call.exec_llm_usable(
            action_type, plugin=cast(Any, plugin), stream_id=stream.stream_id,
            message=message, kwargs=kwargs,
        )
        normalized_results.append(result)
        return result

    speak = AsyncMock()

    async def voice_result(*args: Any, **kwargs: Any) -> dict[str, bool]:
        """只替换实际音频服务，不操作任何通话。"""
        return {"drained": playback_success}

    speak.side_effect = voice_result
    service = SimpleNamespace(
        get_state=lambda stream_id: {"stream_id": stream_id, "phase": "connected"},
        speak=speak,
    )
    monkeypatch.setattr(reply_module, "get_service", lambda signature: service)
    monkeypatch.setattr(llm_tool_call, "_get_action_chat_stream", AsyncMock(return_value=stream))
    monkeypatch.setattr(chatter, "exec_llm_usable", framework_execute)
    typing_delay = Mock(wraps=decision_executor.calculate_typing_delay)
    monkeypatch.setattr(decision_executor, "calculate_typing_delay", typing_delay)
    warning = Mock()
    monkeypatch.setattr(decision_executor, "logger", SimpleNamespace(info=Mock(), debug=Mock(), warning=warning))
    segments = ["第一段正文。", "第二段正文。", "第三段正文。", "第四段正文。"]
    text = "\n".join(segments)
    config = KFCConfig()
    config.reply.typing_chars_per_sec = 0
    response = SimpleNamespace(payloads=[], add_payload=Mock())
    draft = build_decision_draft([
        ToolCall(
            name="action-kfc_voice_reply", id="voice-round",
            args={"content": segments if list_content else text, "voice_style": "warm"},
        )
    ])

    async def send_voice(content: str, voice_style: str) -> tuple[bool, str]:
        """串起实际 Chatter 回复方法与分段执行器。"""
        return await chatter.send_voice_reply(content, voice_style, trigger)

    result = await decision_executor.execute_decision_draft(
        draft, response, usable_map=cast(Any, {}), trigger_msg=trigger, config=config,
        execute_reply_fn=AsyncMock(side_effect=AssertionError("unexpected text reply")),
        execute_voice_reply_fn=send_voice,
        run_tool_call_fn=AsyncMock(side_effect=AssertionError("unexpected external tool")),
    )

    speak.assert_awaited_once_with(
        "stream-1", text, voice_style="warm", trace_id="voice-round"
    )
    receipt = "通话语音已完成发送" if playback_success else "通话语音未确认完成发送"
    assert normalized_results == [(playback_success, receipt)]
    typing_delay.assert_not_called()
    assert all(isinstance(receipt, str) for _, receipt in normalized_results)
    assert result.has_reply is True
    assert result.has_failed_tool is (not playback_success)
    tool_result = response.add_payload.call_args.args[0].content[0]
    assert isinstance(tool_result, ToolResult)
    assert tool_result.call_id == "voice-round"
    assert tool_result.value == receipt
    if playback_success:
        warning.assert_not_called()
    else:
        logged = "\n".join(call.args[0] for call in warning.call_args_list)
        assert "trace_id=voice-round" in logged
        assert "通话语音未确认完成发送" in logged
        assert all(content not in logged for content in segments)


def test_voice_action_is_registered_at_kfc_initialization() -> None:
    """通话动作在初始化时固定注册，不依赖通话当前是否接通。"""
    assert KFCVoiceReplyAction in KFCPlugin().get_components()
    KFCVoiceReplyAction.validate_associated_types()
    schema = KFCVoiceReplyAction.to_schema()
    properties = schema["function"]["parameters"]["properties"]
    required = schema["function"]["parameters"]["required"]
    assert "voice_style" in properties
    assert properties["content"]["type"] == "string"
    assert {
        "thought",
        "expected_reaction",
        "max_wait_seconds",
        "mood",
    } <= set(required)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_to", ["", "message-1"])
async def test_connected_reply_sends_text_without_synthesizing_voice(
    monkeypatch: pytest.MonkeyPatch,
    reply_to: str,
) -> None:
    """普通回复在通话中仍发送文本和引用，不调用语音服务。"""
    stream = SimpleNamespace(stream_id="stream-1")
    action = KFCReplyAction(stream, SimpleNamespace())
    service = SimpleNamespace(
        get_state=lambda stream_id: {"phase": "connected"},
        speak=AsyncMock(return_value={"drained": True}),
    )
    sent_result = PlatformSendResult(success=True, status=PlatformSendStatus.SENT)
    send_text = AsyncMock(return_value=sent_result)
    send_stream = AsyncMock(return_value=sent_result)
    monkeypatch.setattr(reply_module, "get_service", lambda signature: service)
    monkeypatch.setattr(reply_module, "send_text_result", send_text)
    monkeypatch.setattr(action, "_send_to_stream_result", send_stream)

    result = await action.execute(
        ["你好"],
        thought="私有想法",
        expected_reaction="回应",
        mood="开心",
        reply_to=reply_to,
    )

    assert result == (True, "消息已发送: 你好")
    service.speak.assert_not_awaited()
    if reply_to:
        send_text.assert_awaited_once_with(
            content="你好", stream_id="stream-1", reply_to=reply_to
        )
        send_stream.assert_not_awaited()
    else:
        send_text.assert_not_awaited()
        send_stream.assert_awaited_once_with("你好")


@pytest.mark.asyncio
async def test_voice_reply_surfaces_speak_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """专用语音回复显式返回发送失败，不改发文字。"""
    action = KFCVoiceReplyAction(
        SimpleNamespace(stream_id="stream-1", platform="qq", chat_type="private"),
        SimpleNamespace(),
    )
    service = SimpleNamespace(
        get_state=lambda stream_id: {"stream_id": stream_id, "phase": "connected"},
        speak=AsyncMock(side_effect=RuntimeError("voice failed")),
    )
    monkeypatch.setattr(reply_module, "get_service", lambda signature: service)

    result = await action.execute("你好")

    assert result == (False, "通话语音发送失败: voice failed")


@pytest.mark.asyncio
async def test_voice_reply_requires_drained_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """专用语音回复在音频未 drained 时不得返回成功。"""
    action = KFCVoiceReplyAction(
        SimpleNamespace(stream_id="stream-1", platform="qq", chat_type="private"),
        SimpleNamespace(),
    )
    service = SimpleNamespace(
        get_state=lambda stream_id: {"stream_id": stream_id, "phase": "connected"},
        speak=AsyncMock(return_value={"drained": False}),
    )
    monkeypatch.setattr(reply_module, "get_service", lambda signature: service)

    result = await action.execute("你好")

    assert result == (False, "通话语音未确认完成发送")


@pytest.mark.asyncio
async def test_reply_without_connected_call_uses_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无通话或未连接时保持原文本发送路径。"""
    action = KFCReplyAction(SimpleNamespace(stream_id="stream-1"), SimpleNamespace())
    monkeypatch.setattr(
        reply_module,
        "get_service",
        lambda signature: SimpleNamespace(get_state=lambda stream_id: None),
    )
    send_stream = AsyncMock(return_value=PlatformSendResult(success=True, status=PlatformSendStatus.SENT))
    monkeypatch.setattr(action, "_send_to_stream_result", send_stream)

    result = await action.execute(["你好"])

    assert result[0] is True
    send_stream.assert_awaited_once_with("你好")


def test_voice_audio_bypasses_image_switch_without_changing_image_behavior() -> None:
    """通话音频独立于图片开关，普通图片仍遵循原生多模态设置。"""
    voice = Message(
        content="[语音通话音频]",
        media=[{"type": "audio", "data": "YXVkaW8=", "format": "wav"}],
        qq_voice={"call_id": "c", "generation": 1, "mode": "multimodal"},
    )
    image = Message(content="图片", media=[{"type": "image", "data": "aW1hZ2U="}])
    ordinary_audio = Message(
        content="普通音频",
        media=[{"type": "audio", "data": "b3RoZXI=", "format": "wav"}],
    )

    disabled_items = extract_media_from_messages(
        [voice, image, ordinary_audio], include_images=False
    )
    enabled_items = extract_media_from_messages(
        [voice, image, ordinary_audio], include_images=True
    )

    assert [item["type"] for item in disabled_items] == ["audio"]
    assert [item["type"] for item in enabled_items] == ["audio", "image"]
    content = build_multimodal_content("输入", enabled_items)
    assert isinstance(content[1], Audio)
    assert content[1].format == "wav"
    assert isinstance(content[2], Image)


@pytest.mark.asyncio
async def test_voice_context_is_request_only_and_describes_input_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """当前通话上下文仅描述输入和 KFC 播音方式，不构成固定分析要求。"""
    service = SimpleNamespace(
        get_state=lambda stream_id: {
            "stream_id": stream_id,
            "phase": "connected",
            "input_mode": "multimodal",
        },
        get_speech_options=lambda: {"styles": ["warm"], "default_style": "warm"},
    )
    monkeypatch.setattr(
        "src.app.plugin_system.api.service_api.get_service",
        lambda signature: service,
    )
    template = SimpleNamespace(set=Mock(), build=AsyncMock(return_value="共享朗读规则"))
    monkeypatch.setattr(
        "src.app.plugin_system.api.prompt_api.get_template",
        lambda name: template if name == "tts_voice_plugin-neo.speech_rules" else None,
    )

    payload = await _with_voice_call_context(None, "stream-1")

    assert payload is not None
    text = "".join(
        part.text for part in payload.content if isinstance(part, Text)
    )
    assert "multimodal" in text
    assert "action-kfc_voice_reply" in text
    assert "action-kfc_reply 发送文本消息" in text
    assert "两种动作可以在同一轮分别调用" in text
    assert "文本不会自动转成语音" in text
    assert "语音 content 使用完整正文字符串" in text
    assert "不按句拆分或模拟打字等待" in text
    assert "共享朗读规则" in text
    assert "voice_style 可选：warm；默认：warm" in text
    template.set.assert_called_once_with("stream_id", "stream-1")
    assert "会话" in text
    assert "分析" not in text


def test_speech_rule_declaration_tracks_current_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """共享 TTS 仅在当前流已接通时抑制普通语音条提醒。"""
    from plugins.kokoro_flow_chatter.chatter import KokoroFlowChatter

    chatter = object.__new__(KokoroFlowChatter)
    chatter.stream_id = "stream-1"
    state = {"stream_id": "stream-1", "phase": "connected"}
    service = SimpleNamespace(get_state=lambda stream_id: state)
    monkeypatch.setattr(
        "src.app.plugin_system.api.service_api.get_service", lambda signature: service
    )
    assert chatter.speech_rules_template_name == "tts_voice_plugin-neo.speech_rules"
    state["phase"] = "ended"
    assert chatter.speech_rules_template_name is None
    state.update(stream_id="other", phase="connected")
    assert chatter.speech_rules_template_name is None


@pytest.mark.asyncio
async def test_voice_reply_passes_style_and_requires_matching_connected_qq_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """专用语音动作只把当前 QQ 私聊正文和风格送入通话服务。"""
    stream = SimpleNamespace(
        stream_id="stream-1",
        platform="qq",
        chat_type="private",
    )
    action = KFCVoiceReplyAction(stream, SimpleNamespace())
    service = SimpleNamespace(
        get_state=lambda stream_id: {
            "stream_id": stream_id,
            "phase": "connected",
        },
        speak=AsyncMock(return_value={"drained": True}),
    )
    monkeypatch.setattr(reply_module, "get_service", lambda signature: service)

    result = await action.execute(
        "正文一。\n正文二！",
        voice_style="warm",
        thought="私有想法",
        expected_reaction="回应",
        max_wait_seconds=3,
        mood="平静",
    )

    assert result == (True, "通话语音已完成发送")
    assert service.speak.await_count == 1
    assert service.speak.await_args_list[0].args == ("stream-1", "正文一。\n正文二！")
    first_kwargs = service.speak.await_args_list[0].kwargs
    assert first_kwargs["voice_style"] == "warm"
    assert first_kwargs["trace_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stream", "state", "expected"),
    [
        (SimpleNamespace(stream_id="s", platform="qq", chat_type="private"), None,
         "没有已接通的通话"),
        (SimpleNamespace(stream_id="s", platform="qq", chat_type="private"),
         {"stream_id": "other", "phase": "connected"}, "没有已接通的通话"),
        (SimpleNamespace(stream_id="s", platform="qq", chat_type="private"),
         {"stream_id": "s", "phase": "ringing"}, "没有已接通的通话"),
        (SimpleNamespace(stream_id="s", platform="other", chat_type="private"),
         {"stream_id": "s", "phase": "connected"}, "仅支持当前 QQ 私聊通话"),
    ],
)
async def test_voice_reply_rejects_wrong_or_disconnected_stream(
    monkeypatch: pytest.MonkeyPatch,
    stream: SimpleNamespace,
    state: dict[str, str] | None,
    expected: str,
) -> None:
    """不匹配或未接通时失败，且不调用语音发送服务。"""
    service = SimpleNamespace(
        get_state=lambda _stream_id: state,
        speak=AsyncMock(),
    )
    monkeypatch.setattr(reply_module, "get_service", lambda signature: service)

    success, receipt = await execute_voice_reply(stream, "正文", "")

    assert success is False
    assert expected in receipt
    service.speak.assert_not_awaited()