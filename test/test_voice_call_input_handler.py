"""KFC Anima 语音通话输入处理器测试。"""

from __future__ import annotations

import base64
import io
import wave
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from plugins.kokoro_flow_chatter.handlers import voice_call_input_handler as module
from plugins.kokoro_flow_chatter.handlers.voice_call_input_handler import (
    VoiceCallInputHandler,
)
from src.app.plugin_system.api.event_api import EventDecision
from src.app.plugin_system.types import StreamContext


def _voice_input(sequence: int, pcm: bytes, *, generation: int = 3) -> Any:
    """构造带通话来源和合法 WAV 的独立音频消息。"""
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(pcm)
    return SimpleNamespace(
        message_id=f"message-{generation}-{sequence}",
        time=sequence,
        content="[语音通话音频]",
        extra={
            "qq_voice": {
                "call_id": "call-1", "generation": generation, "mode": "multimodal",
                "utterance_id": f"utterance-{sequence}", "sequence": sequence,
                "start_sample": (sequence - 1) * 160,
                "end_sample": (sequence - 1) * 160 + len(pcm) // 2,
            },
            "media": [{"type": "audio", "format": "wav",
                       "data": base64.b64encode(output.getvalue()).decode("ascii")}],
        },
    )


def _audio_payload(messages: list[Any]) -> Any:
    """使用实际多模态转换创建待处理输入。"""
    from plugins.kokoro_flow_chatter.multimodal import (
        build_multimodal_content,
        extract_media_from_messages,
    )
    from src.app.plugin_system.types import ROLE, LLMPayload

    return LLMPayload(ROLE.USER, build_multimodal_content(
        "输入", extract_media_from_messages(messages, include_images=False)
    ))


def _pcm_parts(payloads: list[Any]) -> list[bytes]:
    """解包请求中的全部合法音频，避免把连接 WAV 文件当成 PCM 合并。"""
    from src.app.plugin_system.types import Audio

    result: list[bytes] = []
    for payload in payloads:
        for part in payload.content:
            if isinstance(part, Audio):
                with wave.open(io.BytesIO(base64.b64decode(part.value)), "rb") as audio:
                    assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
                    result.append(audio.readframes(audio.getnframes()))
    return result


def test_pending_voice_merges_interruptions_once_in_capture_order() -> None:
    """连续打断时合并原音频与追加音频，去重且不触及完成的历史。"""
    from plugins.kokoro_flow_chatter.runtime.voice_input import PendingVoiceInput

    historical = _audio_payload([_voice_input(1, b"\x09\x00" * 160)])
    from src.app.plugin_system.types import ROLE, LLMPayload, Text

    response = SimpleNamespace(payloads=[historical, LLMPayload(ROLE.ASSISTANT, Text("完成"))])
    pending = PendingVoiceInput()
    first = _voice_input(1, b"\x01\x00" * 160)
    second = _voice_input(2, b"\x02\x00" * 160)
    third = _voice_input(3, b"\x03\x00" * 160)
    for messages in ([first], [third, second], [second]):
        payload = _audio_payload(messages)
        response.payloads.append(payload)
        pending.merge(response, payload, messages)
    assert _pcm_parts(response.payloads) == [
        b"\x09\x00" * 160,
        b"\x01\x00" * 160 + b"\x02\x00" * 160 + b"\x03\x00" * 160,
    ]


def test_pending_voice_keeps_generations_and_completed_rounds_separate() -> None:
    """新代次和模型已完成的回合不再合入之前的待处理音频。"""
    from plugins.kokoro_flow_chatter.runtime.voice_input import PendingVoiceInput
    from src.app.plugin_system.types import ROLE, LLMPayload, Text

    response = SimpleNamespace(payloads=[])
    pending = PendingVoiceInput()
    for sequence, generation in ((1, 3), (2, 4)):
        message = _voice_input(sequence, bytes([sequence, 0]) * 160, generation=generation)
        payload = _audio_payload([message])
        response.payloads.append(payload)
        pending.merge(response, payload, [message])
    response.payloads.append(LLMPayload(ROLE.ASSISTANT, Text("完成")))
    message = _voice_input(3, b"\x03\x00" * 160, generation=4)
    payload = _audio_payload([message])
    response.payloads.append(payload)
    pending.merge(response, payload, [message])
    assert _pcm_parts(response.payloads) == [bytes([number, 0]) * 160 for number in (1, 2, 3)]


def test_identical_pcm_from_different_generations_does_not_remove_old_audio() -> None:
    """不同代次的音频即使字节相同也不能按内容误删。"""
    from plugins.kokoro_flow_chatter.runtime.voice_input import PendingVoiceInput

    response = SimpleNamespace(payloads=[])
    pending = PendingVoiceInput()
    pcm = b"\x01\x00" * 160
    for generation in (3, 4):
        message = _voice_input(1, pcm, generation=generation)
        payload = _audio_payload([message])
        response.payloads.append(payload)
        pending.merge(response, payload, [message])
    assert _pcm_parts(response.payloads) == [pcm, pcm]


def test_framework_request_keeps_media_payload_identity_for_pending_merge() -> None:
    """真实请求的自动 USER 合并不能绕过待处理 PCM 的单 WAV 合并。"""
    from plugins.kokoro_flow_chatter.runtime.turn_controller import (
        _append_or_merge_user_payload,
    )
    from plugins.kokoro_flow_chatter.runtime.voice_input import PendingVoiceInput
    from src.kernel.llm.request import LLMRequest

    response = LLMRequest([])
    pending = PendingVoiceInput()
    for index in (1, 2, 3):
        message = _voice_input(index, bytes([index, 0]) * 160)
        payload = _audio_payload([message])
        _append_or_merge_user_payload(response, payload, allow_merge=False)
        assert any(item is payload for item in response.payloads)
        pending.merge(response, payload, [message])
    assert _pcm_parts(response.payloads) == [b"\x01\x00" * 160 + b"\x02\x00" * 160 + b"\x03\x00" * 160]


@pytest.mark.asyncio
async def test_orchestrator_interruptions_send_one_wav_and_save_merged_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """主循环被连续打断后只重发合并 WAV，完成后的下一回合独立。"""
    from plugins.kokoro_flow_chatter.config import KFCConfig
    from plugins.kokoro_flow_chatter.runtime import orchestrator as module
    from plugins.kokoro_flow_chatter.runtime.turn_controller import TurnInputResult
    from plugins.kokoro_flow_chatter.snapshot import deserialize_snapshot
    from plugins.kokoro_flow_chatter.test.test_kfc_guard_recovery import (
        _OrchestratorHarness,
        _ScriptExhausted,
    )

    harness = _OrchestratorHarness(monkeypatch, [], blocked=0)
    harness.install(module)
    harness.session.add_interrupt_event = lambda messages: None
    messages = [_voice_input(index, bytes([index, 0]) * 160) for index in range(1, 5)]
    prepared = iter(messages)
    sent_pcm: list[list[bytes]] = []
    persisted_pcm: list[list[bytes]] = []

    async def prepare(*args: Any, **kwargs: Any) -> TurnInputResult:
        """以独立网络分句驱动实际主循环的持久输入分支。"""
        try:
            message = next(prepared)
        except StopIteration as error:
            raise _ScriptExhausted from error
        payload = _audio_payload([message])
        harness.chain.payloads.append(payload)
        return TurnInputResult(
            response=harness.chain, unread_msgs=[message], persistent_user_payload=payload
        )

    async def send(*args: Any, **kwargs: Any) -> tuple[Any, list[Any]]:
        """记录实际发送视图，前两次模拟新消息触发的取消。"""
        view = args[1]
        sent_pcm.append(_pcm_parts(view.payloads))
        restored = deserialize_snapshot(harness.session.context_snapshot)
        assert restored is not None
        persisted_pcm.append(_pcm_parts(restored))
        if len(sent_pcm) < 3:
            return None, [messages[len(sent_pcm)]]
        return harness.response_for_view(len(sent_pcm)), []

    monkeypatch.setattr(module, "prepare_turn_input", prepare)
    monkeypatch.setattr(module, "_send_llm_request", send)
    config = KFCConfig()
    config.buffer.interrupt_cooldown = 0
    config.general.enable_input_status = False
    await harness.drive(module, config)
    first, second, third, fourth = [bytes([index, 0]) * 160 for index in range(1, 5)]
    assert sent_pcm == [[first], [first + second], [first + second + third], [first + second + third, fourth]]
    assert persisted_pcm == sent_pcm


def test_pending_voice_removes_source_overlap_and_rejects_incomplete_wav() -> None:
    """只保留一次重叠采样，损坏的 WAV 不改变待处理链。"""
    from plugins.kokoro_flow_chatter.runtime.voice_input import PendingVoiceInput

    first = _voice_input(1, b"\x01\x00" * 320)
    second = _voice_input(2, b"\x01\x00" * 160 + b"\x02\x00" * 160)
    response = SimpleNamespace(payloads=[])
    pending = PendingVoiceInput()
    for message in (first, second):
        payload = _audio_payload([message])
        response.payloads.append(payload)
        pending.merge(response, payload, [message])
    assert _pcm_parts(response.payloads) == [b"\x01\x00" * 320 + b"\x02\x00" * 160]
    damaged = _voice_input(3, b"\x03\x00" * 160)
    data = damaged.extra["media"][0]["data"]
    damaged.extra["media"][0]["data"] = base64.b64encode(base64.b64decode(data)[:-2]).decode("ascii")
    payload = _audio_payload([damaged])
    response.payloads.append(payload)
    with pytest.raises(ValueError, match="数据不完整"):
        pending.merge(response, payload, [damaged])


def _params(**overrides: object) -> dict[str, object]:
    """构造通话输入事件参数。"""
    params: dict[str, object] = {
        "stream_id": "qq:private:user-1",
        "user_id": "user-1",
        "user_name": "用户",
        "call_id": "call-1",
        "generation": 3,
        "mode": "asr",
        "text": "你好",
        "audio_base64": "",
        "format": "",
    }
    params.update(overrides)
    return params


@pytest.fixture
def route(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[VoiceCallInputHandler, AsyncMock, StreamContext]:
    """安装隔离的当前流、KFC 绑定与通话服务。"""
    stream = SimpleNamespace(
        stream_id="qq:private:user-1",
        platform="qq",
        chat_type="private",
        bot_id="bot-1",
        context=StreamContext(stream_id="qq:private:user-1"),
    )
    stream.context.current_message = SimpleNamespace(sender_id="user-1")
    monkeypatch.setattr(module, "get_stream", AsyncMock(return_value=stream))
    monkeypatch.setattr(
        module,
        "get_chatter_by_stream",
        lambda stream_id: SimpleNamespace(name="kokoro_flow_chatter"),
    )
    service = SimpleNamespace(
        get_state=lambda stream_id: (
            {
                "phase": "connected",
                "call_id": "call-1",
                "generation": 3,
                "user_id": "user-1",
                "user_name": "用户",
                "bot_id": "bot-1",
                "input_mode": "asr",
            }
            if stream_id == "qq:private:user-1"
            else None
        )
    )
    monkeypatch.setattr(module, "get_service", lambda signature: service)
    async def add_message_to_unread(message: object) -> None:
        stream.context.add_unread_message(message)

    add_message = AsyncMock(side_effect=add_message_to_unread)
    monkeypatch.setattr(module, "add_message_to_stream", add_message)
    return VoiceCallInputHandler(SimpleNamespace()), add_message, stream.context


@pytest.mark.asyncio
async def test_asr_input_is_written_as_user_message(
    route: tuple[VoiceCallInputHandler, AsyncMock, StreamContext],
) -> None:
    """ASR 输入写入原流，且保留通话关联元数据。"""
    handler, add_message, context = route
    params = _params()

    decision, returned = await handler.execute(module.VOICE_CALL_INPUT_EVENT, params)

    assert decision is EventDecision.SUCCESS
    assert returned is params
    add_message.assert_awaited_once()
    message = add_message.await_args.args[0]
    assert message.stream_id == "qq:private:user-1"
    assert message.sender_id == "user-1"
    assert message.content == "你好"
    assert message.extra["qq_voice"] == {
        "call_id": "call-1",
        "generation": 3,
        "mode": "asr",
    }
    assert message.extra["media"] == []
    assert context.unread_messages == [message]


@pytest.mark.asyncio
async def test_verified_call_user_is_accepted_when_current_message_is_system_notice(
    route: tuple[VoiceCallInputHandler, AsyncMock, StreamContext],
) -> None:
    """当前消息变为系统通知时仍按已验证的通话身份接收输入。"""
    handler, add_message, context = route
    context.current_message = SimpleNamespace(sender_id="system")

    decision, _ = await handler.execute(module.VOICE_CALL_INPUT_EVENT, _params())

    assert decision is EventDecision.SUCCESS
    add_message.assert_awaited_once()
    assert context.unread_messages[0].sender_id == "user-1"


@pytest.mark.asyncio
async def test_multimodal_input_keeps_audio_media_and_neutral_text(
    route: tuple[VoiceCallInputHandler, AsyncMock, StreamContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """多模态输入只以中性文本展示，原音频保存在媒体字段。"""
    handler, add_message, _ = route
    params = _params(
        mode="multimodal",
        text="",
        audio_base64="YWJj",
        format="wav",
    )
    monkeypatch.setattr(
        module,
        "get_service",
        lambda signature: SimpleNamespace(
            get_state=lambda stream_id: {
                "phase": "connected",
                "call_id": "call-1",
                "generation": 3,
                "user_id": "user-1",
                "user_name": "用户",
                "bot_id": "bot-1",
                "input_mode": "multimodal",
            }
        ),
    )

    decision, _ = await handler.execute(module.VOICE_CALL_INPUT_EVENT, params)

    assert decision is EventDecision.SUCCESS
    message = add_message.await_args.args[0]
    assert message.content == "[语音通话音频]"
    assert message.extra["media"] == [
        {"type": "audio", "data": "YWJj", "format": "wav"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state_override",
    [
        {"user_id": "other-user"},
        {"bot_id": "other-bot"},
        {"input_mode": "multimodal"},
    ],
)
async def test_state_identity_or_mode_mismatch_is_rejected(
    route: tuple[VoiceCallInputHandler, AsyncMock, StreamContext],
    state_override: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """通话服务中的用户或输入模式不匹配时拒绝注入。"""
    handler, add_message, _ = route
    state = {
        "phase": "connected",
        "call_id": "call-1",
        "generation": 3,
        "user_id": "user-1",
        "user_name": "用户",
        "bot_id": "bot-1",
        "input_mode": "asr",
    }
    state.update(state_override)
    monkeypatch.setattr(
        module,
        "get_service",
        lambda signature: SimpleNamespace(get_state=lambda stream_id: state),
    )

    decision, _ = await handler.execute(module.VOICE_CALL_INPUT_EVENT, _params())

    assert decision is EventDecision.PASS
    add_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"call_id": "expired"},
        {"generation": 2},
        {"stream_id": "other-stream"},
        {"user_id": "other-user"},
        {"platform": "telegram"},
        {"chat_type": "group"},
    ],
)
async def test_unmatched_call_input_is_rejected(
    route: tuple[VoiceCallInputHandler, AsyncMock, StreamContext],
    overrides: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """过期通话或另一流的事件不得写入消息流。"""
    handler, add_message, _ = route
    stream = SimpleNamespace(
        stream_id=overrides.get("stream_id", "qq:private:user-1"),
        platform=overrides.get("platform", "qq"),
        chat_type=overrides.get("chat_type", "private"),
        bot_id="bot-1",
        context=StreamContext(stream_id="qq:private:user-1"),
    )
    stream.context.current_message = SimpleNamespace(sender_id="user-1")
    monkeypatch.setattr(module, "get_stream", AsyncMock(return_value=stream))

    decision, _ = await handler.execute(
        module.VOICE_CALL_INPUT_EVENT, _params(**overrides)
    )

    assert decision is EventDecision.PASS
    add_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_kfc_binding_is_rejected_without_rebinding(
    route: tuple[VoiceCallInputHandler, AsyncMock, StreamContext],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非 KFC 当前绑定不处理事件，也不尝试重绑。"""
    handler, add_message, _ = route
    binding = lambda stream_id: SimpleNamespace(name="other_chatter")
    monkeypatch.setattr(module, "get_chatter_by_stream", binding)

    decision, _ = await handler.execute(module.VOICE_CALL_INPUT_EVENT, _params())

    assert decision is EventDecision.PASS
    add_message.assert_not_awaited()
