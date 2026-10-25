"""Anima 语音通话输入处理器。"""

from __future__ import annotations

import time
import uuid
from typing import Any, ClassVar

from src.app.plugin_system.api.chat_api import get_chatter_by_stream
from src.app.plugin_system.api.event_api import EventDecision
from src.app.plugin_system.api.service_api import get_service
from src.app.plugin_system.api.stream_api import add_message_to_stream, get_stream
from src.app.plugin_system.base import BaseEventHandler
from src.app.plugin_system.types import Message

VOICE_CALL_INPUT_EVENT = "anima_voice:call_input"
VOICE_CALL_SERVICE = "anima_voice:service:voice_call"
KFC_CHATTER_NAME = "kokoro_flow_chatter"


class VoiceCallInputHandler(BaseEventHandler):
    """把当前 KFC QQ 私聊通话输入写入原流未读队列。"""

    name: str = "kfc_voice_call_input_handler"
    description: str = "将 Anima 语音通话输入送入当前 KFC 私聊"
    weight: int = 0
    intercept_message: bool = False
    init_subscribe: ClassVar[list[str]] = [VOICE_CALL_INPUT_EVENT]

    async def execute(
        self,
        event_name: str,
        params: dict[str, Any],
    ) -> tuple[EventDecision, dict[str, Any]]:
        """验证通话与流归属并注入一条用户消息。"""
        _ = event_name
        stream_id = params.get("stream_id")
        user_id = params.get("user_id")
        user_name = params.get("user_name")
        call_id = params.get("call_id")
        generation = params.get("generation")
        mode = params.get("mode")
        text = params.get("text")
        audio_base64 = params.get("audio_base64")
        audio_format = params.get("format")
        if not all(isinstance(value, str) and value for value in (
            stream_id, user_id, call_id, mode
        )):
            return EventDecision.PASS, params
        if mode not in ("asr", "multimodal"):
            return EventDecision.PASS, params

        stream = await get_stream(stream_id)
        chatter = get_chatter_by_stream(stream_id)
        if (
            stream is None
            or stream.stream_id != stream_id
            or stream.platform != "qq"
            or str(stream.chat_type) not in ("private", "ChatType.PRIVATE")
            or getattr(chatter, "name", "") != KFC_CHATTER_NAME
        ):
            return EventDecision.PASS, params

        service = get_service(VOICE_CALL_SERVICE)
        if service is None:
            return EventDecision.PASS, params
        state = service.get_state(stream_id)
        if (
            not isinstance(state, dict)
            or state.get("phase") != "connected"
            or state.get("call_id") != call_id
            or state.get("generation") != generation
            or state.get("user_id") != user_id
            or state.get("bot_id") != stream.bot_id
            or state.get("input_mode") != mode
        ):
            return EventDecision.PASS, params

        media: list[dict[str, str]] = []
        if mode == "asr":
            if not isinstance(text, str) or not text.strip() or audio_base64 != "":
                return EventDecision.PASS, params
            body = text.strip()
        else:
            if (
                text != ""
                or not isinstance(audio_base64, str)
                or not audio_base64
                or audio_format != "wav"
            ):
                return EventDecision.PASS, params
            body = "[语音通话音频]"
            media.append({"type": "audio", "data": audio_base64, "format": "wav"})

        source: dict[str, Any] = {"call_id": call_id, "generation": generation, "mode": mode}
        utterance_id = params.get("utterance_id")
        if isinstance(utterance_id, str) and utterance_id:
            source["utterance_id"] = utterance_id
        else:
            utterance_id = uuid.uuid4().hex
        if isinstance(params.get("sequence"), int):
            source["sequence"] = params["sequence"]
        stats = params.get("audio_stats")
        if isinstance(stats, dict):
            for key in ("start_sample", "end_sample", "voice_ms", "end_reason"):
                if key in stats:
                    source[key] = stats[key]

        message = Message(
            message_id=f"anima_voice_{call_id}_{generation}_{utterance_id}",
            time=time.time(),
            content=body,
            processed_plain_text=body,
            sender_id=user_id,
            sender_name=user_name if isinstance(user_name, str) and user_name else "用户",
            platform="qq",
            chat_type="private",
            stream_id=stream_id,
            qq_voice=source,
            media=media,
        )
        await add_message_to_stream(message)
        return EventDecision.SUCCESS, params