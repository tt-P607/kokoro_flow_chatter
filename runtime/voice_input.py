"""当前未完成回合的 QQ 通话 PCM 合并。"""

from __future__ import annotations

import base64
import io
import uuid
import wave
from dataclasses import dataclass
from typing import Any

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.types import ROLE, Audio, LLMPayload

from ..multimodal import get_audio_media_list

logger = get_logger("kfc_voice_input")
_MAX_PENDING_SAMPLES = 16000 * 120


@dataclass(slots=True)
class _VoicePart:
    """一段已校验的来源 PCM 和采集顺序。"""

    order: float
    start: int | None
    end: int | None
    pcm: bytes


def _read_pcm(data: str) -> bytes:
    """解包完整的 16kHz 单声道 S16LE WAV，拒绝格式或长度不符。"""
    with wave.open(io.BytesIO(base64.b64decode(data, validate=True)), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getcomptype()) != (16000, 1, 2, "NONE"):
            raise ValueError("QQ 待处理音频必须是 16kHz 单声道 S16LE WAV")
        pcm = audio.readframes(audio.getnframes())
        if not pcm or len(pcm) != audio.getnframes() * 2:
            raise ValueError("QQ 待处理 WAV 数据不完整")
        return pcm


class PendingVoiceInput:
    """保留尚未完成模型决策的音频，按来源去重并隔离通话代次。"""

    def __init__(self) -> None:
        """建立空的回合缓冲，不读取历史音频。"""
        self._key: tuple[str, int] | None = None
        self._parts: dict[str, _VoicePart] = {}
        self._audio_value = ""
        self.turn_id = uuid.uuid4().hex

    def clear(self) -> None:
        """在回合完成或通话归属改变时释放待处理音频。"""
        self._key = None
        self._parts.clear()
        self._audio_value = ""
        self.turn_id = uuid.uuid4().hex

    def merge(self, response: Any, payload: LLMPayload | None, messages: list[Any]) -> bool:
        """仅替换待处理 USER 尾部内同一通话的音频，保留文字和历史。"""
        if payload is None:
            return False
        inputs: list[tuple[Any, dict[str, Any], str]] = []
        for message in messages:
            extra = getattr(message, "extra", {})
            voice = extra.get("qq_voice") if isinstance(extra, dict) else None
            if not isinstance(voice, dict) or voice.get("mode") != "multimodal":
                continue
            media = get_audio_media_list(message)
            if len(media) != 1 or media[0].get("format") != "wav":
                raise ValueError("QQ 通话消息必须包含一个完整 WAV")
            inputs.append((message, voice, str(media[0]["data"])))
        if not inputs:
            return False
        keys = {(str(voice["call_id"]), int(voice["generation"])) for _, voice, _ in inputs}
        if len(keys) != 1:
            self.clear()
            return False
        key = keys.pop()
        boundary = max((index for index, item in enumerate(response.payloads) if item.role != ROLE.USER), default=-1)
        tail = response.payloads[boundary + 1:]
        previous_payload = next((
            item for item in reversed(tail) if item is not payload and any(
                isinstance(part, Audio) and part.value == self._audio_value for part in item.content
            )
        ), None)
        if self._key != key or previous_payload is None:
            self.clear()
            previous_payload = None
        self._key = key
        candidates = dict(self._parts)
        source_values = {data for _, _, data in inputs}
        if self._audio_value:
            source_values.add(self._audio_value)
        for message, voice, data in inputs:
            source_id = str(voice.get("utterance_id") or message.message_id)
            pcm = _read_pcm(data)
            start = voice.get("start_sample")
            end = voice.get("end_sample")
            if (start is not None or end is not None) and (
                not isinstance(start, int) or not isinstance(end, int)
                or start < 0 or end - start != len(pcm) // 2
            ):
                raise ValueError("QQ 通话音频采样范围不匹配")
            part = _VoicePart(float(voice.get("sequence", message.time)), start, end, pcm)
            previous = candidates.get(source_id)
            if previous is not None and previous != part:
                raise ValueError("QQ 通话音频来源 ID 重复但内容不同")
            candidates[source_id] = part
        if sum(len(part.pcm) // 2 for part in candidates.values()) > _MAX_PENDING_SAMPLES:
            raise ValueError("QQ 当前待处理音频超过 120 秒，未丢弃或截断音频")
        pcm_parts: list[bytes] = []
        last_end: int | None = None
        for part in sorted(candidates.values(), key=lambda item: item.order):
            overlap = max(0, last_end - part.start) if last_end is not None and part.start is not None else 0
            pcm_parts.append(part.pcm[overlap * 2:])
            if part.end is not None:
                last_end = max(last_end or 0, part.end)
        pcm = b"".join(pcm_parts)
        output = io.BytesIO()
        with wave.open(output, "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(pcm)
        merged = base64.b64encode(output.getvalue()).decode("ascii")
        if previous_payload is not None:
            previous_payload.content = [
                part for part in previous_payload.content
                if not (isinstance(part, Audio) and part.value == self._audio_value)
            ]
        payload.content = [
            part for part in payload.content
            if not (isinstance(part, Audio) and part.value in source_values)
        ]
        payload.content.append(Audio(merged, format="wav"))
        self._parts = candidates
        self._audio_value = merged
        logger.info(
            f"QQ 待处理音频合并：turn_id={self.turn_id}, call_id={key[0]}, generation={key[1]}, "
            f"segments={len(candidates)}, pcm_bytes={len(pcm)}, wav_bytes={len(output.getvalue())}, end_reason=pending_turn"
        )
        return True