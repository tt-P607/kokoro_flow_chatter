"""KFC 事件处理器。

- ``ProactiveHandler``：响应主动发起事件，唤醒目标聊天流
- ``VoiceCallHistoryHandler``：通话结束后把通话内容补回对话链
"""

from .proactive_handler import PROACTIVE_TRIGGER_EVENT, ProactiveHandler
from .voice_call_history_handler import VoiceCallHistoryHandler

__all__ = [
    "PROACTIVE_TRIGGER_EVENT",
    "ProactiveHandler",
    "VoiceCallHistoryHandler",
]
