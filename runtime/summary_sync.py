"""KFC 摘要热更新。

近期记忆压缩在后台任务中异步完成。若主循环仍在同一次 ``execute()``
里运行，其上下文中的摘要就还是压缩前的旧版本。本模块负责在检测到
摘要变化时，就地替换上下文中的摘要段落，让新记忆立即生效而无需
重建整条链。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.types import ROLE, Text

from ..context.renderer import SECTION_SEPARATOR
from ..context.sources.history_source import build_history_summary_payload

if TYPE_CHECKING:
    from src.app.plugin_system.types import ChatStream

logger = get_logger("kfc_summary_sync")

_SUMMARY_MARKER_PREFIX = "【你对"
_SUMMARY_MARKER_SUFFIX = "的近期记忆】"
"""摘要段落的定位标记，与 ``build_history_summary_payload`` 的输出一致。"""

_CHANNEL_SECTION_INDEX = 1
"""摘要首次生成时的插入位置——紧跟通道信息段之后。"""


class SummarySynchronizer:
    """跟踪摘要变化并把新摘要热更新进上下文。"""

    def __init__(self, baked_summary: str) -> None:
        """初始化同步器。

        Args:
            baked_summary: 构建初始上下文时"烧入"链中的摘要。
        """
        self._baked_summary = baked_summary

    def sync_if_changed(
        self,
        response: Any,
        chat_stream: ChatStream,
        current_summary: str,
    ) -> bool:
        """检测摘要变化并在必要时热更新上下文。

        Args:
            response: 持有 ``payloads`` 列表的响应对象。
            chat_stream: 当前聊天流，用于渲染摘要标题。
            current_summary: 会话中的最新摘要。

        Returns:
            bool: 是否执行了热更新。
        """
        if current_summary == self._baked_summary:
            return False

        self._baked_summary = current_summary
        if not current_summary:
            return False

        if _replace_summary_section(response, chat_stream, current_summary):
            logger.info("近期记忆摘要已热更新到 LLM 上下文")
            return True
        return False


def _replace_summary_section(
    response: Any,
    chat_stream: ChatStream,
    new_summary: str,
) -> bool:
    """在动态 USER payload 中替换或插入摘要段落。

    动态 payload 的结构为「通道信息 + 摘要 + 历史叙事」，各段以
    ``SECTION_SEPARATOR`` 分隔。通过摘要标记定位既有段落；若原先没有
    摘要（首次生成），则插入到通道信息之后。

    Returns:
        bool: 是否成功写回。
    """
    summary_payload = build_history_summary_payload(chat_stream, new_summary)
    if summary_payload is None:
        return False

    new_summary_text = "".join(
        item.text for item in summary_payload.content if isinstance(item, Text)
    )
    if not new_summary_text:
        return False

    payloads = response.payloads
    dynamic_payload = next(
        (payload for payload in payloads if payload.role == ROLE.USER), None
    )
    if dynamic_payload is None:
        return False

    content = dynamic_payload.content
    if not isinstance(content, list):
        content = [content]
    old_text = "".join(item.text for item in content if isinstance(item, Text))
    if not old_text:
        return False

    sections = old_text.split(SECTION_SEPARATOR)
    for index, section in enumerate(sections):
        if _SUMMARY_MARKER_PREFIX in section and _SUMMARY_MARKER_SUFFIX in section:
            sections[index] = new_summary_text
            break
    else:
        insert_at = min(_CHANNEL_SECTION_INDEX, len(sections))
        sections.insert(insert_at, new_summary_text)

    dynamic_payload.content = [Text(SECTION_SEPARATOR.join(sections))]
    return True
