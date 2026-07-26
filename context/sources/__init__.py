"""KFC 上下文来源。

每个 source 负责一类上下文的取数与文本渲染，彼此独立、无共享状态：

- ``history_source``：聊天记录、心理活动流、存档对话链
- ``initial_source``：启动时的系统模板变量
- ``memo_source``：私人备忘录
- ``plugin_source``：第三方插件通过事件提交的上下文
"""

from .history_source import (
    build_channel_payload,
    build_current_time_payload,
    build_fused_narrative,
    build_history_summary_payload,
    restore_chain_payloads,
)
from .initial_source import build_initial_context_plan
from .memo_source import build_memo_contribution
from .plugin_source import collect_plugin_turn_contributions

__all__ = [
    "build_channel_payload",
    "build_current_time_payload",
    "build_fused_narrative",
    "build_history_summary_payload",
    "build_initial_context_plan",
    "build_memo_contribution",
    "collect_plugin_turn_contributions",
    "restore_chain_payloads",
]
