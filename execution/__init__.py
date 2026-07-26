"""KFC 执行层。

把模型响应转成实际动作：解析 → 执行 → 收敛为 ``Decision``。
本层是唯一产生对外副作用（发消息、调用第三方工具）的地方。
"""

from .decision_executor import (
    ExecuteReplyFn,
    RunToolCallFn,
    calculate_typing_delay,
    execute_decision_draft,
    extract_metadata,
    parse_content_segments,
)
from .runner import run_decision

__all__ = [
    "ExecuteReplyFn",
    "RunToolCallFn",
    "calculate_typing_delay",
    "execute_decision_draft",
    "extract_metadata",
    "parse_content_segments",
    "run_decision",
]
