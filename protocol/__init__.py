"""KFC 协议层。

负责模型响应与 KFC 内部结构之间的双向转换，全部为纯函数：

- ``normalize_response``：把 provider 差异（正文内嵌 tool call）抹平；
- ``build_decision_draft``：把原始 ``call_list`` 归一化为无副作用草稿；
- ``build_decision``：把执行结果收敛为主循环使用的 ``Decision``。

本层不执行任何动作，也不访问 session 或聊天流。
"""

from .decision_parser import build_decision
from .response_normalizer import normalize_response
from .tool_call_adapter import (
    DecisionDraft,
    DecisionDraftCall,
    build_decision_draft,
    ensure_call_id,
    extract_call_args,
    normalize_call_name,
)

__all__ = [
    "DecisionDraft",
    "DecisionDraftCall",
    "build_decision",
    "build_decision_draft",
    "ensure_call_id",
    "extract_call_args",
    "normalize_call_name",
    "normalize_response",
]
