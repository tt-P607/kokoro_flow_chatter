"""KFC 工具调用适配层。

本模块只把 LLM 返回的 tool call 转换为结构化草稿，不执行任何副作用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.app.plugin_system.types import ToolCall

from ..models import DO_NOTHING, KFC_REPLY
from ..parser import _ensure_call_id


@dataclass(slots=True)
class DecisionDraftCall:
    """单个规范化工具调用。"""

    raw_call: ToolCall
    call_id: str
    raw_name: str
    normalized_name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DecisionDraft:
    """模型返回 tool calls 的无副作用草稿。"""

    calls: list[DecisionDraftCall] = field(default_factory=list)

    @property
    def has_calls(self) -> bool:
        """是否包含任何工具调用。"""
        return bool(self.calls)


def normalize_call_name(name: str) -> str:
    """归一化工具调用名称。"""
    if not name:
        return ""
    if ":" in name:
        return name.rsplit(":", 1)[-1]
    for prefix in ("action-", "tool-", "agent-"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def extract_call_args(raw_args: Any) -> dict[str, Any]:
    """提取工具参数字典，兼容字符串 JSON。"""
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def build_decision_draft(call_list: list[ToolCall] | None) -> DecisionDraft:
    """将原始 call_list 转为无副作用 DecisionDraft。"""
    draft = DecisionDraft()
    for call in call_list or []:
        call_id = _ensure_call_id(call)
        raw_name = call.name
        draft.calls.append(
            DecisionDraftCall(
                raw_call=call,
                call_id=call_id,
                raw_name=raw_name,
                normalized_name=normalize_call_name(raw_name),
                args=extract_call_args(call.args),
            )
        )
    return draft


def is_kfc_control_call(call: DecisionDraftCall) -> bool:
    """判断调用是否为 KFC 自有控制动作。"""
    return call.normalized_name in {KFC_REPLY, DO_NOTHING}
