"""KFC 工具调用适配层。

本模块是工具名与参数归一化的**唯一来源**，把 LLM 返回的原始
``call_list`` 转成结构化的 ``DecisionDraft``。全过程无副作用——
实际执行由 ``execution.decision_executor`` 负责。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.app.plugin_system.types import ToolCall

from ..models import DO_NOTHING, KFC_REPLY, PASS_AND_WAIT

_COMPONENT_PREFIXES: tuple[str, ...] = ("action-", "tool-", "agent-")
"""框架组件类型前缀，归一化时需要剥离。"""

_INFO_TOOL_PREFIXES: tuple[str, ...] = ("tool-", "agent-")
"""有返回值的调用前缀——命中时主循环需要续轮让模型消化结果。"""

_KFC_CONTROL_NAMES: frozenset[str] = frozenset(
    {KFC_REPLY, DO_NOTHING, PASS_AND_WAIT}
)
"""由 KFC 执行层特殊解释、不交给框架调度的控制动作。"""


@dataclass(slots=True)
class DecisionDraftCall:
    """单个已归一化的工具调用。"""

    raw_call: ToolCall
    """原始调用对象，交给框架执行时需要原样传回。"""

    call_id: str
    raw_name: str
    normalized_name: str
    args: dict[str, Any] = field(default_factory=dict)

    @property
    def is_kfc_control(self) -> bool:
        """是否为 KFC 自有控制动作。"""
        return self.normalized_name in _KFC_CONTROL_NAMES

    @property
    def is_info_tool(self) -> bool:
        """是否为有返回值的 ``tool-`` / ``agent-`` 类调用。"""
        return self.raw_name.startswith(_INFO_TOOL_PREFIXES)


@dataclass(slots=True)
class DecisionDraft:
    """一次模型响应中全部工具调用的无副作用草稿。"""

    calls: list[DecisionDraftCall] = field(default_factory=list)

    @property
    def has_calls(self) -> bool:
        """是否包含任何工具调用。"""
        return bool(self.calls)


def normalize_call_name(name: str) -> str:
    """归一化工具调用名为末段名。

    先按 ``:`` 取组件签名末段，再剥离 ``action-`` / ``tool-`` / ``agent-``
    前缀，使 ``plugin:action:kfc_reply`` 与 ``action-kfc_reply`` 都归一
    到 ``kfc_reply``。

    Args:
        name: 原始工具名或组件签名。

    Returns:
        str: 归一化后的末段名；输入为空时返回空串。
    """
    if not name:
        return ""
    if ":" in name:
        return name.rsplit(":", 1)[-1]
    for prefix in _COMPONENT_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def extract_call_args(raw_args: Any) -> dict[str, Any]:
    """提取工具参数字典，兼容模型返回的 JSON 字符串。

    Args:
        raw_args: 原始参数，可能是 dict、JSON 字符串或其它类型。

    Returns:
        dict[str, Any]: 参数字典；无法解析为对象时返回空字典。
    """
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except (json.JSONDecodeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def ensure_call_id(call: ToolCall) -> str:
    """确保工具调用具备稳定的 ``call_id``。

    ``ToolCall`` 是 ``frozen=True, slots=True`` 的 dataclass，不能用普通
    赋值写入；但 ``id`` 字段已在 slots 中声明，可以通过
    ``object.__setattr__`` 合法补写。缺失 id 会导致 TOOL_RESULT 无法与
    调用配对，因此这里为其生成一个。

    Args:
        call: 待检查的工具调用。

    Returns:
        str: 原有或新生成的 call id。
    """
    if isinstance(call.id, str) and call.id:
        return call.id
    generated_id = f"call_{uuid.uuid4().hex[:8]}"
    object.__setattr__(call, "id", generated_id)
    return generated_id


def build_decision_draft(call_list: list[ToolCall] | None) -> DecisionDraft:
    """把原始 ``call_list`` 转为无副作用的 ``DecisionDraft``。

    Args:
        call_list: 模型返回的工具调用列表，可为 ``None``。

    Returns:
        DecisionDraft: 归一化后的调用草稿。
    """
    draft = DecisionDraft()
    for call in call_list or []:
        draft.calls.append(
            DecisionDraftCall(
                raw_call=call,
                call_id=ensure_call_id(call),
                raw_name=call.name,
                normalized_name=normalize_call_name(call.name),
                args=extract_call_args(call.args),
            )
        )
    return draft
