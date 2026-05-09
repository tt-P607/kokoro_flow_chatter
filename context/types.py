"""KFC 上下文层共享类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ContextOwner = Literal[
    "policy",
    "self_state",
    "user_state",
    "relationship_state",
    "scene_evidence",
    "notice",
]
ContextScope = Literal["turn", "session", "persistent"]
StatePatchTarget = Literal["self_state", "user_state", "relationship_state", "scene_state"]
StatePatchOp = Literal["set", "merge", "append", "remove"]


@dataclass(slots=True)
class ContextContribution:
    """第三方或内部提交的结构化上下文贡献。"""

    source: str
    owner: ContextOwner
    scope: ContextScope
    priority: int
    content: str
    ttl_turns: int | None = None
    evidence_only: bool = False


@dataclass(slots=True)
class StatePatch:
    """跨轮持久化申请的统一描述。"""

    source: str
    target: StatePatchTarget
    op: StatePatchOp
    path: str
    value: Any
    reason: str


@dataclass(slots=True)
class ContextPlan:
    """单轮上下文规划结果。"""

    user_text: str
    contributions: list[ContextContribution] = field(default_factory=list)
    # 仅含原始消息内容，不含末尾强调指令/平台信息，用于链持久化
    chain_text: str = ""


@dataclass(slots=True)
class InitialContextPlan:
    """初始上下文规划结果。"""

    system_extra_vars: dict[str, str] = field(default_factory=dict)
    history_summary: str = ""
    history_before_ts: float | None = None