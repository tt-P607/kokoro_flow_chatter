"""KFC 上下文层共享类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ContextOwner = Literal[
    "policy",
    "self_state",
    "user_state",
    "relationship_state",
    "notice",
]
"""上下文贡献的归属分区，决定渲染时归入哪个标题段落。"""


@dataclass(slots=True)
class ContextContribution:
    """一条 turn 级上下文贡献。

    由第三方插件通过 ``on_prompt_build`` 提交，或由 KFC 内部 source
    （如备忘录）产出，最终渲染为临时 USER payload 注入本轮请求，
    不进入持久化对话链。
    """

    source: str
    """来源标识，用于渲染时区分 legacy 与结构化贡献。"""

    owner: ContextOwner
    priority: int
    content: str


@dataclass(slots=True)
class ContextPlan:
    """单轮用户输入的上下文规划结果。"""

    user_text: str
    """完整用户提示词，含末尾行为强调指令。"""

    chain_text: str
    """仅含原始消息内容，用于持久化对话链，不含强调指令。"""

    contributions: list[ContextContribution] = field(default_factory=list)


@dataclass(slots=True)
class InitialContextPlan:
    """``execute()`` 启动时的初始上下文规划结果。"""

    system_extra_vars: dict[str, str] = field(default_factory=dict)
    """注入系统提示词模板的额外变量。"""

    history_summary: str = ""
    history_before_ts: float | None = None
    """近期记忆摘要，及融合叙事的截断时间戳（与对话链分界，避免重叠）。"""
