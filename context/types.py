"""KFC 上下文层共享类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.app.plugin_system.types import Content

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
    不进入持久 transcript。
    """

    source: str
    """来源标识，用于渲染时区分 legacy 与结构化贡献。"""

    owner: ContextOwner
    priority: int
    content: str | list[Content]
    """纯文本或 kernel 标准 Content parts；列表允许携带 Image 等真实多模态内容。"""


@dataclass(slots=True)
class ContextPlan:
    """单轮用户输入的上下文规划结果。"""

    user_text: str
    """当前真实用户输入，可作为持久 transcript 记录。"""

    contributions: list[ContextContribution] = field(default_factory=list)


@dataclass(slots=True)
class InitialContextPlan:
    """``execute()`` 启动时的初始上下文规划结果。"""

    system_extra_vars: dict[str, str] = field(default_factory=dict)
    """注入系统提示词模板的额外变量。"""

    history_summary: str = ""
    """近期记忆摘要；它只是动态背景，不是持久对话历史。"""
