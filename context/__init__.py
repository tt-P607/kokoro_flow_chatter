"""KFC 上下文层。

分两个阶段：``planner`` 决定本轮上下文由哪些内容组成（产出纯数据），
``renderer`` 把这些数据组装成 LLM payload。具体取数逻辑收在
``sources`` 子包中，每类来源一个模块。
"""

from .planner import (
    plan_followup_contributions,
    plan_initial_context,
    plan_user_turn,
)
from .renderer import (
    build_system_prompt,
    render_initial_context,
    render_turn_contributions,
    render_user_payload,
)
from .types import ContextContribution, ContextOwner, ContextPlan, InitialContextPlan

__all__ = [
    "ContextContribution",
    "ContextOwner",
    "ContextPlan",
    "InitialContextPlan",
    "build_system_prompt",
    "plan_followup_contributions",
    "plan_initial_context",
    "plan_user_turn",
    "render_initial_context",
    "render_turn_contributions",
    "render_user_payload",
]
