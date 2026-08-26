"""KFC 决策对象。

``Decision`` 是执行层与主循环之间的唯一契约：执行层把原始
``ToolCallResult`` 收敛成 Decision，主循环只依据 Decision 的语义属性
决定本轮是回复、续轮、等待还是收口，不再触碰原始 tool call 结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolCallSpec:
    """一次第三方工具调用的规范化描述。"""

    name: str
    """去掉 ``action-`` / ``tool-`` / ``agent-`` 前缀后的末段名。"""

    call_id: str = ""
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProactiveSchedule:
    """模型通过 ``schedule_proactive`` 提交的下次主动发起计划。"""

    delay_minutes: float
    """延迟分钟数；0 表示取消当前预约。"""

    reason: str = ""


@dataclass(slots=True)
class Decision:
    """一轮模型响应经执行后的统一决策结果。"""

    thought: str = ""
    mood: str = ""
    expected_reaction: str = ""
    wait_seconds: float = 0.0
    """模型的内心活动元数据，来自控制动作参数。"""

    actions: list[dict[str, Any]] = field(default_factory=list)
    visible_reply_segments: list[str] = field(default_factory=list)
    """已执行动作快照，以及其中用户实际可见的回复分段。"""

    has_reply_action: bool = False
    chose_silence: bool = False
    has_meaningful_action: bool = False
    has_info_tool_calls: bool = False
    has_failed_tool: bool = False
    """本轮语义标志。``has_info_tool_calls`` 表示存在有返回值的
    ``tool-`` / ``agent-`` 调用，需要续轮让模型消化结果。"""

    third_party_calls: list[ToolCallSpec] = field(default_factory=list)
    proactive_schedule: ProactiveSchedule | None = None

    @property
    def should_reply(self) -> bool:
        """本轮是否产生了回复动作。"""
        return self.has_reply_action

    @property
    def has_third_party_calls(self) -> bool:
        """本轮是否存在第三方工具调用。"""
        return bool(self.third_party_calls)

    @property
    def reply_text(self) -> str:
        """按发送顺序拼接的用户可见回复文本。"""
        return "\n".join(self.visible_reply_segments)


def build_experience_snapshot(
    decision: Decision,
    *,
    result_signal: str,
    wait_seconds: float = 0.0,
    pending_tool_results: bool = False,
) -> dict[str, Any] | None:
    """构造 SCF 可消费的最小结构化回合快照。

    只有无待消化工具结果的最终 Wait/Stop 才代表 actor round 已闭合。
    输出只包含结构状态与动作名，不携带回复文本、参数或内部推理。
    """
    if pending_tool_results or result_signal not in {"wait", "stop"}:
        return None

    has_proactive_plan = (
        decision.proactive_schedule is not None
        and decision.proactive_schedule.delay_minutes > 0
    )
    if result_signal == "wait":
        attention_hint = "waiting_for_user"
    elif has_proactive_plan:
        attention_hint = "proactive_scheduled"
    else:
        attention_hint = "none"

    return {
        "schema_version": 1,
        "actor_round_closed": True,
        "result_signal": result_signal,
        "has_meaningful_action": decision.has_meaningful_action,
        "reply_attempted": decision.has_reply_action,
        "chose_silence": decision.chose_silence,
        "waiting_after_round": result_signal == "wait",
        "wait_seconds": max(0.0, float(wait_seconds)),
        "attention_hint": attention_hint,
        "proactive_plan_exists": has_proactive_plan,
        "attempted_action_names": [
            str(action.get("type", "")) for action in decision.actions if action.get("type")
        ],
        "has_failed_action": decision.has_failed_tool,
        "pending_tool_results": False,
    }
