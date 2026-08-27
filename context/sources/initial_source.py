"""KFC 初始上下文 source。

从配置与会话状态中提取 ``execute()`` 启动所需的系统模板变量、
近期记忆摘要与融合叙事截断点。
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING

from ..types import InitialContextPlan

if TYPE_CHECKING:
    from src.app.plugin_system.types import ChatStream

    from ...config import KFCConfig
    from ...session import KFCSession

_SCENE_ANCHOR_INFO = (
    "# 场景锚定\n"
    "- 平台、聊天类型、bot ID 只是通道参数，不构成任何实体场景证据。\n"
    "- 除非对话中出现明确证据，否则不要脑补手机、屏幕、房间等物理环境细节。\n"
    "- 进行角色扮演时，优先依据双方关系、语境和时间来组织描写。"
)
"""场景锚定说明。KFC 不维护结构化场景状态，仅以固定约束抑制模型
对物理环境的无据脑补。"""


def build_initial_context_plan(
    *,
    chat_stream: ChatStream,
    config: KFCConfig,
    session: KFCSession,
) -> InitialContextPlan:
    """构建初始上下文规划结果。

    Args:
        chat_stream: 当前聊天流（保留参数以对齐 source 接口）。
        config: KFC 配置。
        session: 当前会话。

    Returns:
        InitialContextPlan: 系统模板变量与动态记忆摘要。
    """
    _ = chat_stream
    from ...prompts.templates import KFC_REPLY_MODE_TOOL_CALLING

    wait_instruction = config.general.wait_instruction.replace(
        "{max_wait_seconds}", str(int(config.wait.max_seconds))
    )
    extra_vars: dict[str, str] = {
        "reply_mode_instruction": KFC_REPLY_MODE_TOOL_CALLING.format(
            segment_instruction=config.general.segment_instruction,
            wait_instruction=wait_instruction,
        ),
        "scene_state_info": _SCENE_ANCHOR_INFO,
    }

    custom_prompt = config.general.custom_decision_prompt.strip()
    if custom_prompt:
        extra_vars["custom_decision_prompt"] = f"# 决策指导\n{custom_prompt}"

    scheduled_info = _build_scheduled_proactive_info(session)
    if scheduled_info:
        extra_vars["scheduled_proactive_info"] = scheduled_info

    return InitialContextPlan(
        system_extra_vars=extra_vars,
        history_summary=session.history_summary,
    )


def _build_scheduled_proactive_info(session: KFCSession) -> str:
    """渲染当前主动发起预约状态；无预约时返回空串。"""
    scheduled_at = session.scheduled_proactive_at
    if not scheduled_at:
        return ""

    remaining_minutes = max(0.0, (scheduled_at - time.time()) / 60)
    scheduled_time = datetime.fromtimestamp(scheduled_at).strftime("%H:%M")
    reason = session.scheduled_proactive_reason.strip()
    reason_text = f"，理由：{reason}" if reason else ""
    return (
        "# 当前预约状态\n"
        f"你已预约在 **{scheduled_time}**（约 {remaining_minutes:.0f} 分钟后）"
        f"主动发起{reason_text}。\n"
        "如需修改，可重新调用 `action-schedule_proactive` 工具"
        "（新预约会覆盖旧的；传 delay_minutes=0 可取消预约）。"
    )
