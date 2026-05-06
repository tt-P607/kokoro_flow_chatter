"""KFC 初始上下文的 session/source 规划辅助。"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ...domain.scene_state import SceneState
from .scene_source import build_scene_state_info
from ..types import InitialContextPlan


def build_initial_context_plan(
    *,
    chat_stream: Any,
    config: Any,
    session: Any,
) -> InitialContextPlan:
    """从配置与 session 提取初始上下文规划结果。"""
    from ...prompts.templates import KFC_REPLY_MODE_TOOL_CALLING

    extra_vars: dict[str, str] = {}
    extra_vars["reply_mode_instruction"] = KFC_REPLY_MODE_TOOL_CALLING.format(
        segment_instruction=config.general.segment_instruction,
        wait_instruction=config.general.wait_instruction,
    )

    custom_prompt = str(config.general.custom_decision_prompt or "").strip()
    if custom_prompt:
        extra_vars["custom_decision_prompt"] = f"# 决策指导\n{custom_prompt}"

    extra_vars["scene_state_info"] = build_scene_state_info(
        chat_stream=chat_stream,
        scene_state=session.scene_state or SceneState(),
    )

    sched_at = session.scheduled_proactive_at
    if sched_at:
        remaining_min = max(0.0, (sched_at - time.time()) / 60)
        sched_time_str = datetime.fromtimestamp(sched_at).strftime("%H:%M")
        sched_reason = str(
            session.scheduled_proactive_reason or ""
        ).strip()
        reason_text = f"，理由：{sched_reason}" if sched_reason else ""
        extra_vars["scheduled_proactive_info"] = (
            f"# 当前预约状态\n"
            f"你已预约在 **{sched_time_str}**（约 {remaining_min:.0f} 分钟后）主动发起{reason_text}。\n"
            "如需修改，可重新调用 `schedule_proactive` 工具（新预约会覆盖旧的；传 delay_minutes=0 可取消预约）。"
        )

    history_summary = session.history_summary or ""
    chain_cutoff_ts = session.chain_cutoff_ts or 0.0
    history_before_ts = chain_cutoff_ts if chain_cutoff_ts > 0 else None

    return InitialContextPlan(
        system_extra_vars=extra_vars,
        history_summary=history_summary,
        history_before_ts=history_before_ts,
    )