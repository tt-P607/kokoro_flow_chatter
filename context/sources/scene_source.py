"""KFC 场景状态 source。"""

from __future__ import annotations

from typing import Any

from ...domain.scene_state import SceneState


def build_scene_state_info(
    *,
    chat_stream: Any,
    scene_state: SceneState | None,
) -> str:
    """把 SceneState 渲染为系统提示词中的场景状态说明。"""
    state = scene_state or SceneState()
    platform = chat_stream.platform or "unknown"
    chat_type = chat_stream.chat_type or "unknown"
    bot_id = chat_stream.bot_id or ""
    social_channel = state.social_channel.strip() or f"{platform}/{chat_type}"
    location_type = state.location_type.strip() or "unknown"
    device_assumption = "允许" if state.device_assumption_allowed else "禁止"

    evidence_lines = [
        (
            f"- {item.content}"
            f"（来源：{item.source}，类型：{item.kind}，置信度：{item.confidence:.2f}）"
        )
        for item in state.evidence
        if item.content.strip()
    ]
    if not evidence_lines:
        evidence_lines = [
            "- 暂无已确认的场景证据。不要把平台、聊天类型、bot 身份、能力提醒或人格设定当作实体场景证据。"
        ]

    if state.certainty == "unknown":
        inference_line = "- 当前没有足够证据确认任何具体生活场景；必须保持为 unknown。"
    elif location_type == "unknown":
        inference_line = "- 已有部分场景证据，但仍不能确认具体地点类型。"
    else:
        inference_line = (
            f"- 当前只能确认到“{location_type}”这一场景类型，"
            "不能外推更多未给出的实体环境。"
        )

    bot_id_line = f"，bot_id={bot_id}" if bot_id else ""

    return (
        "# 场景状态\n"
        f"- 当前场景确定度：{state.certainty}\n"
        f"- 社交通道：{social_channel}（platform={platform}，chat_type={chat_type}{bot_id_line}）\n"
        f"- 当前可确认的场景类型：{location_type}\n"
        f"- 是否允许设备假设：{device_assumption}\n"
        f"{inference_line}\n"
        "- 已确认的场景证据：\n"
        + "\n".join(evidence_lines)
    )