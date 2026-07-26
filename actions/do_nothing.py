"""选择不回复。

控制动作：执行层直接消费其参数并写回执，``execute()`` 不会在 KFC 主
流程中被真正调用，仅作为 schema 的形式入口存在。
"""

from __future__ import annotations

from typing import Annotated

from src.app.plugin_system.base import BaseAction

from .reply import force_kfc_metadata_required


class DoNothingAction(BaseAction):
    """主动选择不做任何回复。"""

    name: str = "do_nothing"
    associated_types: list[str] = ["text"]
    description: str = (
        "决定不做任何回复。当对方的消息不需要回应、纯表情、"
        "或者你选择已读不回时使用。"
        "**调用时必须明确给出 thought / expected_reaction / max_wait_seconds / mood "
        "这四个字段，承载你这次决策的内心活动、对对方反应的预期、"
        "等待时长和当前情绪。**"
    )
    chatter_allow: list[str] = ["kokoro_flow_chatter"]

    @classmethod
    def to_schema(cls) -> dict:  # type: ignore[override]
        """生成 schema，并强制元数据字段必填。"""
        return force_kfc_metadata_required(super().to_schema())

    async def execute(
        self,
        thought: Annotated[
            str,
            "**必填**。你此刻的内心想法，描述你为什么选择不回复。",
        ] = "",
        expected_reaction: Annotated[
            str,
            "**必填**。你预期的对方反应。",
        ] = "",
        max_wait_seconds: Annotated[
            float,
            "**必填**。是否继续等待对方（秒），0 表示不等待。",
        ] = 0.0,
        mood: Annotated[
            str,
            "**必填**。你当前的心情。",
        ] = "",
    ) -> tuple[bool, str]:
        """返回执行回执。

        全部参数由执行层提取用于状态记录，本方法不使用它们。

        Returns:
            tuple: ``(True, "已选择不回复")``。
        """
        _ = (thought, expected_reaction, max_wait_seconds, mood)
        return True, "已选择不回复"
