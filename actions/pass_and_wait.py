"""完成当前动作后登记等待。

与 ``do_nothing`` 的区别：``do_nothing`` 表示"这轮不回复"，
``pass_and_wait`` 表示"该做的都做完了，现在等对方"。

控制动作：执行层直接消费其参数并写回执，``execute()`` 不会在 KFC 主
流程中被真正调用，仅作为 schema 的形式入口存在。
"""

from __future__ import annotations

from typing import Annotated

from src.app.plugin_system.base import BaseAction

from .reply import force_kfc_metadata_required


class PassAndWaitAction(BaseAction):
    """完成本轮动作后登记等待点。"""

    name: str = "pass_and_wait"
    associated_types: list[str] = ["text"]
    description: str = (
        "完成本轮所有动作后，登记一个等待点。"
        "可以在 action-kfc_reply 之后调用，表示发完消息后继续等待对方回复；"
        "也可以单独调用，表示本轮不回复但保持等待状态。"
        "与 action-do_nothing 的区别：action-do_nothing 用于主动选择不回复，"
        "action-pass_and_wait 用于已完成动作后的等待。"
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
            "**必填**。你此刻的内心想法，描述你为什么要等待。",
        ] = "",
        expected_reaction: Annotated[
            str,
            "**必填**。你预期的对方反应。",
        ] = "",
        max_wait_seconds: Annotated[
            float,
            "**必填**。等待对方回复的最长时间（秒），0 表示等待新消息。",
        ] = 0.0,
        mood: Annotated[
            str,
            "**必填**。你当前的心情。",
        ] = "",
    ) -> tuple[bool, str]:
        """返回执行回执。

        全部参数由执行层提取用于状态记录，本方法不使用它们。

        Returns:
            tuple: ``(True, "已登记等待")``。
        """
        _ = (thought, expected_reaction, max_wait_seconds, mood)
        return True, "已登记等待"
