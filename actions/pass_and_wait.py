"""PassAndWait 动作。

当 LLM 完成当前轮次的动作（如发送消息、调用工具）后，
希望继续等待对方回复时调用此动作。
与 do_nothing 的区别：do_nothing 表示"不回复"，
pass_and_wait 表示"已完成当前动作，现在等待"。
"""

from __future__ import annotations

from typing import Annotated

from src.app.plugin_system.base import BaseAction


class PassAndWaitAction(BaseAction):
    """完成当前动作后等待对方回复。"""

    action_name: str = "pass_and_wait"
    action_description: str = (
        "完成本轮所有动作后，登记一个等待点。"
        "可以在 kfc_reply 之后调用，表示发完消息后继续等待对方回复；"
        "也可以单独调用，表示本轮不回复但保持等待状态。"
        "与 do_nothing 的区别：do_nothing 用于主动选择不回复，"
        "pass_and_wait 用于已完成动作后的等待。"
    )
    chatter_allow: list[str] = ["kokoro_flow_chatter"]

    async def execute(
        self,
        thought: Annotated[str, "你此刻的内心想法，描述你为什么要等待"] = "",
        expected_reaction: Annotated[str, "你预期的对方反应"] = "",
        max_wait_seconds: Annotated[
            float, "等待对方回复的最长时间(秒)，0表示等待新消息"
        ] = 0.0,
        mood: Annotated[str, "你当前的心情"] = "",
    ) -> tuple[bool, str]:
        """执行等待逻辑。

        参数由 chatter 策略层提取用于状态记录，
        action 本身不执行任何操作。

        Returns:
            (True, "已登记等待")
        """
        return True, "已登记等待"
