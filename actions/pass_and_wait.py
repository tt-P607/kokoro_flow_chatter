"""PassAndWait 动作。

当 LLM 完成当前轮次的动作（如发送消息、调用工具）后，
希望继续等待对方回复时调用此动作。
与 do_nothing 的区别：do_nothing 表示"不回复"，
pass_and_wait 表示"已完成当前动作，现在等待"。

注：本 Action 在 KFC 主流程中不会被实际调用（parser 层直接消费
其参数），execute() 仅作为 schema 生成的形式入口存在。
"""

from __future__ import annotations

from typing import Annotated

from src.app.plugin_system.base import BaseAction

from .reply import _force_kfc_metadata_required


class PassAndWaitAction(BaseAction):
    """完成当前动作后等待对方回复。"""

    action_name: str = "pass_and_wait"
    associated_types: list[str] = ["text"]
    action_description: str = (
        "完成本轮所有动作后，登记一个等待点。"
        "可以在 kfc_reply 之后调用，表示发完消息后继续等待对方回复；"
        "也可以单独调用，表示本轮不回复但保持等待状态。"
        "与 do_nothing 的区别：do_nothing 用于主动选择不回复，"
        "pass_and_wait 用于已完成动作后的等待。"
        "**调用时必须明确给出 thought / expected_reaction / max_wait_seconds / mood "
        "这四个字段，承载你这次决策的内心活动、对对方反应的预期、等待时长和当前情绪。**"
    )
    chatter_allow: list[str] = ["kokoro_flow_chatter"]

    @classmethod
    def to_schema(cls) -> dict:  # type: ignore[override]
        """把 KFC 元数据字段在 schema 里标记为 required。"""
        return _force_kfc_metadata_required(super().to_schema())

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
        """执行等待逻辑。

        参数由 chatter 策略层提取用于状态记录，
        action 本身不执行任何操作。

        Returns:
            (True, "已登记等待")
        """
        return True, "已登记等待"
