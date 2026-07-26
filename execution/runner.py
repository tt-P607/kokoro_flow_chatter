"""KFC 单轮决策执行入口。

把"解析草稿 → 执行动作 → 收敛决策"这条固定流水线收在一处，
供 runtime 主循环单点调用。依赖方向严格向下：
``execution → protocol → domain``，protocol 不反向依赖 execution。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..models import ToolCallResult
from ..protocol.decision_parser import build_decision
from ..protocol.tool_call_adapter import build_decision_draft
from .decision_executor import ExecuteReplyFn, RunToolCallFn, execute_decision_draft

if TYPE_CHECKING:
    from src.app.plugin_system.types import ToolRegistry

    from ..config import KFCConfig
    from ..domain.decision import Decision


async def run_decision(
    response: Any,
    usable_map: ToolRegistry,
    trigger_msg: Any | None,
    config: KFCConfig,
    *,
    execute_reply_fn: ExecuteReplyFn,
    run_tool_call_fn: RunToolCallFn,
    pre_execute_hook: Callable[[ToolCallResult], None] | None = None,
) -> Decision:
    """执行模型本轮响应并返回统一决策。

    Args:
        response: 模型响应对象。
        usable_map: 框架工具注册表。
        trigger_msg: 触发本轮的消息。
        config: KFC 配置。
        execute_reply_fn: 发送单段回复的回调。
        run_tool_call_fn: 框架批量工具执行回调。
        pre_execute_hook: 动作执行完毕后的汇总日志钩子。

    Returns:
        Decision: 供主循环消费的统一决策对象。
    """
    call_list = response.call_list or []
    draft = build_decision_draft(call_list)
    result = await execute_decision_draft(
        draft,
        response,
        usable_map,
        trigger_msg,
        config,
        execute_reply_fn=execute_reply_fn,
        run_tool_call_fn=run_tool_call_fn,
        pre_execute_hook=pre_execute_hook,
    )
    return build_decision(result, call_list)
