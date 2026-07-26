"""KFC 运行时层。

承载对话主循环及其配套设施。``orchestrator`` 是唯一入口，其余模块各
负责一件事：

- ``model_setup`` / ``context_builder``：启动前的模型与上下文准备
- ``turn_controller``：单轮的输入准备与决策提交
- ``phase_machine`` / ``unread_policy``：相位与未读消息的判定规则
- ``payload_hygiene`` / ``summary_sync``：上下文链的清理与热更新
- ``request_view`` / ``interrupt_controller`` / ``input_status``：发送侧能力
"""

from .interrupt_controller import send_interruptable_response
from .orchestrator import execute_orchestrator
from .request_view import build_request_view
from .turn_controller import (
    TurnControlResult,
    TurnInputResult,
    commit_turn_decision,
    prepare_turn_input,
)

__all__ = [
    "TurnControlResult",
    "TurnInputResult",
    "build_request_view",
    "commit_turn_decision",
    "execute_orchestrator",
    "prepare_turn_input",
    "send_interruptable_response",
]
