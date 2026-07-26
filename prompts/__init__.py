"""KFC 提示词层。

``templates`` 存放静态模板文本，``modules`` 负责注册模板并按运行时
状态渲染动态提示词（主动发起、超时决策）。
"""

from .modules import (
    build_mental_log_hint,
    build_proactive_context,
    build_timeout_payload,
    register_kfc_prompts,
)

__all__ = [
    "build_mental_log_hint",
    "build_proactive_context",
    "build_timeout_payload",
    "register_kfc_prompts",
]
