"""KFC 对话窗口预算与禁止静默裁剪的上下文管理器。"""

from __future__ import annotations

from typing import Any

from src.app.plugin_system.api.llm_api import LLMContextManager
from src.app.plugin_system.types import LLMPayload
from src.kernel.llm.context_budget import compute_effective_context_budget
from src.kernel.llm.token_counter import count_payload_tokens


_ROTATION_RATIO = 0.8


def context_budget(model_set: Any) -> int:
    """返回所有可回退模型中最小的有效窗口。"""
    budgets = [compute_effective_context_budget(model) for model in model_set]
    if not budgets or any(budget is None for budget in budgets):
        raise ValueError("KFC 对话模型必须配置 max_context")
    return min(budgets)


def payload_tokens(payloads: list[LLMPayload], model_set: Any) -> int:
    """按模型集里 token 数最高的模型估算完整请求。"""
    return max(
        count_payload_tokens(payloads, model_identifier=model["model_identifier"])
        for model in model_set
    )


def should_rotate(payloads: list[LLMPayload], model_set: Any) -> bool:
    """在最小有效窗口的 80% 处封存旧回合。"""
    return payload_tokens(payloads, model_set) >= context_budget(model_set) * _ROTATION_RATIO


class KFCContextManager(LLMContextManager):
    """保留框架 reminder 与结构校验，但禁止请求层静默裁剪。"""

    async def prepare_payloads_for_model(
        self,
        payloads: list[LLMPayload],
        model: Any,
        *,
        request: Any = None,
    ) -> list[LLMPayload]:
        """逐个模型检查实际发送的 payload；超限时明确拒绝。"""
        budget = compute_effective_context_budget(model)
        if budget is None:
            raise ValueError("KFC 对话模型必须配置 max_context")
        tokens = count_payload_tokens(
            payloads, model_identifier=model["model_identifier"]
        )
        if tokens > budget:
            raise ValueError(f"KFC 上下文超出模型预算：{tokens} > {budget}")
        return payloads