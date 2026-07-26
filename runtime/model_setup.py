"""KFC 模型集解析。

按配置优先级组装本次对话使用的模型集：显式 ``models`` 列表优先，
其中的模型按顺序串成 fallback 链；列表为空或全部未注册时，回退到
``model_task`` 对应的任务模型。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.llm_api import (
    get_model_set_by_name,
    get_model_set_by_task,
)
from src.app.plugin_system.api.log_api import get_logger

if TYPE_CHECKING:
    from ..config import KFCConfig

logger = get_logger("kfc_model_setup")


def resolve_model_set(config: KFCConfig) -> Any | None:
    """解析本次对话使用的模型集。

    Args:
        config: KFC 配置。

    Returns:
        Any | None: 模型集；无任何可用模型时返回 ``None``。
    """
    general = config.general
    if general.models:
        model_set = _combine_named_models(
            general.models,
            temperature=general.temperature,
            max_tokens=general.max_tokens,
        )
        if model_set is not None:
            return model_set
        logger.warning(
            f"models 中的模型均未注册: {general.models}，"
            f"回退到任务模型 '{general.model_task}'"
        )

    return get_model_set_by_task(general.model_task)


def _combine_named_models(
    model_names: list[str],
    *,
    temperature: float,
    max_tokens: int,
) -> Any | None:
    """把具名模型按顺序串成 fallback 链。

    Args:
        model_names: 配置中的模型名列表。
        temperature: 采样温度。
        max_tokens: 最大输出 token 数。

    Returns:
        Any | None: 合并后的模型集；全部未注册时返回 ``None``。
    """
    valid_parts = [
        part
        for part in (
            get_model_set_by_name(
                model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for model_name in model_names
        )
        if part
    ]
    if not valid_parts:
        return None

    model_set = valid_parts[0]
    for part in valid_parts[1:]:
        model_set = model_set + part
    return model_set
