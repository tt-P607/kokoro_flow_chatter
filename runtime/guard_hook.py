"""KFC 与 response_guard 之间的薄接入层。

主循环在取得工具调用结果之后、执行决策之前调用 :func:`check_response_guard`。
本模块只负责通过 Service API 查找 ``response_guard`` 服务并翻译返回值，不含
任何检测规则：规则全部住在 ``response_guard`` 插件内，KFC 与其它 chatter
共用同一份检测核心。

``response_guard`` 未安装、未启用、调用异常时一律视为未命中，聊天优先。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.service_api import get_service, get_service_class

logger = get_logger("kfc_guard")

GUARD_SIGNATURE = "response_guard:service:response_guard"
"""response_guard 对外 Service 的组件签名。"""


@dataclass(frozen=True, slots=True)
class GuardCheck:
    """一次响应守卫检测的结果。

    Attributes:
        blocked: 是否判定为模型层安全拒答。
        evidence: 判定依据的证据标签。
    """

    blocked: bool
    evidence: tuple[str, ...] = ()


_NOT_BLOCKED = GuardCheck(blocked=False)
"""未命中结果，供服务缺失与异常路径复用。"""


def guard_available() -> bool:
    """response_guard 的 Service 是否已注册。"""
    return get_service_class(GUARD_SIGNATURE) is not None


async def check_response_guard(
    response: Any,
    *,
    request_name: str,
    retry_index: int = 0,
) -> GuardCheck:
    """调用 response_guard 检测本轮响应。

    Args:
        response: 本轮 LLM 响应链。
        request_name: 当前 LLM 请求名，供 Guard 统计与日志识别。
        retry_index: 本轮的守卫重试序号。

    Returns:
        GuardCheck: 检测结果；任何异常都退化为未命中。
    """
    try:
        if not guard_available():
            return _NOT_BLOCKED
        service = get_service(GUARD_SIGNATURE)
        if service is None:
            return _NOT_BLOCKED
        result = await service.inspect(
            reply_text=_as_text(getattr(response, "message", None)),
            reasoning_text=_as_text(getattr(response, "reasoning_content", None)),
            reasoning_parts=tuple(getattr(response, "reasoning_parts", None) or ()),
            tool_calls=tuple(getattr(response, "call_list", None) or ()),
            request_name=request_name,
            retry_index=retry_index,
        )
    except Exception as error:
        logger.error(f"response_guard 调用失败，按未命中处理: {error}", exc_info=True)
        return _NOT_BLOCKED

    evidence = getattr(result, "evidence", None) or ()
    return GuardCheck(
        blocked=bool(getattr(result, "blocks", False)),
        evidence=tuple(str(item) for item in evidence),
    )


def _as_text(value: Any) -> str | None:
    """把响应字段规范成可检测的文本。"""
    return value if isinstance(value, str) and value.strip() else None
