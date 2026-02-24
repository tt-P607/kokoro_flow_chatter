"""KFC 工具调用解析器。

将 LLM 响应中的 call_list 解析为结构化的 ToolCallResult，
提取元数据（thought、expected_reaction 等）并驱动动作执行。
"""

from __future__ import annotations

import asyncio
from typing import Any, TYPE_CHECKING

from src.app.plugin_system.api.log_api import get_logger
from src.kernel.llm import LLMPayload, ROLE, ToolResult

from .models import KFC_REPLY, DO_NOTHING, ToolCallResult

if TYPE_CHECKING:
    from src.kernel.llm import ToolRegistry

    from .config import KFCConfig

logger = get_logger("kfc_parser")


def _normalize_call_name(name: str) -> str:
    """归一化工具调用名称。

    框架可能返回带组件类型前缀的名称（如 ``action:kfc_reply``），
    需要提取末段以匹配常量定义。

    Args:
        name: 原始工具调用名称

    Returns:
        str: 归一化后的名称（末段）
    """
    return name.rsplit(":", 1)[-1] if ":" in name else name


def extract_metadata(result: ToolCallResult, args: dict[str, Any]) -> None:
    """从工具调用参数中提取元数据到 ToolCallResult。

    后调用的工具会覆盖先前的元数据值。

    Args:
        result: 目标结果对象
        args: 工具调用参数
    """
    if "thought" in args:
        result.thought = args["thought"]
    if "expected_reaction" in args:
        result.expected_reaction = args["expected_reaction"]
    if "max_wait_seconds" in args:
        result.max_wait_seconds = float(args["max_wait_seconds"])
    if "mood" in args:
        result.mood = args["mood"]


async def parse_tool_calls(
    response: Any,
    usable_map: ToolRegistry,
    trigger_msg: Any | None,
    config: KFCConfig,
    *,
    execute_reply_fn: Any,
    run_tool_call_fn: Any,
) -> ToolCallResult:
    """遍历 LLM 返回的 call_list，提取元数据并执行动作。

    - kfc_reply: 提取元数据 + 调用 execute_reply_fn + 回传 ToolResult
    - do_nothing: 提取元数据 + 回传 ToolResult
    - 其他: 通过 run_tool_call_fn 执行第三方工具

    Args:
        response: LLM 响应对象
        usable_map: 工具注册表
        trigger_msg: 触发消息
        config: KFC 配置
        execute_reply_fn: 回复执行回调（chatter._execute_reply 的引用）
        run_tool_call_fn: 第三方工具执行回调（chatter.run_tool_call 的引用）

    Returns:
        ToolCallResult: 结构化的解析结果
    """
    result = ToolCallResult()
    is_first_reply = True

    for call in response.call_list or []:
        args = call.args if isinstance(call.args, dict) else {}
        # 归一化名称：框架可能返回 "action:kfc_reply" 格式
        normalized_name = _normalize_call_name(call.name)

        # 记录动作（所有调用都记入 actions 列表）
        action_dict: dict[str, Any] = {"type": normalized_name}
        action_dict.update(args)
        result.actions.append(action_dict)

        if normalized_name == KFC_REPLY:
            result.has_reply = True
            extract_metadata(result, args)

            content = args.get("content", "")
            logger.debug(
                f"[KFC] kfc_reply args 详情: "
                f"content={repr(content[:100]) if content else '(空)'}, "
                f"all_keys={list(args.keys())}"
            )
            if content:
                # 分段发送：非首条 kfc_reply 前模拟打字延迟
                if not is_first_reply:
                    delay = _calculate_typing_delay(content, config)
                    if delay > 0:
                        logger.debug(f"[KFC] 模拟打字延迟 {delay:.2f}s")
                        await asyncio.sleep(delay)
                is_first_reply = False

                await execute_reply_fn(content, config, trigger_msg)

            response.add_payload(
                LLMPayload(
                    ROLE.TOOL_RESULT,
                    ToolResult(  # type: ignore[arg-type]
                        value="已发送",
                        call_id=call.id,
                        name=call.name,
                    ),
                )
            )

        elif normalized_name == DO_NOTHING:
            result.has_do_nothing = True
            extract_metadata(result, args)

            response.add_payload(
                LLMPayload(
                    ROLE.TOOL_RESULT,
                    ToolResult(  # type: ignore[arg-type]
                        value="已选择不回复",
                        call_id=call.id,
                        name=call.name,
                    ),
                )
            )

        else:
            # 第三方工具：通过 run_tool_call 执行并回传结果
            result.has_third_party = True
            await run_tool_call_fn(call, response, usable_map, trigger_msg)

    # 调试日志
    if config.debug.show_prompt:
        call_names = [c.name for c in response.call_list] if response.call_list else []
        logger.debug(f"[KFC] LLM 响应: tool_calls={len(call_names)} {call_names}")

    return result


def _calculate_typing_delay(content: str, config: KFCConfig) -> float:
    """根据文本长度计算模拟打字延迟（秒）。

    Args:
        content: 要发送的消息文本
        config: KFC 配置

    Returns:
        float: 延迟秒数
    """
    reply_cfg = config.reply
    chars_per_sec = reply_cfg.typing_chars_per_sec
    if chars_per_sec <= 0:
        return 0.0

    base_delay = len(content) / chars_per_sec
    return max(reply_cfg.typing_delay_min, min(base_delay, reply_cfg.typing_delay_max))
