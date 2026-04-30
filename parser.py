"""KFC 工具调用解析器。

当前主链只保留原生 tool calling。
本模块负责执行已标准化的 call_list，并回传 ToolCallResult。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, TYPE_CHECKING

from src.app.plugin_system.api.log_api import get_logger
from src.kernel.llm import LLMPayload, ROLE, ToolResult

from .models import KFC_REPLY, DO_NOTHING, ToolCallResult

if TYPE_CHECKING:
    from src.kernel.llm import ToolRegistry

    from .config import KFCConfig

logger = get_logger("kfc_parser")


def _normalize_call_name(name: str) -> str:
    """归一化工具调用名称。"""
    if not name:
        return ""
    if ":" in name:
        return name.rsplit(":", 1)[-1]
    for prefix in ("action-", "tool-", "agent-"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _extract_args(raw_args: Any) -> dict[str, Any]:
    """提取工具参数字典，兼容字符串 JSON。"""
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _ensure_call_id(call: Any) -> str:
    """确保工具调用具备稳定的 call_id。"""
    current_id = getattr(call, "id", None)
    if isinstance(current_id, str) and current_id:
        return current_id
    generated_id = f"call_{uuid.uuid4().hex[:8]}"
    try:
        setattr(call, "id", generated_id)
    except Exception:
        try:
            object.__setattr__(call, "id", generated_id)
        except Exception:
            pass
    resolved_id = getattr(call, "id", None)
    if isinstance(resolved_id, str) and resolved_id:
        return resolved_id
    return generated_id


def extract_metadata(result: ToolCallResult, args: dict[str, Any]) -> None:
    """从工具调用参数中提取元数据到 ToolCallResult。"""
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
    execute_reply_fn: Callable[[str, KFCConfig, Any | None, str], Awaitable[bool]],
    run_tool_call_fn: Callable[[list[Any], Any, Any, Any | None], Awaitable[list[tuple[bool, bool]]]],
    pre_execute_hook: Callable[[ToolCallResult], None] | None = None,
) -> ToolCallResult:
    """遍历 LLM 返回的 call_list，提取元数据并执行动作。

    - kfc_reply: 等待前面的第三方工具完成 → 分段发送文本 → 回传 ToolResult
    - do_nothing: 提取元数据 + 回传 ToolResult
    - 其他: 批量交由 run_tool_call_fn（BaseChatter.run_tool_call）执行

    Args:
        response: LLM 响应对象
        usable_map: 工具注册表
        trigger_msg: 触发消息
        config: KFC 配置
        execute_reply_fn: 回复执行回调
        run_tool_call_fn: 框架内置批量工具执行回调（BaseChatter.run_tool_call）
        pre_execute_hook: 所有动作执行完毕后的汇总日志钩子

    Returns:
        ToolCallResult: 结构化的解析结果
    """
    result = ToolCallResult()
    is_first_reply = True
    pending_third_party_calls: list[Any] = []

    async def flush_pending_third_party() -> None:
        """批量交由框架执行暂存的第三方工具。"""
        if not pending_third_party_calls:
            return
        current_pending = list(pending_third_party_calls)
        pending_third_party_calls.clear()
        logger.debug(f"[KFC] 批量执行 {len(current_pending)} 个第三方工具")
        call_results = await run_tool_call_fn(current_pending, response, usable_map, trigger_msg)
        for call, (appended, success) in zip(current_pending, call_results, strict=False):
            if not success:
                logger.warning(f"[KFC] 工具 {call.name} 执行失败或被跳过")

    # 预处理：提前提取元数据用于日志展示
    if response.call_list:
        for call in response.call_list:
            args = _extract_args(call.args)
            if _normalize_call_name(call.name) in (KFC_REPLY, DO_NOTHING):
                extract_metadata(result, args)
                break

    for call in response.call_list or []:
        args = _extract_args(call.args)
        call_id = _ensure_call_id(call)
        reason = args.pop("reason", "未提供原因")
        normalized_name = _normalize_call_name(call.name)
        logger.info(f"LLM 调用 {call.name}，原因: {reason}")

        if normalized_name == KFC_REPLY:
            await flush_pending_third_party()

            result.has_reply = True
            extract_metadata(result, args)
            content_raw = args.get("content", "")
            if isinstance(content_raw, list):
                segments = [str(s).strip() for s in content_raw if str(s).strip()]
            elif isinstance(content_raw, str):
                stripped = content_raw.strip()
                segments = [stripped] if stripped else []
            else:
                segments = []

            send_ok = True
            for segment in segments:
                if not is_first_reply:
                    delay = _calculate_typing_delay(segment, config)
                    if delay > 0:
                        await asyncio.sleep(delay)
                is_first_reply = False
                seg_reply_to = args.get("reply_to", "") or ""
                send_ok = await execute_reply_fn(segment, config, trigger_msg, seg_reply_to)
                args.pop("reply_to", None)
                seg_reply_to = ""
                if not send_ok:
                    break

            action_dict: dict[str, Any] = {"type": normalized_name}
            action_dict.update(args)
            action_dict["content"] = segments
            result.actions.append(action_dict)

            response.add_payload(
                LLMPayload(
                    ROLE.TOOL_RESULT,
                    ToolResult(  # type: ignore[arg-type]
                        value="已发送" if send_ok else "发送失败",
                        call_id=call_id,
                        name=call.name,
                    ),
                )
            )
            continue

        if normalized_name == DO_NOTHING:
            result.has_do_nothing = True
            extract_metadata(result, args)
            action_dict = {"type": normalized_name}
            action_dict.update(args)
            result.actions.append(action_dict)

            response.add_payload(
                LLMPayload(
                    ROLE.TOOL_RESULT,
                    ToolResult(  # type: ignore[arg-type]
                        value="已选择不回复",
                        call_id=call_id,
                        name=call.name,
                    ),
                )
            )
            continue

        # 第三方工具：暂存，遇到 kfc_reply 或循环结束时批量执行
        result.has_third_party = True
        if call.name.startswith(("agent-", "tool-")):
            result.has_info_tool = True
        action_dict = {"type": normalized_name}
        action_dict.update(args)
        result.actions.append(action_dict)
        pending_third_party_calls.append(call)

    await flush_pending_third_party()

    if pre_execute_hook is not None:
        pre_execute_hook(result)

    if config.debug.show_prompt:
        call_names = [c.name for c in response.call_list] if response.call_list else []
        logger.debug(f"[KFC] LLM 响应: tool_calls={len(call_names)} {call_names}")

    return result


def _calculate_typing_delay(content: str, config: KFCConfig) -> float:
    """根据文本长度计算模拟打字延迟（秒）。"""
    reply_cfg = config.reply
    chars_per_sec = reply_cfg.typing_chars_per_sec
    if chars_per_sec <= 0:
        return 0.0
    base_delay = len(content) / chars_per_sec
    return max(reply_cfg.typing_delay_min, min(base_delay, reply_cfg.typing_delay_max))
