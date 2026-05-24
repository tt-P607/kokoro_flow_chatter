"""KFC 工具调用执行层。

执行层只特殊解释 KFC 自身控制动作，其余 action/tool/agent 统一交给
框架 ``run_tool_call`` 调度，避免重复维护通用工具执行逻辑。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.types import LLMPayload, ROLE, ToolRegistry, ToolResult

from ..config import KFCConfig
from ..models import DO_NOTHING, KFC_REPLY, ToolCallResult
from ..parser import _calculate_typing_delay, _parse_content_segments, extract_metadata
from ..protocol.tool_call_adapter import DecisionDraft, DecisionDraftCall

logger = get_logger("kfc_decision_executor")


def _is_kfc_control_call(draft_call: DecisionDraftCall) -> bool:
    """判断是否为 KFC 自身需要特殊解释的控制动作。"""
    return draft_call.normalized_name in {KFC_REPLY, DO_NOTHING}


async def execute_decision_draft(
    draft: DecisionDraft,
    response: Any,
    usable_map: ToolRegistry,
    trigger_msg: Any | None,
    config: KFCConfig,
    *,
    execute_reply_fn: Callable[[str, KFCConfig, Any | None, str], Awaitable[bool]],
    run_tool_call_fn: Callable[[list[Any], Any, Any, Any | None], Awaitable[list[tuple[bool, bool]]]],
    pre_execute_hook: Callable[[ToolCallResult], None] | None = None,
) -> ToolCallResult:
    """执行 DecisionDraft 并产出 ToolCallResult。"""
    result = ToolCallResult()
    is_first_reply = True
    pending_framework_calls: list[Any] = []

    async def flush_pending_framework_calls() -> None:
        """批量交由框架执行暂存的普通工具/action/agent。"""
        if not pending_framework_calls:
            return
        current_pending = list(pending_framework_calls)
        pending_framework_calls.clear()
        logger.debug(f"[KFC] 交由框架执行 {len(current_pending)} 个普通工具/action/agent")
        call_results = await run_tool_call_fn(current_pending, response, usable_map, trigger_msg)
        for call, (_appended, success) in zip(current_pending, call_results, strict=False):
            if not success:
                logger.warning(f"[KFC] 工具/action/agent {call.name} 执行失败或被跳过")

    for draft_call in draft.calls:
        if _is_kfc_control_call(draft_call):
            extract_metadata(result, dict(draft_call.args))
            break

    for draft_call in draft.calls:
        args = dict(draft_call.args)
        reason = args.pop("reason", "未提供原因")
        normalized_name = draft_call.normalized_name
        logger.info(f"LLM 调用 {draft_call.raw_name}，原因: {reason}")

        if normalized_name == KFC_REPLY:
            await flush_pending_framework_calls()
            result.has_reply = True
            extract_metadata(result, args)
            content_raw = args.get("content", "")
            segments = _parse_content_segments(content_raw)

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
                if not send_ok:
                    break

            action_dict: dict[str, Any] = {"type": normalized_name}
            action_dict.update(args)
            action_dict["content"] = segments
            result.actions.append(action_dict)
            response.add_payload(
                LLMPayload(
                    ROLE.TOOL_RESULT,
                    ToolResult(
                        value="已发送" if send_ok else "发送失败",
                        call_id=draft_call.call_id,
                        name=draft_call.raw_name,
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
                    ToolResult(
                        value="已选择不回复",
                        call_id=draft_call.call_id,
                        name=draft_call.raw_name,
                    ),
                )
            )
            continue

        result.has_third_party = True
        if draft_call.raw_name.startswith(("agent-", "tool-")):
            result.has_info_tool = True
        action_dict = {"type": normalized_name}
        action_dict.update(args)
        result.actions.append(action_dict)
        pending_framework_calls.append(draft_call.raw_call)

    await flush_pending_framework_calls()

    if pre_execute_hook is not None:
        pre_execute_hook(result)

    if config.debug.show_prompt:
        call_names = [call.raw_name for call in draft.calls]
        logger.debug(f"[KFC] LLM 响应: tool_calls={len(call_names)} {call_names}")

    return result
