"""KFC 工具调用执行层。

执行层只特殊解释 KFC 自有的三个控制动作（``kfc_reply`` /
``do_nothing`` / ``pass_and_wait``），其余 action / tool / agent 一律
批量交给框架 ``run_tool_call`` 调度，避免重复维护通用执行逻辑。

每个被处理的调用都会往 ``response`` 写入一条 TOOL_RESULT，保证
模型在续轮时能看到完整、配对的工具执行回执。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.types import LLMPayload, ROLE, ToolRegistry, ToolResult

from ..models import DO_NOTHING, KFC_REPLY, PASS_AND_WAIT, ToolCallResult
from ..protocol.tool_call_adapter import DecisionDraft, DecisionDraftCall

if TYPE_CHECKING:
    from ..config import KFCConfig

logger = get_logger("kfc_execution")

ExecuteReplyFn = Callable[[str, "KFCConfig", Any | None, str], Awaitable[bool]]
"""发送单段回复的回调：``(文本, 配置, 触发消息, 引用消息 ID) -> 是否成功``。"""

RunToolCallFn = Callable[
    [list[Any], Any, Any, Any | None], Awaitable[list[tuple[bool, bool]]]
]
"""框架批量工具执行回调，对应 ``BaseChatter.run_tool_call``。"""

_METADATA_FIELDS: tuple[str, ...] = (
    "thought",
    "expected_reaction",
    "max_wait_seconds",
    "mood",
)
"""控制动作携带的内心活动元数据字段。"""

_RESULT_REPLY_SENT = "已发送"
_RESULT_REPLY_EMPTY = "发送失败：content 参数为空或解析后无有效内容，请重新思考并提供有效回复。"
_RESULT_REPLY_FAILED = "发送失败：内部发送环节异常，请重试或更换回复策略。"
_RESULT_REPLY_DUPLICATED = (
    "忽略重复调用：每轮响应只允许一个 kfc_reply。请将多段内容合并到一个 content 列表中。"
)
_RESULT_SILENCE = "已选择不回复"
_RESULT_WAIT_REGISTERED = "已登记等待"


def extract_metadata(result: ToolCallResult, args: dict[str, Any]) -> None:
    """把控制动作参数中的内心活动元数据提取到执行结果。

    Args:
        result: 待填充的执行结果。
        args: 控制动作的调用参数。
    """
    if "thought" in args:
        result.thought = str(args["thought"])
    if "expected_reaction" in args:
        result.expected_reaction = str(args["expected_reaction"])
    if "mood" in args:
        result.mood = str(args["mood"])
    if "max_wait_seconds" in args:
        try:
            result.max_wait_seconds = float(args["max_wait_seconds"])
        except (TypeError, ValueError):
            result.max_wait_seconds = 0.0


def parse_content_segments(raw_content: Any) -> list[str]:
    """把 ``kfc_reply`` 的 content 参数解析为文本分段列表。

    兼容模型的三种输出形态：真实列表、JSON 数组字符串、普通文本。

    Args:
        raw_content: content 参数原始值。

    Returns:
        list[str]: 去除空白后的非空分段列表。
    """
    if isinstance(raw_content, list):
        return [str(item).strip() for item in raw_content if str(item).strip()]
    if not isinstance(raw_content, str):
        return []

    stripped = raw_content.strip()
    if not stripped:
        return []
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return [stripped]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [stripped]


def calculate_typing_delay(content: str, config: KFCConfig) -> float:
    """按文本长度计算模拟打字延迟。

    Args:
        content: 待发送的文本分段。
        config: KFC 配置，提供打字速度与延迟上下限。

    Returns:
        float: 延迟秒数；打字速度非正时返回 0。
    """
    reply_config = config.reply
    if reply_config.typing_chars_per_sec <= 0:
        return 0.0
    base_delay = len(content) / reply_config.typing_chars_per_sec
    return max(
        reply_config.typing_delay_min,
        min(base_delay, reply_config.typing_delay_max),
    )


def _append_tool_result(
    response: Any,
    draft_call: DecisionDraftCall,
    value: str,
) -> None:
    """为一次调用写入 TOOL_RESULT 回执。"""
    response.add_payload(
        LLMPayload(
            ROLE.TOOL_RESULT,
            ToolResult(
                value=value,
                call_id=draft_call.call_id,
                name=draft_call.raw_name,
            ),
        )
    )


def _record_action(
    result: ToolCallResult,
    normalized_name: str,
    args: dict[str, Any],
    content_segments: list[str] | None = None,
) -> None:
    """把一次已执行动作记入结果快照。"""
    action: dict[str, Any] = {"type": normalized_name, **args}
    if content_segments is not None:
        action["content"] = content_segments
    result.actions.append(action)


async def _send_reply_segments(
    segments: list[str],
    reply_to: str,
    config: KFCConfig,
    trigger_msg: Any | None,
    execute_reply_fn: ExecuteReplyFn,
) -> bool:
    """按打字节奏依次发送回复分段。

    仅第一段携带 ``reply_to`` 引用——同一轮回复的后续分段再引用同一条
    消息会让对话界面出现重复引用块。

    Args:
        segments: 待发送的文本分段。
        reply_to: 要引用的消息 ID，可为空。
        config: KFC 配置。
        trigger_msg: 触发本轮的消息。
        execute_reply_fn: 实际发送回调。

    Returns:
        bool: 是否全部发送成功；任一段失败即中断并返回 False。
    """
    for index, segment in enumerate(segments):
        if index > 0:
            delay = calculate_typing_delay(segment, config)
            if delay > 0:
                await asyncio.sleep(delay)
        segment_reply_to = reply_to if index == 0 else ""
        if not await execute_reply_fn(segment, config, trigger_msg, segment_reply_to):
            return False
    return True


async def execute_decision_draft(
    draft: DecisionDraft,
    response: Any,
    usable_map: ToolRegistry,
    trigger_msg: Any | None,
    config: KFCConfig,
    *,
    execute_reply_fn: ExecuteReplyFn,
    run_tool_call_fn: RunToolCallFn,
    pre_execute_hook: Callable[[ToolCallResult], None] | None = None,
) -> ToolCallResult:
    """执行一份决策草稿并产出执行结果。

    第三方调用会先暂存，在遇到 ``kfc_reply`` 或草稿遍历结束时批量交给
    框架执行——这样"先查资料再回复"的组合能保证回复晚于工具结果产生。

    Args:
        draft: 已归一化的调用草稿。
        response: 模型响应对象，用于追加 TOOL_RESULT。
        usable_map: 框架工具注册表。
        trigger_msg: 触发本轮的消息。
        config: KFC 配置。
        execute_reply_fn: 发送单段回复的回调。
        run_tool_call_fn: 框架批量工具执行回调。
        pre_execute_hook: 全部动作执行完毕后的汇总日志钩子。

    Returns:
        ToolCallResult: 本轮执行结果汇总。
    """
    result = ToolCallResult()
    has_processed_reply = False
    pending_framework_calls: list[Any] = []

    async def flush_pending_framework_calls() -> None:
        """把暂存的第三方调用批量交给框架执行。"""
        if not pending_framework_calls:
            return
        current_pending = list(pending_framework_calls)
        pending_framework_calls.clear()
        logger.debug(f"交由框架执行 {len(current_pending)} 个工具/action/agent")
        call_results = await run_tool_call_fn(
            current_pending, response, usable_map, trigger_msg
        )
        for call, (_appended, success) in zip(
            current_pending, call_results, strict=False
        ):
            if not success:
                result.has_failed_tool = True
                logger.warning(f"工具 {call.name} 执行失败或被跳过")

    # 元数据只取首个控制动作——同一轮内多个控制动作的元数据语义等价。
    for draft_call in draft.calls:
        if draft_call.is_kfc_control:
            extract_metadata(result, draft_call.args)
            break

    for draft_call in draft.calls:
        args = dict(draft_call.args)
        reason = args.pop("reason", "未提供原因")
        normalized_name = draft_call.normalized_name
        logger.info(f"LLM 调用 {draft_call.raw_name}，原因: {reason}")

        if normalized_name == KFC_REPLY:
            if has_processed_reply:
                logger.warning(f"忽略重复的 kfc_reply 调用: {draft_call.call_id}")
                _append_tool_result(response, draft_call, _RESULT_REPLY_DUPLICATED)
                continue

            await flush_pending_framework_calls()
            has_processed_reply = True
            result.has_reply = True

            segments = parse_content_segments(args.get("content", ""))
            if not segments:
                result.has_failed_tool = True
                logger.warning(f"{KFC_REPLY} 的 content 解析后为空，视为工具失败")
                _append_tool_result(response, draft_call, _RESULT_REPLY_EMPTY)
                continue

            reply_to = str(args.pop("reply_to", "") or "")
            send_ok = await _send_reply_segments(
                segments, reply_to, config, trigger_msg, execute_reply_fn
            )
            if not send_ok:
                result.has_failed_tool = True

            _record_action(result, normalized_name, args, content_segments=segments)
            _append_tool_result(
                response,
                draft_call,
                _RESULT_REPLY_SENT if send_ok else _RESULT_REPLY_FAILED,
            )
            continue

        if normalized_name == DO_NOTHING:
            result.has_do_nothing = True
            _record_action(result, normalized_name, args)
            _append_tool_result(response, draft_call, _RESULT_SILENCE)
            continue

        if normalized_name == PASS_AND_WAIT:
            result.has_pass_and_wait = True
            _record_action(result, normalized_name, args)
            _append_tool_result(response, draft_call, _RESULT_WAIT_REGISTERED)
            continue

        result.has_third_party = True
        if draft_call.is_info_tool:
            result.has_info_tool = True
        _record_action(result, normalized_name, args)
        pending_framework_calls.append(draft_call.raw_call)

    await flush_pending_framework_calls()

    if pre_execute_hook is not None:
        pre_execute_hook(result)

    return result
