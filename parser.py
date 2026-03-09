"""KFC 工具调用解析器。

将 LLM 响应中的 call_list 解析为结构化的 ToolCallResult，
提取元数据（thought、expected_reaction 等）并驱动动作执行。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any, TYPE_CHECKING

from src.app.plugin_system.api.log_api import get_logger
from src.kernel.llm import LLMPayload, ROLE, ToolResult

from .models import KFC_REPLY, DO_NOTHING, ToolCallResult
from .reply_json import extract_json_reply, normalize_reply_data

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
    if not name:
        return ""

    # 兼容格式：plugin:action:kfc_reply / action:kfc_reply
    if ":" in name:
        return name.rsplit(":", 1)[-1]

    # 兼容格式：action-kfc_reply / tool-query_person / agent-xxx
    for prefix in ("action-", "tool-", "agent-"):
        if name.startswith(prefix):
            return name[len(prefix) :]

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
    execute_reply_fn: Callable[[str, KFCConfig, Any | None, str], Awaitable[bool]],
    run_tool_call_fn: Callable[[Any, Any, ToolRegistry, Any | None], Awaitable[None]],
    pre_execute_hook: Callable[[ToolCallResult], None] | None = None,
) -> ToolCallResult:
    """遍历 LLM 返回的 call_list，提取元数据并执行动作。

    在遇到第一个 kfc_reply 之前，先执行一遍收集循环以提取所有元数据。
    然后触发 pre_execute_hook 输出调试日志，最后按序执行所有动作。

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
        pre_execute_hook: 在执行任何动作前调用的钩子，接收预填好的 ToolCallResult

    Returns:
        ToolCallResult: 结构化的解析结果
    """
    result = ToolCallResult()
    is_first_reply = True
    hook_called = False

    # ── 阶段一：从消息文本提取 JSON 回复（JSON 模式）─────────────────────
    json_data = extract_json_reply(getattr(response, "message", None))
    if json_data:
        norm = normalize_reply_data(json_data)

        result.thought = norm["thought"]
        result.expected_reaction = norm["expected_reaction"]
        result.max_wait_seconds = norm["max_wait_seconds"]
        result.mood = norm["mood"]

        # 构建 action 记录（供 log_kfc_result 使用）
        action_type = DO_NOTHING if norm["is_do_nothing"] else KFC_REPLY
        action_dict: dict[str, Any] = {"type": action_type}
        if norm["thought"]:
            action_dict["thought"] = norm["thought"]
        if norm["content"] is not None:
            action_dict["content"] = norm["content"]
        if norm["reply_to"]:
            action_dict["reply_to"] = norm["reply_to"]
        result.actions.append(action_dict)

        if norm["is_do_nothing"]:
            result.has_do_nothing = True
        else:
            result.has_reply = True

        # 触发 hook
        if pre_execute_hook is not None:
            pre_execute_hook(result)
        hook_called = True

        # 执行分段发送
        if not norm["is_do_nothing"] and norm["content"]:
            segments = norm["content"]
            reply_to = norm["reply_to"]
            send_ok = True
            for segment in segments:
                if not is_first_reply:
                    delay = _calculate_typing_delay(segment, config)
                    if delay > 0:
                        logger.debug(f"[KFC-JSON] 模拟打字延迟 {delay:.2f}s")
                        await asyncio.sleep(delay)
                is_first_reply = False
                # 只在第一段应用引用，后续分段不重复引界
                seg_reply_to = reply_to if (is_first_reply is False and reply_to) else reply_to

                send_ok = await execute_reply_fn(segment, config, trigger_msg, seg_reply_to)
                if not send_ok:
                    logger.warning(f"[KFC-JSON] 段落发送失败: {repr(segment[:50])}")
                    break
                # 后续段不再引用，避免多段全部引用同一条
                reply_to = ""

            logger.debug(
                f"[KFC-JSON] JSON 回复完成: segments={len(segments)}, "
                f"preview={repr(segments[0][:80]) if segments else '(空)'}"
            )

    # ── 阶段二：处理工具调用（第三方 action/tool/agent）───────────────
    for call in response.call_list or []:
        args = _extract_args(call.args)
        normalized_name = _normalize_call_name(call.name)

        # kfc_reply / do_nothing 已由阶段一的 JSON 解析处理，此处跳过
        # （正常情况下模型不再生成这两个 tool call；保留分支作为边缘兜底）
        if normalized_name == KFC_REPLY:
            if not json_data:
                # JSON 解析未命中时，降级走旧 tool call 路径
                result.has_reply = True
                extract_metadata(result, args)
                if not hook_called and pre_execute_hook is not None:
                    pre_execute_hook(result)
                    hook_called = True

                content_raw = args.get("content", "")
                if isinstance(content_raw, list):
                    segments = [s.strip() for s in content_raw if isinstance(s, str) and s.strip()]
                elif isinstance(content_raw, str):
                    stripped = content_raw.strip()
                    if stripped.startswith("["):
                        try:
                            parsed = json.loads(stripped)
                            segments = [s.strip() for s in parsed if isinstance(s, str) and s.strip()] if isinstance(parsed, list) else [stripped]
                        except json.JSONDecodeError:
                            segments = [stripped] if stripped else []
                    else:
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
                    # 后续段不引用
                    args.pop("reply_to", None)
                    seg_reply_to = ""
                    if not send_ok:
                        break

                action_dict: dict[str, Any] = {"type": normalized_name}
                action_dict.update(args)
                result.actions.append(action_dict)

                response.add_payload(
                    LLMPayload(
                        ROLE.TOOL_RESULT,
                        ToolResult(  # type: ignore[arg-type]
                            value="已发送" if send_ok else "发送失败",
                            call_id=call.id,
                            name=call.name,
                        ),
                    )
                )
            continue

        if normalized_name == DO_NOTHING:
            if not json_data:
                result.has_do_nothing = True
                extract_metadata(result, args)
                if not hook_called and pre_execute_hook is not None:
                    pre_execute_hook(result)
                    hook_called = True

                action_dict = {"type": normalized_name}
                action_dict.update(args)
                result.actions.append(action_dict)

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
            continue

        # 第三方工具
        result.has_third_party = True
        # agent-* / tool-* 有实际返回值，需要续轮让 LLM 看到结果后才能正式回复
        if call.name.startswith(("agent-", "tool-")):
            result.has_info_tool = True
        action_dict = {"type": normalized_name}
        action_dict.update(args)
        result.actions.append(action_dict)

        await run_tool_call_fn(call, response, usable_map, trigger_msg)

    # 如果没有 kfc_reply 或 do_nothing，也要触发 hook
    if not hook_called and pre_execute_hook is not None:
        pre_execute_hook(result)

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
