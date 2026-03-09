"""KFC JSON 回复解析器。

模型在文本消息中输出 JSON 对象来表达回复决策，
本模块负责从原始响应文本中提取并规范化该 JSON。

支持的格式：
  1. 纯 JSON 对象（模型直接输出 ``{...}``）
  2. Markdown 代码块包裹（````json ... ``` ``）
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.app.plugin_system.api.log_api import get_logger

logger = get_logger("kfc_reply_json")

# 识别关键字：解析出的 dict 必须含其中至少一个才认定为 KFC 回复 JSON
_REPLY_KEYS: frozenset[str] = frozenset(
    {"content", "thought", "expected_reaction", "max_wait_seconds", "mood"}
)


def _extract_balanced_json(text: str) -> list[str]:
    """用括号深度扫描从文本中提取所有完整 JSON 对象字符串。

    正确处理嵌套括号和字符串内的 `{}`，不受 regex 非贪婪截断影响。

    Args:
        text: 待扫描的原始文本

    Returns:
        list[str]: 所有找到的顶层 JSON 对象字符串
    """
    results: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_string = False
        escape_next = False
        start = i
        j = i
        while j < n:
            c = text[j]
            if escape_next:
                escape_next = False
                j += 1
                continue
            if c == "\\" and in_string:
                escape_next = True
                j += 1
                continue
            if c == '"':
                in_string = not in_string
                j += 1
                continue
            if in_string:
                j += 1
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    results.append(text[start : j + 1])
                    i = j + 1
                    break
            j += 1
        else:
            break  # 未找到匹配的 }，停止扫描
    return results


def extract_json_reply(text: str | None) -> dict[str, Any] | None:
    """尝试从文本中提取 KFC 回复 JSON 对象。

    识别规则：解析出的 dict 必须包含至少一个回复相关键
    (content / thought / expected_reaction / max_wait_seconds / mood)。

    提取策略（按优先级）：
    1. 代码块内 JSON（```json...```），用括号平衡扫描确保嵌套完整
    2. 裸 JSON 对象，同样用括号平衡扫描

    Args:
        text: LLM 响应的原始文本

    Returns:
        dict | None: 解析成功返回字段字典，失败返回 None
    """
    if not text or not text.strip():
        return None

    candidates: list[str] = []

    # 优先尝试代码块内容（用括号平衡扫描，避免非贪婪截断）
    for block_m in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text):
        block_content = block_m.group(1)
        candidates.extend(_extract_balanced_json(block_content))

    # 再从整个文本中用括号平衡扫描提取裸 JSON
    candidates.extend(_extract_balanced_json(text))

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            data = json.loads(candidate)
            if isinstance(data, dict) and _REPLY_KEYS & data.keys():
                return data
        except json.JSONDecodeError:
            continue

    return None


def normalize_reply_data(data: dict[str, Any]) -> dict[str, Any]:
    """规范化 JSON 回复字段。

    - content: 统一为 list[str] 或 None（None 表示不回复）
    - max_wait_seconds: 转为 float
    - is_do_nothing: content 为 None 时为 True

    Args:
        data: 从文本提取的原始 JSON dict

    Returns:
        规范化后的字段字典
    """
    raw_content = data.get("content")

    if raw_content is None:
        # 显式 null → do_nothing
        content: list[str] | None = None
    elif isinstance(raw_content, str):
        stripped = raw_content.strip()
        if stripped.startswith("["):
            # 模型把数组序列化为字符串的降级兼容
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    content = [s.strip() for s in parsed if isinstance(s, str) and s.strip()] or None
                else:
                    content = [stripped] if stripped else None
            except json.JSONDecodeError:
                content = [stripped] if stripped else None
        else:
            content = [stripped] if stripped else None
    elif isinstance(raw_content, list):
        content = [s.strip() for s in raw_content if isinstance(s, str) and s.strip()] or None
    else:
        content = None

    return {
        "content": content,
        "thought": str(data.get("thought", "")).strip(),
        "expected_reaction": str(data.get("expected_reaction", "")).strip(),
        "max_wait_seconds": float(data.get("max_wait_seconds", 0) or 0),
        "mood": str(data.get("mood", "")).strip(),
        "reply_to": str(data.get("reply_to", "") or "").strip(),
        "is_do_nothing": content is None,
    }
