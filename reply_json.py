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

# JSON 代码块（```json...``` 或 ```...```）
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
# 裸 JSON 对象（贪婪匹配最外层大括号，支持嵌套）
_JSON_BARE_RE = re.compile(r"\{.*\}", re.DOTALL)

# 识别关键字：解析出的 dict 必须含其中至少一个才认定为 KFC 回复 JSON
_REPLY_KEYS: frozenset[str] = frozenset(
    {"content", "thought", "expected_reaction", "max_wait_seconds", "mood"}
)


def extract_json_reply(text: str | None) -> dict[str, Any] | None:
    """尝试从文本中提取 KFC 回复 JSON 对象。

    识别规则：解析出的 dict 必须包含至少一个回复相关键
    (content / thought / expected_reaction / max_wait_seconds / mood)。

    Args:
        text: LLM 响应的原始文本

    Returns:
        dict | None: 解析成功返回字段字典，失败返回 None
    """
    if not text or not text.strip():
        return None

    candidates: list[str] = []

    # 优先尝试代码块格式（精确）
    for m in _JSON_BLOCK_RE.finditer(text):
        candidates.append(m.group(1))

    # 再尝试裸 JSON（宽松）
    for m in _JSON_BARE_RE.finditer(text):
        candidates.append(m.group(0))

    for candidate in candidates:
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
        "is_do_nothing": content is None,
    }
