"""KFC 持久化对话链条目模型。

``session.chain_payloads`` 历史上以 ``list[dict]`` 存储，导致：

1. 字段名靠魔法字符串（如 ``"role"``、``"tool_calls"``）；
2. ``text`` 为空 + ``tool_calls`` 非空 的脏数据可绕过任何静态检查；
3. 序列化与反序列化分散在 ``session.py`` / ``turn_controller.py`` /
   ``history_source.py`` 三处，任何字段微调都可能复现 A1 类 bug。

本模块提供单一可信 schema：

- 在内存里强约束（dataclass + 工厂方法）
- 在磁盘上仍以 ``dict`` 存储（保持 JSON 友好与向后兼容）
- ``from_dict`` / ``to_dict`` 是序列化的唯一入口
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


_BRIDGE_PLACEHOLDER = "好的。"


@dataclass(slots=True)
class ChainEntry:
    """一条持久化对话链条目。

    Attributes:
        role: 仅允许 ``"user"`` 或 ``"assistant"``。
        text: 显式文本；``user`` 不得为空，``assistant`` 在含 ``tool_calls``
            时若为空会自动用 ``"好的。"`` 占位，避免存档里出现无法
            还原为可读历史的空 assistant。
        tool_calls: 仅 assistant 使用，每条形如
            ``{"id": str | None, "name": str, "args": dict | str}``。
            该字段只用于审计/调试，不在历史读取时重新注入模型上下文。
        ts: 仅 user 使用；assistant 可省略。
    """

    role: str
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    ts: float | None = None

    # ---- factories -------------------------------------------------

    @classmethod
    def user(cls, text: str, ts: float | None = None) -> "ChainEntry":
        """构造 USER 条目。"""
        return cls(role="user", text=text, ts=ts)

    @classmethod
    def assistant(
        cls,
        text: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> "ChainEntry":
        """构造 ASSISTANT 条目。

        当 ``tool_calls`` 非空但 ``text`` 为空时，自动填占位 ``"好的。"``，
        确保存档仍能还原出可读的 assistant 历史消息。
        """
        calls = list(tool_calls or [])
        if calls and not text:
            text = _BRIDGE_PLACEHOLDER
        return cls(role="assistant", text=text, tool_calls=calls)

    # ---- predicates ------------------------------------------------

    @property
    def is_user(self) -> bool:
        return self.role == "user"

    @property
    def is_assistant(self) -> bool:
        return self.role == "assistant"

    @property
    def has_tool_calls(self) -> bool:
        return self.is_assistant and bool(self.tool_calls)

    # ---- (de)serialization ----------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 友好的 dict（仅写入非默认字段）。"""
        data: dict[str, Any] = {"role": self.role, "text": self.text}
        if self.tool_calls:
            data["tool_calls"] = list(self.tool_calls)
        if self.ts is not None:
            data["ts"] = self.ts
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChainEntry | None":
        """从存储 dict 还原一条 ChainEntry；脏数据返回 ``None``。

        过滤规则：
        - role 必须是 user / assistant；
        - user 且 text 为空 → 丢弃；
        - assistant 且 text 与 tool_calls 同时为空 → 丢弃；
        - tool_calls 内仅保留含 ``name`` 的 dict 条目。
        """
        role = str(data.get("role", "") or "")
        if role not in ("user", "assistant"):
            return None
        text = str(data.get("text", "") or "")
        raw_calls = data.get("tool_calls") or []
        tool_calls: list[dict[str, Any]] = [
            dict(tc)
            for tc in raw_calls
            if isinstance(tc, Mapping) and tc.get("name")
        ] if role == "assistant" else []

        if role == "user" and not text:
            return None
        if role == "assistant" and not text and not tool_calls:
            return None
        # 修复存档脏数据：assistant 含 tool_calls 但 text 空 → 兜底占位
        if role == "assistant" and tool_calls and not text:
            text = _BRIDGE_PLACEHOLDER

        ts_raw = data.get("ts")
        ts = float(ts_raw) if isinstance(ts_raw, (int, float)) and ts_raw > 0 else None
        return cls(role=role, text=text, tool_calls=tool_calls, ts=ts)
