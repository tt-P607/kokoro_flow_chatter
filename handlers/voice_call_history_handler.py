"""voice_call.ended 事件处理器：把通话整段打包成一对摘要补到 chain_payloads。

设计背景（详见 ``docs/anima_chatter_vtb_dev_log.md`` 第十一节）：

- 用户在 KFC 私聊（通常是 QQ）里和 bot 聊天 → 模型调 ``start_voice_call`` →
  anima_chatter 接管该 stream，进入语音通话。
- 通话期间所有 user / assistant 对话由 anima_chatter 处理；它会写入
  ``chat_stream.context.history_messages``，但**不会**写入 KFC 的
  ``session.chain_payloads``。
- 通话结束时 anima_chatter 广播 ``voice_call.ended`` 事件，payload 含通话
  期间发生的所有消息。

关键设计权衡（**不**逐条入 chain）：

- KFC 的 ``chain_payloads`` 默认上限 20 条（``max_context_payloads``）。
  一通 5 分钟的通话很可能产生 10+ 条消息，逐条补入会**吞掉一半 chain 额度**。
- 改为把整段通话**打包成一对**（1 user + 1 assistant）摘要：
  - user 条目：把通话期间用户发的所有话拼成一段，加边界标记。
  - assistant 条目：把 bot 的所有回复 + 状态描述拼成一段。
- 这样无论通话多长，对 chain 占用都恒定为 1 对。

如果未来想保留更细粒度，可以把 ``messages_in_call`` 序列化进
``ChainEntry`` 的扩展字段（暂时不做）。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src.app.plugin_system.api.event_api import EventDecision
from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.base import BaseEventHandler

from ..domain.chain_entry import ChainEntry

if TYPE_CHECKING:
    from src.app.plugin_system.api.event_api import EventType


logger = get_logger("kfc_voice_call_history_handler")


_VOICE_CALL_ENDED_EVENT = "voice_call.ended"

# KFC chatter 的组件签名——与 :class:`KokoroFlowChatter` 的 :meth:`get_signature` 输出一致。
_KFC_SIGNATURE = "kokoro_flow_chatter:chatter:kokoro_flow_chatter"


class VoiceCallHistoryHandler(BaseEventHandler):
    """订阅 ``voice_call.ended``，把通话历史打包成一对摘要补回 KFC session。"""

    handler_name: str = "kfc_voice_call_history_handler"
    handler_description: str = (
        "通话结束后把 anima_chatter 在 KFC 流上记录的整段对话打包成一对 "
        "user/assistant 摘要补回 session.chain_payloads，保证挂断后上下文连贯，"
        "且不会用一通通话挤占多个 chain 槽位。"
    )
    weight: int = 0
    intercept_message: bool = False
    init_subscribe: list[EventType | str] = [_VOICE_CALL_ENDED_EVENT]

    async def execute(
        self,
        event_name: str,
        params: dict[str, Any],
    ) -> tuple[EventDecision, dict[str, Any]]:
        """处理 voice_call.ended 事件。"""

        # 只处理通话发起前是 KFC 接管的 stream——其他 chatter（如
        # default_chatter）的 stream 不应被本插件改写。
        previous_signature = str(params.get("previous_chatter_signature") or "")
        if previous_signature != _KFC_SIGNATURE:
            return EventDecision.PASS, params

        stream_id = str(params.get("caller_stream_id") or "")
        if not stream_id:
            return EventDecision.PASS, params

        messages_in_call = params.get("messages_in_call") or []
        if not isinstance(messages_in_call, list) or not messages_in_call:
            logger.debug(f"voice_call.ended 无消息，跳过 stream={stream_id[:8]}")
            return EventDecision.PASS, params

        try:
            await self._patch_chain(stream_id, messages_in_call, params)
        except Exception as exc:
            logger.error(
                f"补 chain_payloads 异常 stream={stream_id[:8]}: {exc}",
                exc_info=True,
            )
            return EventDecision.PASS, params

        return EventDecision.SUCCESS, params

    @staticmethod
    def _summarize_messages(
        messages_in_call: list[Any],
    ) -> tuple[str, str, float]:
        """把整段通话压缩成"严格按时间顺序的编号交替对话稿"。

        与简单的 ``- 列表`` 不同，这里把整段通话按时间线展开，每一条消息
        都带 **轮次编号 + 角色 + 内容**，方便模型清楚地分辨每一轮谁说了什么、
        相对位置怎样。

        组装策略：
        - 把 user / assistant 消息按时间顺序穿插编号；
        - system 标注（开始 / 结束元事件）单独写在对话稿的顶部 / 底部，作为
          通话边界，不参与编号；
        - user 摘要 = 顶部边界 + 整段对话稿（这样模型读 USER 这条 chain entry
          时就能完整看到通话内容）；
        - assistant 摘要 = 底部边界 + 一句简短确认（"通话结束"），保持 chain
          交替合法（user → assistant → user → ...）但不重复发言全文，避免
          token 浪费。

        Args:
            messages_in_call: anima_chatter 传过来的消息列表，每条形如
                ``{"role": "user"|"assistant"|"system", "text": str, "ts": float}``。

        Returns:
            ``(user_summary, assistant_summary, first_user_ts)``：

            - ``user_summary``：通话边界 + 编号交替对话稿。
            - ``assistant_summary``：底部边界 + 简短确认。
            - ``first_user_ts``：第一条 user 消息的时间戳，用于 chain entry。
        """

        # 1) 拆出 system 边界 + 时间序列对话
        system_open: str = ""
        system_close: str = ""
        seen_system_count = 0
        timeline: list[tuple[str, str, float]] = []  # (role, text, ts)
        first_user_ts: float = 0.0
        fallback_ts: float = 0.0

        for msg in messages_in_call:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "")
            text = str(msg.get("text") or "").strip()
            if not text:
                continue
            ts_raw = msg.get("ts")
            ts = float(ts_raw) if isinstance(ts_raw, (int, float)) and ts_raw > 0 else 0.0

            if role == "system":
                # 第一条系统标注当通话开始边界，最后一条当通话结束边界
                if seen_system_count == 0:
                    system_open = text
                else:
                    system_close = text
                seen_system_count += 1
                if fallback_ts == 0.0 and ts > 0:
                    fallback_ts = ts
            elif role in ("user", "assistant"):
                timeline.append((role, text, ts))
                if role == "user" and first_user_ts == 0.0 and ts > 0:
                    first_user_ts = ts

        # 2) 编号交替对话稿。轮次按 user 第一次发声的"轮"算——bot 第一句话
        # （比如接通寒暄）放在第 0 轮，之后每出现一条 user 消息就 +1 轮。
        # 这样模型读"第 1 轮 用户：你好"就知道是用户主动说的第一句。
        lines: list[str] = []
        round_no = 0
        bot_pre_label = "（接通时）"  # 第一条 user 之前 bot 的台词都打这个标
        last_role = ""
        for role, text, _ in timeline:
            if role == "user":
                round_no += 1
                lines.append(f"【第 {round_no} 轮 / 用户】{text}")
            else:  # assistant
                if round_no == 0:
                    label = bot_pre_label
                else:
                    label = f"（第 {round_no} 轮回应）"
                lines.append(f"【你的回应{label}】{text}")
            last_role = role
        _ = last_role  # 仅辅助调试，未使用

        if not lines:
            timeline_block = "（通话期间没有任何对话发生。）"
        else:
            timeline_block = "\n".join(lines)

        # 3) user 摘要：顶部边界 + 完整时间线
        user_parts: list[str] = []
        if system_open:
            user_parts.append(system_open)
        user_parts.append("【通话对话稿（按时间顺序）】")
        user_parts.append(timeline_block)
        user_summary = "\n\n".join(user_parts)

        # 4) assistant 摘要：底部边界 + 简短确认（保持 chain 交替合法）
        assistant_parts: list[str] = []
        assistant_parts.append("（已收到上面整段通话稿。）")
        if system_close:
            assistant_parts.append(system_close)
        assistant_summary = "\n\n".join(assistant_parts)

        return (
            user_summary,
            assistant_summary,
            first_user_ts or fallback_ts or time.time(),
        )

    async def _patch_chain(
        self,
        stream_id: str,
        messages_in_call: list[Any],
        event_params: dict[str, Any],
    ) -> None:
        """把整段通话打包成一对 (user, assistant) 摘要写入 KFC chain_payloads。"""

        from ..config import KFCConfig
        from ..plugin import KFCPlugin

        plugin = self.plugin
        if not isinstance(plugin, KFCPlugin):
            logger.warning("VoiceCallHistoryHandler 不在 KFCPlugin 上下文，跳过")
            return

        config = plugin.config
        if not isinstance(config, KFCConfig):
            logger.warning("KFC 配置未加载，跳过 chain_payloads 补丁")
            return

        user_summary, assistant_summary, first_user_ts = self._summarize_messages(
            messages_in_call
        )

        if not user_summary.strip() and not assistant_summary.strip():
            logger.debug(f"通话摘要全为空 stream={stream_id[:8]}，跳过 chain 补丁")
            return

        entries = [
            ChainEntry.user(text=user_summary, ts=first_user_ts).to_dict(),
            ChainEntry.assistant(text=assistant_summary).to_dict(),
        ]

        # 走 per-stream 锁串行化，与 ProactiveThinker 等并发读写者隔离。
        store = plugin._session_store  # type: ignore[attr-defined]
        async with store.lock(stream_id):
            session = await store.get_or_create(stream_id)
            session.update_chain(entries, config.prompt.max_context_payloads)
            await store.save(session)

        # 摘要本身可能很短，但消息条目原始数量也记日志，方便排查"明明聊了半天
        # 怎么 chain 只多了一对"的疑惑。
        raw_count = sum(
            1
            for m in messages_in_call
            if isinstance(m, dict) and str(m.get("text") or "").strip()
        )
        duration = float(event_params.get("duration_seconds") or 0.0)
        logger.info(
            f"已把通话打包成 1 对 chain entry stream={stream_id[:8]} "
            f"(原始消息 {raw_count} 条 / 持续 {duration:.0f}s)"
        )


__all__ = ["VoiceCallHistoryHandler"]
