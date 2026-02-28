"""等待检查器。

WaitChecker 在 Bot 等待期间运行连续思考，
根据进度阈值触发内心独白并写入 Session。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.app.plugin_system.api.log_api import get_logger

if TYPE_CHECKING:
    from ..config import KFCConfig
    from ..session import KFCSession

logger = get_logger("kfc_wait_checker")


class WaitChecker:
    """等待期间的连续思考检查器。

    按照 progress_thresholds 设定的进度阈值，
    在等待期间触发连续思考，并将思考结果写入 Session.pending_thoughts。
    由 Scheduler 定期调用。
    """

    def __init__(self, config: KFCConfig) -> None:
        self._config = config

    async def check_and_think(self, session: KFCSession) -> str | None:
        """检查等待进度，必要时生成连续思考。

        Args:
            session: 当前会话状态

        Returns:
            str | None: 生成的思考内容，未触发则返回 None
        """
        ct_config = self._config.continuous_thinking
        if not ct_config.enabled:
            return None

        if not session.waiting_config.is_active():
            return None

        # 检查最小间隔
        now = time.time()
        last_thinking = session.waiting_config.last_thinking_at
        if last_thinking > 0 and (now - last_thinking) < ct_config.min_interval:
            return None

        # 检查进度阈值
        progress = session.waiting_config.get_progress()
        thinking_count = session.waiting_config.thinking_count
        thresholds = ct_config.progress_thresholds

        if thinking_count >= len(thresholds):
            return None

        target_threshold = thresholds[thinking_count]
        if progress < target_threshold:
            return None

        # 生成连续思考
        thought = await self._generate_thought(session, progress)
        if thought:
            # 更新状态
            session.waiting_config.last_thinking_at = now
            session.waiting_config.thinking_count += 1
            session.pending_thoughts.append(thought)
            session.add_waiting_update(thought)

            logger.debug(
                f"连续思考 #{thinking_count + 1}: "
                f"progress={progress:.0%}, thought={thought[:50]}"
            )

        return thought

    async def _generate_thought(
        self, session: KFCSession, progress: float
    ) -> str:
        """调用 LLM 生成连续思考内容。"""
        from ..prompts.modules import build_continuous_thinking_context
        from src.app.plugin_system.api.llm_api import (
            get_model_set_by_task,
            create_llm_request,
        )
        from src.kernel.llm import LLMPayload, ROLE, Text, LLMContextManager

        elapsed = session.waiting_config.get_elapsed_seconds()
        expected = session.waiting_config.expected_reaction
        last_bot_message = session.mental_log.get_last_bot_reply_content()

        context_text = await build_continuous_thinking_context(
            elapsed_seconds=elapsed,
            progress=progress,
            expected_reaction=expected,
            last_bot_message=last_bot_message,
        )

        try:
            # 连续思考使用 sub_actor 模型任务，比主对话模型更轻量
            model_set = get_model_set_by_task("sub_actor")
            if not model_set:
                return self._fallback_thought(progress)

            context_manager = LLMContextManager(max_payloads=5)
            request = create_llm_request(
                model_set,
                "kfc_continuous_thinking",
                context_manager=context_manager,
            )
            request.add_payload(
                LLMPayload(
                    ROLE.SYSTEM,
                    Text("你是一个正在等待对方回复的人，请简短地描述你此刻的内心感受。"),
                )
            )
            request.add_payload(LLMPayload(ROLE.USER, Text(context_text)))

            response = await request.send(stream=False)
            await response

            result = response.message
            if result and result.strip():
                return result.strip()[:200]

        except Exception as e:
            logger.warning(f"连续思考 LLM 调用失败: {e}")

        return self._fallback_thought(progress)

    @staticmethod
    def _fallback_thought(progress: float) -> str:
        """兜底思考内容。"""
        if progress < 0.3:
            return "刚发完消息，有点期待对方的回复呢"
        if progress < 0.6:
            return "对方还没回复，是不是在忙呢"
        if progress < 0.85:
            return "等了一会儿了，不知道对方有没有看到消息"
        return "等了挺久了，也许该做点别的了"
