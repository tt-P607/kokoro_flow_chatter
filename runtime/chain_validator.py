"""持久化对话链时效性校验。

当用户在禁用 KFC 期间使用其他 chatter（如 default_chatter）进行了大量对话后
重新启用 KFC，session.chain_payloads 中缓存的旧对话对会误导模型——旧对话
权重很高，导致模型忽略新消息、出现"对话对不上"的问题。

本模块在 KFC execute() 入口处校验 chain_payloads 是否仍然与数据库消息一致：
取 chain 最后一条 user 消息的文本，在数据库最近 N 条消息中查找匹配。
匹配说明 chain 仍然有效；不匹配说明中间已有大量新对话将其"挤出"，
chain 已过时，需要清空 chain + history_summary + 压缩计数器。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.stream_api import get_stream_messages

if TYPE_CHECKING:
    from src.app.plugin_system.types import ChatStream

    from ..session import KFCSession

logger = get_logger("kfc_chain_validator")

# 从数据库取最近 N 条消息用于比对
_VERIFY_FETCH_LIMIT: int = 5

# chain 少于此条数时跳过校验（刚建立的 chain 没必要校验）
_MIN_CHAIN_LEN_TO_VERIFY: int = 2


async def validate_chain_freshness(
    session: "KFCSession",
    chat_stream: "ChatStream",
) -> None:
    """校验 chain_payloads 是否仍然与数据库消息一致，过时则清空。

    校验逻辑：
    1. chain 为空或条目太少 → 跳过
    2. 取 chain 最后一条 user 条目的 text
    3. 从数据库取该 stream 最近 5 条消息
    4. 在这 5 条消息中查找是否有 processed_plain_text 匹配 chain 最后一条 user 的 text
    5. 匹配上 → chain 有效；匹配不上 → 清空 chain + history_summary + 压缩计数器

    Args:
        session: KFC 会话状态
        chat_stream: 当前聊天流
    """
    chain = session.chain_payloads
    if len(chain) < _MIN_CHAIN_LEN_TO_VERIFY:
        return

    # 找到 chain 最后一条 user 条目
    last_user_text = ""
    for entry in reversed(chain):
        if entry.get("role") == "user":
            last_user_text = str(entry.get("text", "")).strip()
            break

    if not last_user_text:
        return

    stream_id = session.stream_id
    try:
        recent_messages = await get_stream_messages(
            stream_id=stream_id,
            limit=_VERIFY_FETCH_LIMIT,
        )
    except Exception as exc:
        logger.warning(f"校验 chain 时读取数据库失败 (stream={stream_id[:8]}): {exc}")
        return

    if not recent_messages:
        return

    # 在最近的消息中查找内容匹配
    for msg in recent_messages:
        msg_text = str(getattr(msg, "processed_plain_text", "") or "").strip()
        if msg_text and _text_matches(last_user_text, msg_text):
            return  # 匹配成功，chain 有效

    # 所有消息都匹配不上 → chain 已过时
    logger.info(
        f"chain_payloads 已过时（最后一条 user 消息在数据库最近 "
        f"{_VERIFY_FETCH_LIMIT} 条中找不到匹配），清空 chain + history_summary "
        f"(stream={stream_id[:8]})"
    )
    session.clear_chain()
    session.history_summary = ""
    session.compress_round_count = 0
    session.last_compress_at = 0.0


def _text_matches(chain_text: str, db_text: str) -> bool:
    """判断 chain 中的文本和数据库中的文本是否匹配。

    采用包含关系判断：chain_text 是 db_text 的子串，或 db_text 是 chain_text
    的子串。这是因为数据库可能对消息做了额外处理（如前后空格、格式化），
    精确匹配可能漏判。
    """
    if not chain_text or not db_text:
        return False
    # 先尝试精确匹配
    if chain_text == db_text:
        return True
    # 再尝试包含匹配（取较短的作为子串）
    shorter, longer = (chain_text, db_text) if len(chain_text) <= len(db_text) else (db_text, chain_text)
    return shorter in longer
