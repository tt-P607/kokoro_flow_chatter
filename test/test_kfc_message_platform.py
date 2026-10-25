"""验证 KFC 新消息的平台标签及发送者与消息 ID 的展示边界。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from plugins.kokoro_flow_chatter.chatter import KokoroFlowChatter
from plugins.kokoro_flow_chatter.runtime.unread_policy import format_unread_messages
from src.app.plugin_system.types import Message


def _message(platform: str, *, sender_id: str = "00042") -> Message:
    """创建不访问数据库或平台服务的合成核心消息。"""
    return Message(
        platform=platform,
        sender_id=sender_id,
        sender_name="Synthetic sender",
        sender_cardname="Synthetic card",
        sender_role="member",
        message_id="message:synthetic-platform-01",
        time=1_700_000_000.0,
        content="unprocessed body",
        processed_plain_text="processed body",
    )


@pytest.mark.parametrize(
    ("platform", "label"),
    [
        ("douyin", "douyin"),
        ("qq", "qq"),
        ("telegram", "telegram"),
        ("custom_platform", "custom_platform"),
        ("MixedCase", "MixedCase"),
        ("", "unknown"),
    ],
)
def test_message_label_uses_its_actual_platform(platform: str, label: str) -> None:
    """真实平台和原始账号展示正确，不再强制标记 QQ。"""
    line = KokoroFlowChatter.format_message_line(_message(platform))

    assert f"[{label}:00042]" in line
    assert "[QQ:" not in line
    assert "Synthetic sender$Synthetic card" in line
    assert "[消息id:message:synthetic-platform-01]" in line
    assert line.endswith("： processed body")


def test_message_without_sender_id_does_not_invent_an_account_tag() -> None:
    """缺失账号时保留消息 ID 和正文，不把消息 ID 充当账号。"""
    line = KokoroFlowChatter.format_message_line(_message("douyin", sender_id=""))

    assert "[douyin:" not in line
    assert "[QQ:" not in line
    assert "[消息id:message:synthetic-platform-01]" in line


def test_unread_payload_keeps_each_message_platform_and_string_id() -> None:
    """运行时合并未读消息时逐条保留平台，不共用固定平台标签。"""
    chatter = SimpleNamespace(format_message_line=KokoroFlowChatter.format_message_line)
    rendered = format_unread_messages(
        cast(Any, chatter),
        [_message("douyin", sender_id="uid:00042"), _message("qq", sender_id="00042")],
    )

    first, second = rendered.splitlines()
    assert "[douyin:uid:00042]" in first
    assert "[qq:00042]" in second
    assert "[QQ:" not in rendered
