"""KFC 上下文生命周期的定向回归测试。"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plugins.kokoro_flow_chatter.context.planner import (  # noqa: E402
    build_last_mile_payload,
)
from plugins.kokoro_flow_chatter.context.renderer import (  # noqa: E402
    render_initial_context,
    render_turn_contributions,
)
from plugins.kokoro_flow_chatter.context.sources.plugin_source import (  # noqa: E402
    _normalize_contribution,
)
from plugins.kokoro_flow_chatter.context.types import InitialContextPlan  # noqa: E402
from plugins.kokoro_flow_chatter.config import KFCConfig  # noqa: E402
from plugins.kokoro_flow_chatter.context_budget import KFCContextManager
from plugins.kokoro_flow_chatter.handlers.voice_call_history_handler import (  # noqa: E402
    VoiceCallHistoryHandler,
)
from plugins.kokoro_flow_chatter.plugin import KFCPlugin  # noqa: E402
from plugins.kokoro_flow_chatter.runtime.turn_controller import (  # noqa: E402
    commit_turn_decision,
)
from plugins.kokoro_flow_chatter.services.summary_service import (  # noqa: E402
    SummaryService,
)
from plugins.kokoro_flow_chatter.domain.decision import Decision  # noqa: E402
from plugins.kokoro_flow_chatter.session import KFCSession, KFCSessionStore  # noqa: E402
from plugins.kokoro_flow_chatter import session as session_module  # noqa: E402
from plugins.kokoro_flow_chatter.snapshot import (  # noqa: E402
    DYNAMIC_BACKGROUND_MARKER,
    capture_snapshot,
    deserialize_snapshot,
)
from plugins.kokoro_flow_chatter.runtime.request_view import (  # noqa: E402
    _without_transient_payloads,
    build_request_view,
)
from plugins.kokoro_flow_chatter.runtime import context_builder  # noqa: E402
from plugins.kokoro_flow_chatter.runtime.summary_sync import SummarySynchronizer  # noqa: E402
from src.app.plugin_system.types import (  # noqa: E402
    Content,
    Image,
    LLMPayload,
    ROLE,
    Text,
    ToolCall,
    ToolResult,
)


@pytest.mark.asyncio
async def test_session_files_use_platform_user_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """状态与快照以平台用户 ID 命名，不生成账号索引。"""
    directory = tmp_path / "sessions"
    monkeypatch.setattr(session_module, "_STORAGE_DIR", str(directory))
    session = KFCSession(user_id="10001", stream_id="stream-1", platform="qq")
    session.history_summary = "已有日记"
    session.scheduled_proactive_at = 123.0
    session.context_snapshot = [
        {"role": "user", "content": [{"type": "text", "text": "原话"}]}
    ]
    await KFCSessionStore().save(session)

    assert (directory / "state" / "qq" / "10001.json").is_file()
    assert (directory / "context" / "qq" / "10001.json").is_file()
    assert not list(directory.rglob("_index.json"))
    assert not list(directory.rglob("stream-1.json"))

    restarted = KFCSessionStore()
    assert await restarted.list_all_stream_ids() == [session.stream_id]
    loaded = await restarted.peek(session.stream_id)
    assert loaded is not None
    assert restarted.get_all_cached() == {}
    assert loaded.user_id == session.user_id
    assert loaded.platform == session.platform
    assert loaded.history_summary == session.history_summary
    assert loaded.context_snapshot == session.context_snapshot
    assert loaded.scheduled_proactive_at == session.scheduled_proactive_at


@pytest.mark.asyncio
async def test_same_user_id_is_isolated_by_platform(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """不同平台的同名账号不会覆盖彼此的日记与快照。"""
    directory = tmp_path / "sessions"
    monkeypatch.setattr(session_module, "_STORAGE_DIR", str(directory))
    store = KFCSessionStore()
    for platform in ("qq", "telegram"):
        await store.save(KFCSession(
            user_id="10001", stream_id=f"stream-{platform}",
            platform=platform, history_summary=platform,
        ))

    restarted = KFCSessionStore()
    assert set(await restarted.list_all_stream_ids()) == {"stream-qq", "stream-telegram"}
    for platform in ("qq", "telegram"):
        loaded = await restarted.get(f"stream-{platform}")
        assert loaded is not None
        assert loaded.history_summary == platform
        assert (directory / "state" / platform / "10001.json").is_file()
        assert (directory / "context" / platform / "10001.json").is_file()


@pytest.mark.asyncio
async def test_hash_named_cache_is_renamed_without_losing_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """已有双文件缓存改名后，原话、封存段、日记和预约均可重启恢复。"""
    directory = tmp_path / "sessions"
    monkeypatch.setattr(session_module, "_STORAGE_DIR", str(directory))
    session = KFCSession(user_id="10001", stream_id="stream-legacy", platform="qq")
    session.history_summary = "已有日记"
    session.scheduled_proactive_at = 123.0
    session.context_snapshot = [
        {"role": "user", "content": [{"type": "text", "text": "原话"}]}
    ]
    session.sealed_segments = [session.context_snapshot]
    old_context = session.context_to_dict()
    for field in ("user_id", "stream_id", "platform"):
        old_context.pop(field)
    for part, data in (("state", session.to_dict()), ("context", old_context)):
        parent = directory / part
        parent.mkdir(parents=True)
        (parent / f"{session.stream_id}.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
    (directory / "state" / "_index.json").write_text("{}", encoding="utf-8")

    loaded = await KFCSessionStore().get(session.stream_id)
    assert loaded is not None
    assert loaded.history_summary == session.history_summary
    assert loaded.context_snapshot == session.context_snapshot
    assert loaded.sealed_segments == session.sealed_segments
    assert loaded.scheduled_proactive_at == session.scheduled_proactive_at
    assert not list(directory.rglob(f"{session.stream_id}.json"))
    assert not list(directory.rglob("_index.json"))

    restarted = await KFCSessionStore().get(session.stream_id)
    assert restarted is not None
    assert restarted.history_summary == session.history_summary
    assert restarted.sealed_segments == session.sealed_segments
    assert (directory / "state" / "qq" / "10001.json").is_file()
    assert (directory / "context" / "qq" / "10001.json").is_file()


@pytest.mark.asyncio
async def test_context_only_save_resolves_user_from_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """首次通话回填或后台日记可从公开流 API 获取缓存命名所需身份。"""
    directory = tmp_path / "sessions"
    monkeypatch.setattr(session_module, "_STORAGE_DIR", str(directory))
    monkeypatch.setattr(session_module, "get_stream", AsyncMock(return_value=SimpleNamespace(
        platform="qq", bot_id="bot",
        context=SimpleNamespace(current_message=SimpleNamespace(sender_id="10001")),
    )))
    session = KFCSession(user_id="", stream_id="stream-1", history_summary="已有日记")
    await KFCSessionStore().save_context(session)

    assert (directory / "context" / "qq" / "10001.json").is_file()
    assert not list((directory / "state").rglob("*.json"))
    loaded = await KFCSessionStore().get(session.stream_id)
    assert loaded is not None
    assert loaded.user_id == "10001"
    assert loaded.history_summary == session.history_summary


@pytest.mark.asyncio
async def test_old_session_identity_is_resolved_before_context_only_rename(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """旧状态缺身份时，首次上下文写入必须同时改名两份缓存。"""
    directory = tmp_path / "sessions"
    monkeypatch.setattr(session_module, "_STORAGE_DIR", str(directory))
    monkeypatch.setattr(session_module, "get_stream", AsyncMock(return_value=SimpleNamespace(
        platform="qq", bot_id="bot",
        context=SimpleNamespace(current_message=SimpleNamespace(sender_id="10001")),
    )))
    session = KFCSession(user_id="", stream_id="stream-1", history_summary="旧日记")
    for part, data in (("state", session.to_dict()), ("context", session.context_to_dict())):
        parent = directory / part
        parent.mkdir(parents=True)
        (parent / "stream-1.json").write_text(json.dumps(data), encoding="utf-8")

    store = KFCSessionStore()
    loaded = await store.get_or_create(session.stream_id)
    loaded.history_summary = "新的日记"
    await store.save_context(loaded)

    assert not list(directory.rglob("stream-1.json"))
    assert (directory / "state" / "qq" / "10001.json").is_file()
    assert (directory / "context" / "qq" / "10001.json").is_file()
    restarted = await KFCSessionStore().get(session.stream_id)
    assert restarted is not None
    assert restarted.history_summary == "新的日记"


@pytest.mark.asyncio
async def test_cache_rename_write_failure_keeps_original_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """新状态写入失败时不删除旧状态或旧快照，重试仍可完整恢复。"""
    directory = tmp_path / "sessions"
    monkeypatch.setattr(session_module, "_STORAGE_DIR", str(directory))
    session = KFCSession(
        user_id="10001", stream_id="stream-1", platform="qq", history_summary="已有日记"
    )
    original_files: dict[Path, bytes] = {}
    for part, data in (("state", session.to_dict()), ("context", session.context_to_dict())):
        parent = directory / part
        parent.mkdir(parents=True)
        path = parent / "stream-1.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        original_files[path] = path.read_bytes()

    original_replace = session_module.os.replace

    def fail_state_replace(source: str, destination: Path) -> None:
        """允许上下文写入，但让新状态提交报告磁盘错误。"""
        if destination.parent.parent.name == "state":
            raise OSError("disk full")
        original_replace(source, destination)

    monkeypatch.setattr(session_module.os, "replace", fail_state_replace)
    with pytest.raises(OSError, match="disk full"):
        await KFCSessionStore().get(session.stream_id)
    for path, original in original_files.items():
        assert path.read_bytes() == original

    monkeypatch.setattr(session_module.os, "replace", original_replace)
    loaded = await KFCSessionStore().get(session.stream_id)
    assert loaded is not None
    assert loaded.history_summary == session.history_summary
    assert not list(directory.rglob("stream-1.json"))


@pytest.mark.asyncio
async def test_missing_user_id_does_not_write_hash_named_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """没有真实账号身份时不得回退到内部流 ID 文件名。"""
    directory = tmp_path / "sessions"
    monkeypatch.setattr(session_module, "_STORAGE_DIR", str(directory))
    monkeypatch.setattr(session_module, "get_stream", AsyncMock(return_value=None))
    with pytest.raises(ValueError, match="缺少真实用户 ID"):
        await KFCSessionStore().save_context(KFCSession(user_id="", stream_id="stream-1"))
    assert not list(directory.rglob("*.json"))


@pytest.mark.asyncio
async def test_same_account_cannot_overwrite_another_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """同平台同账号的内部流冲突必须显式报错，不覆盖已有缓存。"""
    monkeypatch.setattr(session_module, "_STORAGE_DIR", str(tmp_path / "sessions"))
    store = KFCSessionStore()
    session = KFCSession(
        user_id="10001", stream_id="stream-1", platform="qq", history_summary="已有日记"
    )
    await store.save(session)
    with pytest.raises(ValueError, match="拒绝覆盖"):
        await store.save(KFCSession(
            user_id="10001", stream_id="stream-2", platform="qq", history_summary="其他日记"
        ))
    loaded = await KFCSessionStore().get(session.stream_id)
    assert loaded is not None
    assert loaded.history_summary == session.history_summary


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", ["../10001", "10001/part", "10001\\part", "CON", "10001:part"])
async def test_cache_user_id_cannot_escape_storage_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, user_id: str
) -> None:
    """账号 ID 含路径或 Windows 保留名称时不得落盘。"""
    directory = tmp_path / "sessions"
    monkeypatch.setattr(session_module, "_STORAGE_DIR", str(directory))
    with pytest.raises(ValueError, match="安全的文件名"):
        await KFCSessionStore().save(KFCSession(
            user_id=user_id, stream_id="stream-1", platform="qq"
        ))
    assert not list(directory.rglob("*.json"))


@pytest.mark.asyncio
async def test_corrupt_context_clears_only_context_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """上下文损坏后，独立保存的会话状态仍可恢复。"""
    monkeypatch.setattr(session_module, "_STORAGE_DIR", str(tmp_path / "sessions"))
    store = KFCSessionStore()
    session = KFCSession(user_id="u1", stream_id="stream-1", platform="qq")
    session.scheduled_proactive_at = 123.0
    session.context_snapshot = [
        {"role": "user", "content": [{"type": "text", "text": "原话"}]}
    ]
    await store.save(session)

    state_path = tmp_path / "sessions" / "state" / "qq" / "u1.json"
    context_path = tmp_path / "sessions" / "context" / "qq" / "u1.json"
    assert state_path.exists()
    assert context_path.exists()
    context_path.write_text("{broken", encoding="utf-8")

    loaded = await KFCSessionStore().get_or_create("stream-1")
    assert loaded.scheduled_proactive_at == 123.0
    assert loaded.context_snapshot is None
    assert not context_path.exists()
    assert state_path.exists()


@pytest.mark.asyncio
async def test_incomplete_context_cannot_silently_drop_tool_tail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """无效活动链落盘清空，不影响日记和完整封存段的恢复。"""
    monkeypatch.setattr(session_module, "_STORAGE_DIR", str(tmp_path / "sessions"))
    session = KFCSession(
        user_id="user", stream_id="stream-1", platform="qq", history_summary="我已有日记"
    )
    session.context_snapshot = [
        {"role": "user", "content": [{"type": "text", "text": "原话"}]},
        {"role": "assistant", "content": [
            {"type": "tool_call", "id": "c1", "name": "lookup", "args": {}}
        ]},
    ]
    session.sealed_segments = [[
        {"role": "user", "content": [{"type": "text", "text": "已封存原话"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "已封存回复"}]},
    ]]
    await KFCSessionStore().save(session)

    loaded = await KFCSessionStore().get_or_create("stream-1")
    assert loaded.user_id == "user"
    assert loaded.context_snapshot is None
    assert loaded.history_summary == "我已有日记"
    assert loaded.sealed_segments == session.sealed_segments
    context_path = tmp_path / "sessions" / "context" / "qq" / "user.json"
    persisted = json.loads(context_path.read_text(encoding="utf-8"))
    assert persisted["context_snapshot"] is None
    assert persisted["history_summary"] == session.history_summary
    assert persisted["sealed_segments"] == session.sealed_segments
    assert (tmp_path / "sessions" / "state" / "qq" / "user.json").exists()
    restarted = await KFCSessionStore().get_or_create("stream-1")
    assert restarted.history_summary == "我已有日记"
    assert restarted.context_snapshot is None
    assert restarted.sealed_segments == session.sealed_segments
    restarted.append_context_entries([
        LLMPayload(ROLE.USER, Text("新的输入")),
        LLMPayload(ROLE.ASSISTANT, Text("新的回复")),
    ])
    await KFCSessionStore().save(restarted)
    resumed = await KFCSessionStore().get_or_create("stream-1")
    assert resumed.context_snapshot == restarted.context_snapshot
    assert resumed.history_summary == session.history_summary
    assert deserialize_snapshot(resumed.context_snapshot) is not None
    recent_history = SimpleNamespace(
        bot_id="bot",
        context=SimpleNamespace(history_messages=[SimpleNamespace(
            time=time.time() - 60,
            processed_plain_text="近期聊天",
            sender_id="u1",
            sender_name="用户",
            message_id="m1",
        )]),
    )
    restarted.sealed_segments = []
    assert not SummaryService.maybe_schedule_compression(
        restarted, KFCConfig(), recent_history
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "invalid_value", "cleared_value"),
    [
        ("context_snapshot", [], None),
        ("context_snapshot", "invalid", None),
        ("context_snapshot", [{"role": "user", "content": []}], None),
        ("history_summary", 123, ""),
        ("sealed_segments", "invalid", []),
        ("sealed_segments", [[{"role": "user", "content": []}]], []),
    ],
)
async def test_invalid_context_field_is_cleared_on_disk_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    field: str, invalid_value: object, cleared_value: object,
) -> None:
    """损坏字段单独清空落盘，重复磁盘读取无需再次修复。"""
    monkeypatch.setattr(session_module, "_STORAGE_DIR", str(tmp_path / "sessions"))
    session = KFCSession(
        user_id="10001", stream_id="stream-1", platform="qq", history_summary="已有日记"
    )
    session.context_snapshot = [
        {"role": "user", "content": [{"type": "text", "text": "有效原话"}]}
    ]
    session.sealed_segments = [session.context_snapshot]
    session.scheduled_proactive_at = 123.0
    await KFCSessionStore().save(session)
    path = tmp_path / "sessions" / "context" / "qq" / "10001.json"
    original = session.context_to_dict()
    corrupted = dict(original)
    corrupted[field] = invalid_value
    path.write_text(json.dumps(corrupted), encoding="utf-8")
    repair = AsyncMock(wraps=session_module._atomic_save)
    monkeypatch.setattr(session_module, "_atomic_save", repair)

    loaded = await KFCSessionStore().get_or_create(session.stream_id)
    expected = dict(original)
    expected[field] = cleared_value
    assert loaded.context_to_dict() == expected
    assert loaded.scheduled_proactive_at == session.scheduled_proactive_at
    assert json.loads(path.read_text(encoding="utf-8")) == expected
    assert repair.await_count == 1

    restarted = KFCSessionStore()
    for _attempt in range(2):
        peeked = await restarted.peek(session.stream_id)
        assert peeked is not None
        assert peeked.context_to_dict() == expected
    assert repair.await_count == 1


@pytest.mark.asyncio
async def test_context_repair_failure_keeps_original_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """修复提交失败不得删掉原文件，恢复磁盘后仍可清空损坏字段。"""
    monkeypatch.setattr(session_module, "_STORAGE_DIR", str(tmp_path / "sessions"))
    session = KFCSession(
        user_id="10001", stream_id="stream-1", platform="qq", history_summary="已有日记"
    )
    session.context_snapshot = []
    await KFCSessionStore().save(session)
    path = tmp_path / "sessions" / "context" / "qq" / "10001.json"
    original = path.read_bytes()
    original_replace = session_module.os.replace

    def fail_replace(source: str, destination: Path) -> None:
        """模拟缓存原子提交时的磁盘错误。"""
        raise OSError("disk full")

    monkeypatch.setattr(session_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        await KFCSessionStore().get(session.stream_id)
    assert path.read_bytes() == original
    assert {entry.name for entry in path.parent.iterdir()} == {path.name}

    monkeypatch.setattr(session_module.os, "replace", original_replace)
    loaded = await KFCSessionStore().get(session.stream_id)
    assert loaded is not None
    assert loaded.context_snapshot is None
    assert loaded.history_summary == session.history_summary
    assert json.loads(path.read_text(encoding="utf-8"))["context_snapshot"] is None


@pytest.mark.asyncio
async def test_corrupt_state_preserves_independent_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """状态文件损坏不应删除或阻止恢复独立的活动原话与日记。"""
    monkeypatch.setattr(session_module, "_STORAGE_DIR", str(tmp_path / "sessions"))
    store = KFCSessionStore()
    session = KFCSession(
        user_id="old", stream_id="stream-1", platform="qq", history_summary="我记得原话"
    )
    session.context_snapshot = [
        {"role": "user", "content": [{"type": "text", "text": "原话"}]}
    ]
    await store.save(session)
    state_path = tmp_path / "sessions" / "state" / "qq" / "old.json"
    context_path = tmp_path / "sessions" / "context" / "qq" / "old.json"
    state_path.write_text("{broken", encoding="utf-8")

    loaded = await KFCSessionStore().get_or_create("stream-1")
    assert loaded.user_id == "old"
    assert loaded.context_snapshot == session.context_snapshot
    assert loaded.history_summary == "我记得原话"
    assert not state_path.exists()
    assert context_path.exists()


def test_diary_hot_update_targets_dynamic_background_only() -> None:
    """后台日记只能写进动态背景，不能污染真实用户消息。"""
    real_user = LLMPayload(ROLE.USER, Text("真实原话"))
    manager = KFCContextManager()
    response = SimpleNamespace(payloads=[real_user], context_manager=manager)
    stream = SimpleNamespace(stream_name="对方")
    sync = SummarySynchronizer("")
    assert not sync.sync_if_changed(response, stream, "我记得这件事")
    assert _text(real_user) == "真实原话"

    background = LLMPayload(ROLE.USER, Text(
        f"{DYNAMIC_BACKGROUND_MARKER}\n\n[当前通道参数]\n\n---\n\n当前时间：现在"
    ))
    manager.background_payloads.append(background)
    assert sync.sync_if_changed(response, stream, "我记得这件事")
    assert "我记得这件事" in _text(background)
    assert _text(real_user) == "真实原话"


@pytest.mark.asyncio
async def test_legacy_session_file_is_deleted_without_migration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """旧单文件缓存只报告并删除，绝不导入新格式。"""
    monkeypatch.setattr(session_module, "_STORAGE_DIR", str(tmp_path / "sessions"))
    legacy = tmp_path / "sessions" / "stream-1.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"user_id": "old", "context_snapshot": []}', encoding="utf-8")

    loaded = await KFCSessionStore().get_or_create("stream-1")
    assert loaded.user_id == ""
    assert loaded.context_snapshot is None
    assert not legacy.exists()


@pytest.mark.asyncio
async def test_session_write_failure_does_not_replace_existing_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """写入失败不得把新状态伪装成成功，也不破坏上次的状态文件。"""
    monkeypatch.setattr(session_module, "_STORAGE_DIR", str(tmp_path / "sessions"))
    store = KFCSessionStore()
    session = KFCSession(user_id="old", stream_id="stream-1")
    await store.save(session)
    session.user_id = "new"

    def fail_replace(source: str, destination: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(session_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        await store.save(session)
    assert (await KFCSessionStore().get_or_create("stream-1")).user_id == "old"


def _text(payload: LLMPayload) -> str:
    content = payload.content
    if not isinstance(content, list):
        content = [content]
    return "".join(part.text for part in content if isinstance(part, Text))


def test_context_snapshot_is_independent_of_session_state() -> None:
    """状态序列化不包含对话快照，活动链仍能单独恢复。"""
    state_data = {
        "user_id": "u1",
        "stream_id": "stream-1",
    }
    loaded = KFCSession.from_dict(state_data)
    loaded.context_snapshot = [
        {"role": "user", "content": [{"type": "text", "text": "已有对话"}]}
    ]

    loaded.append_context_entries([
        LLMPayload(ROLE.USER, Text("你好")),
        LLMPayload(ROLE.ASSISTANT, Text("你好呀")),
    ])
    assert "context_snapshot" not in loaded.to_dict()
    assert "context_snapshot" in loaded.context_to_dict()
    restored = deserialize_snapshot(loaded.context_snapshot)
    assert restored is not None
    assert [_text(payload) for payload in restored] == ["已有对话", "你好", "你好呀"]
    assert importlib.util.find_spec(
        "plugins.kokoro_flow_chatter.domain.chain_entry"
    ) is None


@pytest.mark.asyncio
async def test_build_initial_request_clears_invalid_snapshot_and_saves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非法快照应被清除并持久化，正常快照不得被清除。"""
    class _Request:
        def __init__(self) -> None:
            self.payloads: list[LLMPayload] = []

        def add_payload(self, payload: LLMPayload) -> None:
            self.payloads.append(payload)

    class _Chatter:
        stream_id = "stream-test"

        def __init__(self) -> None:
            self.saved: list[KFCSession] = []

        async def save_session(self, session: KFCSession) -> None:
            self.saved.append(session)

        async def inject_usables(self, request: _Request) -> dict[str, object]:
            return {}

    plan = InitialContextPlan(system_extra_vars={}, history_summary="")
    monkeypatch.setattr(context_builder, "create_llm_request", lambda *args, **kwargs: _Request())
    monkeypatch.setattr(context_builder, "plan_initial_context", lambda **kwargs: plan)
    async def render(**kwargs: object) -> tuple[list[LLMPayload], list[LLMPayload], list[LLMPayload], bool]:
        return _render_snapshot_for_test(kwargs["serialized_context_snapshot"])

    monkeypatch.setattr(context_builder, "render_initial_context", render)

    async def run(snapshot: list[dict[str, object]] | None) -> tuple[KFCSession, _Chatter]:
        chatter = _Chatter()
        session = KFCSession(user_id="u1", stream_id="stream-test")
        session.context_snapshot = snapshot
        await context_builder.build_initial_request(
            chatter,
            SimpleNamespace(),
            KFCConfig(),
            session,
            object(),
        )
        return session, chatter

    invalid_session, invalid_chatter = await run([{"invalid": True}])
    assert invalid_session.context_snapshot is None
    assert invalid_chatter.saved == [invalid_session]

    valid_snapshot = [{"role": "user", "content": [{"type": "text", "text": "历史"}]}]
    valid_session, valid_chatter = await run(valid_snapshot)
    assert valid_session.context_snapshot == valid_snapshot
    assert valid_chatter.saved == []


def _render_snapshot_for_test(
    snapshot: list[dict[str, object]] | None,
) -> tuple[list[LLMPayload], list[LLMPayload], list[LLMPayload], bool]:
    """构造初始请求测试所需的最小渲染结果。"""
    has_history = bool(snapshot and "role" in snapshot[0])
    restored = [LLMPayload(ROLE.USER, Text("历史"))] if has_history else []
    return [], [], restored, has_history


def test_dynamic_background_and_transients_do_not_enter_snapshot() -> None:
    """通道、摘要、叙事、last-mile 和 retry 都不能被捕获为历史。"""
    background_text = (
        f"{DYNAMIC_BACKGROUND_MARKER}\n\n"
        "[当前通道参数]\n\n---\n\n【近期记忆】长期摘要\n\n---\n\n融合叙事"
    )
    payloads = [
        LLMPayload(ROLE.SYSTEM, Text("稳定系统规则")),
        LLMPayload(ROLE.USER, Text("真实用户输入")),
        LLMPayload(ROLE.ASSISTANT, Text("真实回复")),
    ]
    snapshot = capture_snapshot(payloads)
    assert snapshot is not None
    response = SimpleNamespace(
        payloads=payloads,
        context_manager=KFCContextManager(background_payloads=[
            LLMPayload(ROLE.USER, Text(background_text))
        ]),
    )
    view = build_request_view(response)
    assert "当前通道参数" in str(view.payloads)
    text = str(snapshot)
    assert "稳定系统规则" not in text
    for forbidden in ("当前通道参数", "长期摘要", "融合叙事"):
        assert forbidden not in text

    source = [LLMPayload(ROLE.USER, Text("真实用户输入"))]
    transients = [
        build_last_mile_payload(),
        LLMPayload(ROLE.USER, Text("重试提醒")),
    ]
    assert transients[0].role == ROLE.SYSTEM
    result = [
        source[0],
        *transients,
        LLMPayload(ROLE.ASSISTANT, Text("真实回复")),
    ]
    persistent = _without_transient_payloads(
        result,
        source_payloads=source,
        transient_payloads=transients,
    )
    assert [_text(payload) for payload in persistent] == ["真实用户输入", "真实回复"]
    assert "请务必使用工具" not in str(persistent)
    assert "重试提醒" not in str(persistent)


def test_tool_continuation_shape_survives_snapshot_reload() -> None:
    """闭合工具链的调用与回执按原顺序恢复。"""
    payloads = [
        LLMPayload(ROLE.USER, Text("查一下")),
        LLMPayload(
            ROLE.ASSISTANT,
            [ToolCall(id="call-1", name="action-kfc_reply", args={"q": "天气"})],
        ),
        LLMPayload(
            ROLE.TOOL_RESULT,
            ToolResult(value="晴", call_id="call-1", name="action-kfc_reply"),
        ),
        LLMPayload(ROLE.ASSISTANT, Text("今天是晴天")),
    ]
    restored = deserialize_snapshot(capture_snapshot(payloads))
    assert restored is not None
    assert [payload.role for payload in restored] == [
        ROLE.USER,
        ROLE.ASSISTANT,
        ROLE.TOOL_RESULT,
        ROLE.ASSISTANT,
    ]


@pytest.mark.asyncio
async def test_turn_decision_does_not_schedule_compression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """回合提交只记录决策；没有封存段时不启动后台压缩。"""
    session = KFCSession(user_id="u1", stream_id="stream-turn")
    session.append_context_entries([LLMPayload(ROLE.USER, Text("真实输入"))])

    class _Chatter:
        session_store = object()

        async def save_session(self, _session: KFCSession) -> None:
            return None

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record(*args: object, **kwargs: object):
        calls.append((args, kwargs))
        return False

    monkeypatch.setattr(
        SummaryService, "maybe_schedule_compression", staticmethod(record)
    )
    result = await commit_turn_decision(
        _Chatter(),
        Decision(has_meaningful_action=False),
        SimpleNamespace(message="纯文本", call_list=[]),
        session,
        KFCConfig(),
        SimpleNamespace(bot_id="bot"),
        is_final_timeout=False,
    )

    assert result.next_signal is not None
    assert calls == []


def test_context_contribution_supports_multimodal_transient_parts() -> None:
    """字符串贡献保持原行为；标准 parts 渲染为临时多模态 USER。"""
    string_contribution = _normalize_contribution(
        {"source": "plugin.text", "owner": "notice", "priority": 1, "content": "提醒"}
    )
    media_content: list[Content] = [Text("参考图"), Image("aW1hZ2U=")]
    media_contribution = _normalize_contribution(
        {
            "source": "plugin.media",
            "owner": "notice",
            "priority": 0,
            "content": media_content,
        }
    )
    invalid_media = _normalize_contribution(
        {
            "source": "plugin.invalid",
            "owner": "notice",
            "priority": 0,
            "content": ["not-content"],
        }
    )

    assert isinstance(string_contribution.content, str)
    string_payload = render_turn_contributions([string_contribution])
    assert isinstance(string_payload.content, list)
    assert isinstance(string_payload.content[0], Text)
    assert string_payload.content[0].text == "[附加上下文]\n提醒"

    assert media_contribution is not None
    media_payload = render_turn_contributions(
        [string_contribution, media_contribution]
    )
    assert media_payload.role == ROLE.USER
    assert isinstance(media_payload.content, list)
    assert any(isinstance(part, Text) and part.text == "参考图" for part in media_payload.content)
    assert any(isinstance(part, Image) and part.value == "aW1hZ2U=" for part in media_payload.content)
    assert invalid_media is None

    source_payloads = [LLMPayload(ROLE.USER, Text("真实输入"))]
    result_payloads = [
        source_payloads[0],
        media_payload,
        LLMPayload(ROLE.ASSISTANT, Text("回复")),
    ]
    persistent = _without_transient_payloads(
        result_payloads,
        source_payloads=source_payloads,
        transient_payloads=[media_payload],
    )
    assert [_text(payload) for payload in persistent] == ["真实输入", "回复"]
    assert "aW1hZ2U=" not in str(persistent)


def test_legacy_image_quota_config_is_removed() -> None:
    """无效历史图片配额不应再出现在配置模型中。"""
    from plugins.kokoro_flow_chatter.config import KFCConfig

    assert "max_images_per_payload" not in KFCConfig.GeneralSection.model_fields


@pytest.mark.asyncio
async def test_voice_call_backfills_then_reloads_from_snapshot() -> None:
    """voice call 直接写入快照；重启重建动态背景后仍能读到该记录。"""
    session = KFCSession(user_id="u1", stream_id="stream-voice")
    store = _FakeStore(session)
    plugin = KFCPlugin.__new__(KFCPlugin)
    plugin.config = KFCConfig()
    plugin.session_store = store
    handler = VoiceCallHistoryHandler.__new__(VoiceCallHistoryHandler)
    handler.plugin = plugin

    _, params = await handler.execute(
        "voice_call.ended",
        {
            "previous_chatter_signature": (
                "kokoro_flow_chatter:chatter:kokoro_flow_chatter"
            ),
            "caller_stream_id": "stream-voice",
            "duration_seconds": 61,
            "messages_in_call": [
                {"role": "user", "text": "通话里说什么？", "ts": 20},
                {"role": "assistant", "text": "说了测试。"},
            ],
        },
    )
    assert params is not None
    assert store.saved == [session]
    text = str(session.context_snapshot)
    assert "通话里说什么？" in text
    assert "说了测试。" in text

    chat_stream = SimpleNamespace(
        platform="qq",
        chat_type="private",
        bot_id="bot",
        bot_nickname="Bot",
        stream_name="对方",
        context=SimpleNamespace(history_messages=[]),
    )

    async def system_prompt(_stream: object, _extra: dict[str, str] | None) -> str:
        return "SYSTEM"

    _, _, history, _ = await render_initial_context(
        chat_stream=chat_stream,
        plan=InitialContextPlan(history_summary="动态摘要"),
        mental_log=None,
        serialized_context_snapshot=session.context_snapshot,
        build_system_prompt_fn=system_prompt,
    )
    payload_text = "\n".join(_text(payload) for payload in history)
    assert "通话里说什么？" in payload_text
    assert "说了测试。" in payload_text


def _deserialize_user_text(snapshot: list[dict[str, object]] | None) -> str:
    if not snapshot:
        return ""
    content = snapshot[0].get("content")
    if not isinstance(content, list) or not content:
        return ""
    first = content[0]
    if not isinstance(first, dict):
        return ""
    return str(first.get("text", ""))


class _FakeStore:
    def __init__(self, session: KFCSession) -> None:
        self.session = session
        self.saved: list[KFCSession] = []

    @asynccontextmanager
    async def lock(self, _stream_id: str):
        yield

    async def get_or_create(self, _stream_id: str) -> KFCSession:
        return self.session

    async def save(self, session: KFCSession) -> None:
        self.saved.append(session)
