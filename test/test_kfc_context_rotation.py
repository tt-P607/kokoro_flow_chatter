"""KFC 封存、重建和预算保护的定向测试。"""

from __future__ import annotations

import asyncio
import copy
import json
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plugins.kokoro_flow_chatter.config import KFCConfig  # noqa: E402
from plugins.kokoro_flow_chatter import context_budget  # noqa: E402
from plugins.kokoro_flow_chatter import diary_compressor  # noqa: E402
from plugins.kokoro_flow_chatter import session as session_module
from plugins.kokoro_flow_chatter.context import renderer  # noqa: E402
from plugins.kokoro_flow_chatter.context.sources.history_source import build_recent_chat_history  # noqa: E402
from plugins.kokoro_flow_chatter.prompts import modules as prompt_modules  # noqa: E402
from plugins.kokoro_flow_chatter.runtime import context_builder
from plugins.kokoro_flow_chatter.runtime import context_rotation  # noqa: E402
from plugins.kokoro_flow_chatter.runtime.request_view import build_request_view
from plugins.kokoro_flow_chatter.context.types import InitialContextPlan
from plugins.kokoro_flow_chatter.services import summary_service  # noqa: E402
from plugins.kokoro_flow_chatter.session import KFCSession, KFCSessionStore  # noqa: E402
from plugins.kokoro_flow_chatter.snapshot import deserialize_snapshot, serialize_payloads  # noqa: E402
from src.app.plugin_system.types import LLMPayload, ROLE, Text, ToolCall, ToolResult  # noqa: E402
from src.kernel.concurrency import get_task_manager  # noqa: E402
from src.kernel.llm.request import LLMRequest
from src.core.prompt.template import PromptTemplate  # noqa: E402


@pytest.mark.asyncio
async def test_request_view_sends_background_without_persisting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """实际请求追加与发送视图保留首条原话，但不保存背景和注入。"""
    manager = context_budget.KFCContextManager(background_payloads=[
        LLMPayload(ROLE.USER, Text("background only"))
    ])
    request = LLMRequest(
        cast(Any, [{"model_identifier": "gpt-4", "max_context": 1000}]),
        request_name="test", context_manager=manager,
    )
    request.add_payload(LLMPayload(ROLE.SYSTEM, Text("policy")))
    request.add_payload(LLMPayload(ROLE.USER, Text("【动态背景】 real user text")))
    request.add_payload(LLMPayload(ROLE.USER, Text("second user text")))
    transient = LLMPayload(ROLE.USER, Text("request only"))
    outgoing: list[LLMPayload] = []

    async def send(actual: LLMRequest, **kwargs: Any) -> Any:
        """只替换模型返回，保留真实请求类的追加和校验。"""
        outgoing.extend(actual.payloads)
        actual.add_payload(LLMPayload(ROLE.ASSISTANT, Text("answer")))
        return SimpleNamespace(payloads=actual.payloads, _consumed=True)

    monkeypatch.setattr(LLMRequest, "send", send)
    result = await build_request_view(request, [transient]).send()
    assert [payload.role for payload in request.payloads] == [ROLE.SYSTEM, ROLE.USER, ROLE.USER]
    assert "background only" in str(outgoing)
    assert "request only" in str(outgoing)
    snapshot = serialize_payloads(result.payloads)
    assert [entry["role"] for entry in snapshot] == ["user", "user", "assistant"]
    assert "real user text" in str(snapshot)
    assert "second user text" in str(snapshot)
    assert "background only" not in str(snapshot)
    assert "request only" not in str(snapshot)


@pytest.mark.asyncio
@pytest.mark.parametrize("tokens", [79, 80, 100, 101])
async def test_rotation_threshold_without_completed_round(
    monkeypatch: pytest.MonkeyPatch, tokens: int,
) -> None:
    """80% 无旧回合时继续；预算包括独立背景，只有真超限才拒绝。"""
    monkeypatch.setattr(
        context_budget, "count_payload_tokens",
        lambda payloads, **kwargs: sum(
            len(part.text) for payload in payloads for part in payload.content
            if isinstance(part, Text)
        ),
    )
    response = LLMRequest(
        cast(Any, [{"model_identifier": "gpt-4", "max_context": 100}]),
        context_manager=context_budget.KFCContextManager(background_payloads=[
            LLMPayload(ROLE.USER, Text("b" * (tokens - 35)))
        ]),
    )
    response.add_payload(LLMPayload(ROLE.USER, Text("u" * 10)))
    session = KFCSession(user_id="user", stream_id="stream")
    session.context_snapshot = serialize_payloads(response.payloads)
    before = copy.deepcopy(session.context_to_dict())
    store = _Store()
    operation = context_rotation.rotate_if_needed(
        cast(Any, SimpleNamespace(session_store=store)), response,
        cast(Any, SimpleNamespace()), KFCConfig(), session, response.model_set,
        [LLMPayload(ROLE.USER, Text("t" * 25))],
    )
    if tokens > 100:
        with pytest.raises(ValueError, match="动态背景=.*当前对话=.*临时注入="):
            await operation
    else:
        actual, tools, rotated = await operation
        assert actual is response and tools is None and not rotated
    assert session.context_to_dict() == before
    assert store.saved == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["rebuild", "cancel", "budget"])
async def test_rotation_rebuild_failure_preserves_all_authority(
    monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    """重建失败、取消或仍超预算时不提交任何新的权威状态。"""
    monkeypatch.setattr(context_rotation, "should_rotate", lambda *args: True)
    payloads = [
        LLMPayload(ROLE.USER, Text("old")),
        LLMPayload(ROLE.ASSISTANT, Text("answer")),
        LLMPayload(ROLE.USER, Text("current")),
    ]
    session = KFCSession(user_id="user", stream_id="stream")
    session.context_snapshot = serialize_payloads(payloads)
    session.completed_context_length = 2
    session.sealed_segments = [serialize_payloads(payloads[:2])]
    before = copy.deepcopy(session.context_to_dict())
    store = _Store()

    async def rebuild(*args: Any) -> Any:
        """分别模拟重建异常、取消与过大的重建背景。"""
        if failure == "rebuild":
            raise RuntimeError("rebuild failed")
        if failure == "cancel":
            raise asyncio.CancelledError
        monkeypatch.setattr(context_budget, "count_payload_tokens", lambda *a, **k: 1001)
        return SimpleNamespace(payloads=deserialize_snapshot(args[3].context_snapshot)), {}

    monkeypatch.setattr(context_rotation, "build_initial_request", rebuild)
    expected = {"rebuild": RuntimeError, "cancel": asyncio.CancelledError, "budget": ValueError}[failure]
    with pytest.raises(expected):
        await context_rotation.rotate_if_needed(
            cast(Any, SimpleNamespace(session_store=store)),
            SimpleNamespace(payloads=payloads), cast(Any, SimpleNamespace()),
            KFCConfig(), session, [{"model_identifier": "gpt-4", "max_context": 1000}], [],
        )
    assert session.context_to_dict() == before
    assert store.saved == []


class _Store:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.saved: list[dict[str, object]] = []

    @asynccontextmanager
    async def lock(self, stream_id: str):
        yield

    async def save_context(self, session: KFCSession) -> None:
        if self.fail:
            raise OSError("disk full")
        self.saved.append(copy.deepcopy(session.context_to_dict()))


@pytest.mark.asyncio
@pytest.mark.parametrize("completed_length", [None, 0, 2, -1, 3, 5, True, "2"])
async def test_completed_prefix_restart_preserves_transcript(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, completed_length: Any,
) -> None:
    """重启恢复可信完成前缀；缺失或损坏的边界不得删除合法原话。"""
    directory = tmp_path / "sessions"
    monkeypatch.setattr(session_module, "_STORAGE_DIR", str(directory))
    session = KFCSession(user_id="user", stream_id="stream", platform="test")
    session.context_snapshot = serialize_payloads([
        LLMPayload(ROLE.USER, Text("first")),
        LLMPayload(ROLE.ASSISTANT, Text("answer")),
        LLMPayload(ROLE.USER, Text("current")),
        LLMPayload(ROLE.ASSISTANT, Text("current answer")),
    ])
    session.sealed_segments = [copy.deepcopy(session.context_snapshot[:2])]
    await KFCSessionStore().save(session)
    path = directory / "context" / "test" / "user.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if completed_length is None:
        data.pop("completed_context_length")
    else:
        data["completed_context_length"] = completed_length
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = await KFCSessionStore().get("stream")
    assert loaded is not None
    assert loaded.context_snapshot == session.context_snapshot
    assert loaded.sealed_segments == session.sealed_segments
    assert loaded.completed_context_length == (
        completed_length if type(completed_length) is int and completed_length in (0, 2)
        else 0
    )
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["context_snapshot"] == session.context_snapshot
    assert persisted["sealed_segments"] == session.sealed_segments


@pytest.mark.asyncio
async def test_rotation_cancellation_during_atomic_write_restores_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """写盘过程中重复取消后，原对话、封存段及边界均恢复到磁盘。"""
    directory = tmp_path / "sessions"
    monkeypatch.setattr(session_module, "_STORAGE_DIR", str(directory))
    monkeypatch.setattr(context_rotation, "should_rotate", lambda *args: True)
    payloads = [
        LLMPayload(ROLE.USER, Text("old")),
        LLMPayload(ROLE.ASSISTANT, Text("answer")),
        LLMPayload(ROLE.USER, Text("current")),
    ]
    session = KFCSession(user_id="user", stream_id="stream", platform="test")
    session.context_snapshot = serialize_payloads(payloads)
    session.completed_context_length = 2
    store = KFCSessionStore()
    await store.save(session)
    path = directory / "context" / "test" / "user.json"
    before = json.loads(path.read_text(encoding="utf-8"))
    started = threading.Event()
    release = threading.Event()
    replacements: list[str] = []
    original_replace = session_module.os.replace

    def blocking_replace(source: Any, destination: Any) -> None:
        """只在目标上下文的第一轮原子替换前等待测试释放。"""
        if Path(destination) == path:
            replacements.append(str(destination))
            if len(replacements) == 1:
                started.set()
                assert release.wait(5)
        original_replace(source, destination)

    async def rebuild(*args: Any) -> Any:
        """从候选快照恢复实际 payload，不改权威会话。"""
        return SimpleNamespace(payloads=deserialize_snapshot(args[3].context_snapshot)), {}

    monkeypatch.setattr(session_module.os, "replace", blocking_replace)
    monkeypatch.setattr(context_rotation, "build_initial_request", rebuild)
    task = get_task_manager().create_task(context_rotation.rotate_if_needed(
        cast(Any, SimpleNamespace(session_store=store)), SimpleNamespace(payloads=payloads),
        cast(Any, SimpleNamespace()), KFCConfig(), session,
        [{"model_identifier": "gpt-4", "max_context": 1000}], [],
    )).task
    assert task is not None
    try:
        assert await asyncio.to_thread(started.wait, 3)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert session.context_to_dict() == before
    assert json.loads(path.read_text(encoding="utf-8")) == before
    assert len(replacements) == 2
    assert await store.get("stream") is session


def test_voice_entries_do_not_close_an_unfinished_round() -> None:
    """通话补写仅在现有前缀已完成时扩展完成边界。"""
    session = KFCSession(user_id="user", stream_id="stream")
    pair = [
        LLMPayload(ROLE.USER, Text("call")),
        LLMPayload(ROLE.ASSISTANT, Text("answer")),
    ]
    session.append_context_entries(pair, complete=True)
    assert session.completed_context_length == 2
    session.append_context_entries([LLMPayload(ROLE.USER, Text("pending"))])
    session.append_context_entries(pair, complete=True)
    assert session.completed_context_length == 2
    assert len(session.context_snapshot or []) == 5


@pytest.mark.asyncio
async def test_initial_request_preserves_first_real_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实请求组装后，动态背景不能吞掉活动链的首个 USER。"""
    session = KFCSession(user_id="user", stream_id="stream")
    snapshot = serialize_payloads([
        LLMPayload(ROLE.USER, Text("original question")),
        LLMPayload(ROLE.ASSISTANT, Text("original answer")),
    ])
    session.context_snapshot = copy.deepcopy(snapshot)
    stream = SimpleNamespace(
        platform="test", chat_type="private", bot_id="bot", bot_nickname="bot",
        stream_id="stream", stream_name="partner",
        context=SimpleNamespace(history_messages=[]),
    )

    async def system_prompt(*args: Any) -> str:
        """提供不依赖运行时配置的系统提示词。"""
        return "stable policy"

    async def inject_usables(request: LLMRequest) -> dict[str, object]:
        """保留真实请求组装，不注册外部组件。"""
        return {}

    monkeypatch.setattr(renderer, "build_system_prompt", system_prompt)
    monkeypatch.setattr(
        context_builder, "plan_initial_context",
        lambda **kwargs: InitialContextPlan(history_summary="background diary"),
    )
    monkeypatch.setattr(
        context_builder, "create_llm_request",
        lambda models, name, **kwargs: LLMRequest(
            models, request_name=name, context_manager=kwargs["context_manager"],
        ),
    )
    request, _ = await context_builder.build_initial_request(
        cast(Any, SimpleNamespace(stream_id="stream", inject_usables=inject_usables)),
        cast(Any, stream), KFCConfig(), session,
        [{"model_identifier": "gpt-4", "max_context": 1000}],
    )
    assert serialize_payloads(request.payloads) == snapshot
    assert session.context_snapshot == snapshot


def test_initial_diary_history_uses_three_day_window() -> None:
    """三天内的有效消息保留，窗口之外的消息不写入初始日记素材。"""
    start = time.time() - 3 * 86400
    end = start + 3 * 86400
    messages = [
        SimpleNamespace(time=start - 1, processed_plain_text="太早", sender_id="u", sender_name="用户", message_id="a"),
        SimpleNamespace(time=start, processed_plain_text="边界", sender_id="u", sender_name="用户", message_id="b"),
        SimpleNamespace(time=end + 1, processed_plain_text="未来", sender_id="u", sender_name="用户", message_id="c"),
        SimpleNamespace(time="invalid", processed_plain_text="无效", sender_id="u", sender_name="用户", message_id="d"),
    ]
    stream = cast(Any, SimpleNamespace(
        bot_id="bot", context=SimpleNamespace(history_messages=messages)
    ))

    assert "边界" in build_recent_chat_history(stream, start, before_ts=end)
    assert "太早" not in build_recent_chat_history(stream, start, before_ts=end)
    assert "未来" not in build_recent_chat_history(stream, start, before_ts=end)
    assert "无效" not in build_recent_chat_history(stream, start, before_ts=end)


@pytest.mark.asyncio
async def test_empty_diary_schedules_recent_history_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """空日记从最近三天的框架消息初始化，不重复总结已有日记。"""
    now = time.time()
    messages = [
        SimpleNamespace(time=now - 4 * 86400, processed_plain_text="四天前", sender_id="u", sender_name="用户", message_id="m1"),
        SimpleNamespace(time=now - 3600, processed_plain_text="一小时前", sender_id="u", sender_name="用户", message_id="m2"),
    ]
    stream = cast(Any, SimpleNamespace(
        bot_id="bot", context=SimpleNamespace(history_messages=messages)
    ))
    session = KFCSession(user_id="u", stream_id="bootstrap-stream")
    recorded: list[str] = []

    async def bootstrap(
        _session: KFCSession, _config: KFCConfig, _stream: Any,
        history_text: str, *, session_store: Any,
    ) -> bool:
        recorded.append(history_text)
        return True

    monkeypatch.setattr(summary_service, "bootstrap_diary", bootstrap)
    assert summary_service.SummaryService.maybe_schedule_compression(
        session, KFCConfig(), stream, session_store=cast(Any, _Store())
    )
    task_id = summary_service.SummaryService._task_ids[session.stream_id]
    task = get_task_manager().get_task(task_id).task
    assert task is not None
    await task
    assert len(recorded) == 1
    assert "一小时前" in recorded[0]
    assert "四天前" not in recorded[0]

    session.history_summary = "已有日记"
    assert not summary_service.SummaryService.maybe_schedule_compression(
        session, KFCConfig(), stream, session_store=cast(Any, _Store())
    )


@pytest.mark.asyncio
async def test_empty_diary_without_recent_history_does_not_schedule() -> None:
    """没有近三天有效消息时，不请求模型也不建立后台任务。"""
    stream = cast(Any, SimpleNamespace(
        bot_id="bot", context=SimpleNamespace(history_messages=[])
    ))
    session = KFCSession(user_id="u", stream_id="empty-history")
    assert not summary_service.SummaryService.maybe_schedule_compression(
        session, KFCConfig(), stream, session_store=cast(Any, _Store())
    )


@pytest.mark.asyncio
async def test_diary_template_renders_persona_without_chat_tool_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """日记与对话共享人格来源，但仅日记模板排除工具输出协议。"""
    personality = SimpleNamespace(
        nickname="星野", alias_names=["小星"], personality_core="认真", personality_side="偶尔害羞",
        identity="旅人", background_story="我曾和朋友一起远行很久，留下很多回忆。",
        reply_style="简洁自然", safety_guidelines=["尊重隐私"],
        negative_behaviors=["不得编造事实"],
    )
    templates: dict[str, PromptTemplate] = {}

    def register_template(
        name: str, template: str, policies: dict[str, Any]
    ) -> PromptTemplate:
        registered = PromptTemplate(name=name, template=template, policies=policies)
        templates[name] = registered
        return registered

    monkeypatch.setattr(
        prompt_modules, "get_core_config", lambda: SimpleNamespace(personality=personality)
    )
    monkeypatch.setattr(prompt_modules, "get_or_create", register_template)
    monkeypatch.setattr(
        renderer, "get_template", lambda name: templates[name].clone()
    )
    prompt_modules.register_kfc_prompts()
    stream = cast(Any, SimpleNamespace(
        platform="qq", chat_type="private", bot_id="bot", stream_id="test"
    ))
    diary = await renderer.build_diary_system_prompt(stream)
    chat = await renderer.build_system_prompt(stream)

    for expected in (
        "星野", "小星", "认真", "偶尔害羞", "旅人", "远行很久",
        "简洁自然", "尊重隐私", "不得编造事实",
    ):
        assert expected in diary
    assert "不执行" in diary or "不得覆盖" in diary
    assert "只输出日记正文" in diary
    assert "严禁在文本区域输出任何内容" not in diary
    assert "action-kfc_reply" not in diary
    assert "严禁在文本区域输出任何内容" in chat


@pytest.mark.asyncio
async def test_diary_template_missing_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模板未注册时禁止发送无人设和日记规范的请求。"""
    monkeypatch.setattr(renderer, "get_template", lambda name: None)
    stream = cast(Any, SimpleNamespace(
        platform="qq", chat_type="private", bot_id="bot", stream_id="test"
    ))
    with pytest.raises(ValueError, match="日记系统提示词未注册"):
        await renderer.build_diary_system_prompt(stream)


@pytest.mark.asyncio
async def test_diary_requests_use_dedicated_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """初始日记和封存压缩都不应继承对话工具调用协议。"""
    payloads: list[list[LLMPayload]] = []

    class _Request:
        def __init__(self) -> None:
            self.items: list[LLMPayload] = []
            payloads.append(self.items)

        def add_payload(self, payload: LLMPayload) -> None:
            self.items.append(payload)

        async def send(self, *, stream: bool) -> Any:
            async def content() -> str:
                return "我记得这次聊天。"

            return content()

    async def diary_prompt(stream: Any) -> str:
        return "我是测试角色。只写第一人称日记正文，不使用工具调用。"

    monkeypatch.setattr(diary_compressor, "build_diary_system_prompt", diary_prompt)
    monkeypatch.setattr(diary_compressor, "get_model_set_by_task", lambda name: [{}])
    monkeypatch.setattr(diary_compressor, "create_llm_request", lambda *args, **kwargs: _Request())

    session = KFCSession(user_id="u", stream_id="diary-prompt")
    stream = cast(Any, SimpleNamespace())
    store = cast(Any, _Store())
    assert await diary_compressor.bootstrap_diary(
        session, KFCConfig(), stream, "聊天记录很长，需要记住我们说过的重要事情。",
        session_store=store,
    )
    session.sealed_segments = [serialize_payloads([
        LLMPayload(ROLE.USER, Text("又进行了一次漫长的聊天，约定下次见面。"))
    ])]
    assert await diary_compressor.compress_history(
        session, KFCConfig(), stream, session_store=store
    )
    assert len(payloads) == 2
    for request_payloads in payloads:
        assert request_payloads[0].role == ROLE.SYSTEM
        assert request_payloads[0].content == [
            Text("我是测试角色。只写第一人称日记正文，不使用工具调用。")
        ]
        assert request_payloads[1].role == ROLE.USER


@pytest.mark.asyncio
async def test_initial_diary_commit_failure_preserves_pending_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """初始日记写盘失败时不污染内存日记、封存队列或活动链。"""
    session = KFCSession(user_id="u", stream_id="bootstrap-save")
    session.context_snapshot = serialize_payloads([LLMPayload(ROLE.USER, Text("活动原话"))])
    segment = serialize_payloads([LLMPayload(ROLE.USER, Text("此前封存的原话"))])
    session.sealed_segments = [segment]
    store: Any = _Store(fail=True)

    class _Request:
        def add_payload(self, payload: LLMPayload) -> None:
            return None

        async def send(self, *, stream: bool) -> Any:
            async def content() -> str:
                return "我记得旧事。"

            return content()

    async def system_prompt(stream: object) -> str:
        return "actor"

    monkeypatch.setattr(diary_compressor, "build_diary_system_prompt", system_prompt)
    monkeypatch.setattr(diary_compressor, "get_model_set_by_task", lambda name: [{}])
    monkeypatch.setattr(diary_compressor, "create_llm_request", lambda *args, **kwargs: _Request())
    history = "[2025-01-01 12:00] 用户说：我们聊了很久，也约定了下一次见面的日期。"
    assert not await diary_compressor.bootstrap_diary(
        session, KFCConfig(), cast(Any, SimpleNamespace()), history, session_store=store
    )
    assert session.history_summary == ""
    assert session.sealed_segments == [segment]
    assert "活动原话" in str(session.context_snapshot)
    assert store.saved == []

    store.fail = False
    assert await diary_compressor.bootstrap_diary(
        session, KFCConfig(), cast(Any, SimpleNamespace()), history, session_store=store
    )
    assert session.history_summary == "我记得旧事。"
    assert store.saved[0]["sealed_segments"] == [segment]


@pytest.mark.asyncio
async def test_initial_diary_model_failure_preserves_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """初始日记模型连续失败时不清理待压缩原话。"""
    session = KFCSession(user_id="u", stream_id="bootstrap-model")
    segment = serialize_payloads([LLMPayload(ROLE.USER, Text("封存原话"))])
    session.sealed_segments = [segment]
    store: Any = _Store()
    attempts: list[int] = []

    class _Request:
        def add_payload(self, payload: LLMPayload) -> None:
            return None

        async def send(self, *, stream: bool) -> Any:
            attempts.append(1)
            raise RuntimeError("model unavailable")

    async def system_prompt(stream: object) -> str:
        return "actor"

    monkeypatch.setattr(diary_compressor, "build_diary_system_prompt", system_prompt)
    monkeypatch.setattr(diary_compressor, "get_model_set_by_task", lambda name: [{}])
    monkeypatch.setattr(diary_compressor, "create_llm_request", lambda *args, **kwargs: _Request())
    assert not await diary_compressor.bootstrap_diary(
        session, KFCConfig(), cast(Any, SimpleNamespace()),
        "最近三天的聊天", session_store=store,
    )
    assert len(attempts) == 3
    assert session.history_summary == ""
    assert session.sealed_segments == [segment]
    assert store.saved == []


@pytest.mark.asyncio
async def test_initial_diary_keeps_new_turns_while_model_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """初始化期间新到的活动回合和封存段随日记一起持久化。"""
    session = KFCSession(user_id="u", stream_id="bootstrap-race")
    store: Any = _Store()
    started = asyncio.Event()
    finish = asyncio.Event()

    class _Request:
        def add_payload(self, payload: LLMPayload) -> None:
            return None

        async def send(self, *, stream: bool) -> Any:
            started.set()
            await finish.wait()

            async def content() -> str:
                return "我记得那次谈话。"

            return content()

    async def system_prompt(stream: object) -> str:
        return "actor"

    monkeypatch.setattr(diary_compressor, "build_diary_system_prompt", system_prompt)
    monkeypatch.setattr(diary_compressor, "get_model_set_by_task", lambda name: [{}])
    monkeypatch.setattr(diary_compressor, "create_llm_request", lambda *args, **kwargs: _Request())
    task = asyncio.create_task(diary_compressor.bootstrap_diary(
        session, KFCConfig(), cast(Any, SimpleNamespace()),
        "最近三天的聊天记录", session_store=store,
    ))
    await asyncio.wait_for(started.wait(), timeout=2)
    session.append_context_entries([LLMPayload(ROLE.USER, Text("新活动原话"))])
    segment = serialize_payloads([LLMPayload(ROLE.USER, Text("新封存原话"))])
    session.sealed_segments.append(segment)
    finish.set()
    assert await task
    assert store.saved[0]["sealed_segments"] == [segment]
    assert "新活动原话" in str(store.saved[0]["context_snapshot"])
    assert session.sealed_segments == [segment]


@pytest.mark.asyncio
async def test_initial_diary_precedes_pending_segment_compression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一后台任务先提交历史日记，再基于它压缩封存原话。"""
    session = KFCSession(user_id="u", stream_id="bootstrap-queued")
    segment = serialize_payloads([
        LLMPayload(ROLE.USER, Text("后来我们再次讨论了很长时间，约定一起去看展览。"))
    ])
    session.sealed_segments = [segment]
    messages = [SimpleNamespace(
        time=time.time() - 60, processed_plain_text="我们第一次聊到了旅行和展览，聊了很长时间。",
        sender_id="u", sender_name="用户", message_id="m1",
    )]
    stream = cast(Any, SimpleNamespace(
        bot_id="bot", context=SimpleNamespace(history_messages=messages)
    ))
    store: Any = _Store()
    responses = iter(["我记得初次谈到旅行。", "我记得旅行，也约定去看展览。"])

    class _Request:
        def add_payload(self, payload: LLMPayload) -> None:
            return None

        async def send(self, *, stream: bool) -> Any:
            async def content() -> str:
                return next(responses)

            return content()

    async def system_prompt(stream: object) -> str:
        return "actor"

    monkeypatch.setattr(diary_compressor, "build_diary_system_prompt", system_prompt)
    monkeypatch.setattr(diary_compressor, "get_model_set_by_task", lambda name: [{}])
    monkeypatch.setattr(diary_compressor, "create_llm_request", lambda *args, **kwargs: _Request())
    assert summary_service.SummaryService.maybe_schedule_compression(
        session, KFCConfig(), stream, session_store=store
    )
    task_id = summary_service.SummaryService._task_ids[session.stream_id]
    task = get_task_manager().get_task(task_id).task
    assert task is not None
    await task

    assert len(store.saved) == 2
    assert store.saved[0]["history_summary"] == "我记得初次谈到旅行。"
    assert store.saved[0]["sealed_segments"] == [segment]
    assert store.saved[1]["history_summary"] == "我记得旅行，也约定去看展览。"
    assert store.saved[1]["sealed_segments"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_rotation_commits_old_turn_and_rebuilds_from_latest(
    monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    """封存与重建原子交接；保存失败时不修改权威会话状态。"""
    monkeypatch.setattr(context_rotation, "should_rotate", lambda *args: True)
    payloads = [
        LLMPayload(ROLE.USER, Text("old question")),
        LLMPayload(ROLE.ASSISTANT, Text("old answer")),
        LLMPayload(ROLE.USER, Text("current question")),
    ]
    session = KFCSession(user_id="u", stream_id="s")
    session.context_snapshot = serialize_payloads(payloads)
    session.completed_context_length = 2
    original = session.context_snapshot
    store = _Store(fail=fail)
    chatter = SimpleNamespace(session_store=store)

    async def rebuild(*args):
        candidate = args[3]
        assert candidate.context_snapshot is not original
        return SimpleNamespace(payloads=deserialize_snapshot(candidate.context_snapshot)), {"tool": object()}

    monkeypatch.setattr(context_rotation, "build_initial_request", rebuild)
    operation = context_rotation.rotate_if_needed(
        cast(Any, chatter), SimpleNamespace(payloads=payloads), cast(Any, SimpleNamespace()),
        KFCConfig(), session, [{"model_identifier": "gpt-4", "max_context": 1000}], []
    )
    if fail:
        with pytest.raises(OSError, match="disk full"):
            await operation
        assert session.context_snapshot == original
        assert session.sealed_segments == []
        return

    rebuilt, tools, rotated = await operation
    assert rotated and "tool" in tools
    assert len(store.saved) == 1
    assert [entry["role"] for entry in cast(Any, store.saved[0]["sealed_segments"])[0]] == [
        "user", "assistant"
    ]
    assert len(rebuilt.payloads) == 1
    assert rebuilt.payloads[0].content == [Text("current question")]


@pytest.mark.asyncio
async def test_rotation_keeps_closed_tool_segment_in_current_turn(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """工具调用和回执留在活动回合；旧完整回合仍能先封存。"""
    monkeypatch.setattr(context_rotation, "should_rotate", lambda *args: True)
    session = KFCSession(user_id="u", stream_id="s")
    session.completed_context_length = 2
    store = _Store()
    payloads = [
        LLMPayload(ROLE.USER, Text("old question")),
        LLMPayload(ROLE.ASSISTANT, Text("old answer")),
        LLMPayload(ROLE.USER, Text("current question")),
        LLMPayload(ROLE.ASSISTANT, ToolCall(id="c1", name="lookup", args={})),
        LLMPayload(ROLE.TOOL_RESULT, ToolResult(value="result", call_id="c1", name="lookup")),
    ]

    async def rebuild(*args):
        return SimpleNamespace(payloads=deserialize_snapshot(args[3].context_snapshot)), {}

    monkeypatch.setattr(context_rotation, "build_initial_request", rebuild)
    _, _, rotated = await context_rotation.rotate_if_needed(
        cast(Any, SimpleNamespace(session_store=store)), SimpleNamespace(payloads=payloads),
        cast(Any, SimpleNamespace()), KFCConfig(), session,
        [{"model_identifier": "gpt-4", "max_context": 1000}], []
    )
    assert rotated
    assert len(session.sealed_segments) == 1
    assert session.context_snapshot is not None
    assert [entry["role"] for entry in session.context_snapshot] == [
        "user", "assistant", "tool_result"
    ]


@pytest.mark.asyncio
async def test_rotation_refuses_unpaired_tool_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """活动回合的工具调用未配对时，不得持久化一个会被恢复逻辑裁掉的尾部。"""
    monkeypatch.setattr(context_rotation, "should_rotate", lambda *args: True)
    session = KFCSession(user_id="u", stream_id="s")
    session.completed_context_length = 2
    store = _Store()
    payloads = [
        LLMPayload(ROLE.USER, Text("old question")),
        LLMPayload(ROLE.ASSISTANT, Text("old answer")),
        LLMPayload(ROLE.USER, Text("current question")),
        LLMPayload(ROLE.ASSISTANT, ToolCall(id="c1", name="lookup", args={})),
    ]
    with pytest.raises(ValueError, match="未闭合工具段"):
        await context_rotation.rotate_if_needed(
            cast(Any, SimpleNamespace(session_store=store)), SimpleNamespace(payloads=payloads),
            cast(Any, SimpleNamespace()), KFCConfig(), session,
            [{"model_identifier": "gpt-4", "max_context": 1000}], []
        )
    assert store.saved == []
    assert session.sealed_segments == []


@pytest.mark.asyncio
async def test_diary_commit_does_not_overwrite_new_active_turns(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """后台结果只消费对应的封存段，不覆盖生成期间的新对话。"""
    store = _Store()
    session = KFCSession(user_id="u", stream_id="s")
    session.sealed_segments = [serialize_payloads([
        LLMPayload(ROLE.USER, Text("very old question")),
        LLMPayload(ROLE.ASSISTANT, Text("very old answer")),
    ])]
    session.context_snapshot = serialize_payloads([LLMPayload(ROLE.USER, Text("current"))])
    started = asyncio.Event()
    finish = asyncio.Event()

    class _Request:
        def add_payload(self, payload: LLMPayload) -> None:
            return None

        async def send(self, *, stream: bool):
            started.set()
            await finish.wait()

            async def content():
                return "我记得那次谈话。"

            return content()

    async def system_prompt(stream: object) -> str:
        return "actor"

    monkeypatch.setattr(diary_compressor, "build_diary_system_prompt", system_prompt)
    monkeypatch.setattr(diary_compressor, "get_model_set_by_task", lambda name: [{}])
    monkeypatch.setattr(diary_compressor, "create_llm_request", lambda *args, **kwargs: _Request())
    task = asyncio.create_task(diary_compressor.compress_history(
        session, KFCConfig(), SimpleNamespace(), session_store=store
    ))
    await asyncio.wait_for(started.wait(), timeout=2)
    session.append_context_entries([LLMPayload(ROLE.ASSISTANT, Text("new answer"))])
    finish.set()
    assert await task
    assert session.sealed_segments == []
    assert "new answer" in str(session.context_snapshot)
    assert store.saved[-1]["context_snapshot"] == session.context_snapshot


@pytest.mark.asyncio
async def test_diary_failed_three_times_keeps_sealed_source(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """三次模型失败后不删封存原文和已有日记。"""
    store = _Store()
    session = KFCSession(user_id="u", stream_id="s", history_summary="我已有日记")
    session.sealed_segments = [serialize_payloads([LLMPayload(ROLE.USER, Text("old"))])]
    attempts: list[int] = []

    class _Request:
        def add_payload(self, payload: LLMPayload) -> None:
            return None

        async def send(self, *, stream: bool):
            attempts.append(1)
            raise RuntimeError("model unavailable")

    async def system_prompt(stream: object) -> str:
        return "actor"

    monkeypatch.setattr(diary_compressor, "build_diary_system_prompt", system_prompt)
    monkeypatch.setattr(diary_compressor, "get_model_set_by_task", lambda name: [{}])
    monkeypatch.setattr(diary_compressor, "create_llm_request", lambda *args, **kwargs: _Request())
    assert not await diary_compressor.compress_history(
        session, KFCConfig(), SimpleNamespace(), session_store=store
    )
    assert len(attempts) == 3
    assert len(session.sealed_segments) == 1
    assert session.history_summary == "我已有日记"
    assert store.saved == []


@pytest.mark.asyncio
async def test_diary_write_failure_keeps_source_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模型成功但落盘失败，日记和封存段也不能在内存中提前提交。"""
    store = _Store(fail=True)
    session = KFCSession(user_id="u", stream_id="s", history_summary="旧日记")
    segment = serialize_payloads([
        LLMPayload(ROLE.USER, Text("那天我讲了一个很长的经历，其中有明确的重要约定和日期。"))
    ])
    session.sealed_segments = [segment]

    class _Request:
        def add_payload(self, payload: LLMPayload) -> None:
            return None

        async def send(self, *, stream: bool):
            async def content():
                return "我记得约定。"

            return content()

    async def system_prompt(stream: object) -> str:
        return "actor"

    monkeypatch.setattr(diary_compressor, "build_diary_system_prompt", system_prompt)
    monkeypatch.setattr(diary_compressor, "get_model_set_by_task", lambda name: [{}])
    monkeypatch.setattr(diary_compressor, "create_llm_request", lambda *args, **kwargs: _Request())
    assert not await diary_compressor.compress_history(
        session, KFCConfig(), SimpleNamespace(), session_store=store
    )
    assert session.history_summary == "旧日记"
    assert session.sealed_segments == [segment]
    assert store.saved == []

    store.fail = False
    assert await diary_compressor.compress_history(
        session, KFCConfig(), SimpleNamespace(), session_store=store
    )
    assert session.history_summary == "我记得约定。"
    assert session.sealed_segments == []


@pytest.mark.asyncio
async def test_minimum_model_budget_rotates_and_never_trims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """按最小回退窗口提前轮换，超限时拒绝静默丢掉历史。"""
    models = [
        {"model_identifier": "small", "max_context": 100},
        {"model_identifier": "large", "max_context": 300},
    ]
    payloads = [LLMPayload(ROLE.USER, Text("一整条原话"))]
    monkeypatch.setattr(context_budget, "count_payload_tokens", lambda *args, **kwargs: 85)
    assert context_budget.should_rotate(payloads, models)

    monkeypatch.setattr(context_budget, "count_payload_tokens", lambda *args, **kwargs: 101)
    manager = context_budget.KFCContextManager()
    with pytest.raises(ValueError, match="超出模型预算"):
        await manager.prepare_payloads_for_model(payloads, models[0])
    assert len(payloads) == 1