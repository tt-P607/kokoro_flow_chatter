"""KokoroFlow Chatter 配置、生命周期和兼容边界测试。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plugins.kokoro_flow_chatter import framework_compat  # noqa: E402
from plugins.kokoro_flow_chatter.config import KFCConfig  # noqa: E402
from plugins.kokoro_flow_chatter.plugin import KFCPlugin  # noqa: E402
from plugins.kokoro_flow_chatter.services.summary_service import (  # noqa: E402
    SummaryService,
)


def test_config_mutable_defaults_are_isolated() -> None:
    """列表配置默认值不应在实例之间共享。"""

    first = KFCConfig()
    second = KFCConfig()

    first.general.models.append("model-a")
    first.general.blocked_tools.append("custom-tool")

    assert second.general.models == []
    assert "custom-tool" not in second.general.blocked_tools


def test_wait_rules_normalize_reversed_bounds() -> None:
    """等待上下限写反时仍应稳定夹取到有效区间。"""

    wait = KFCConfig.WaitSection(min_seconds=100.0, max_seconds=10.0)

    assert wait.apply_rules(5.0, 0) == 10.0
    assert wait.apply_rules(50.0, 0) == 50.0
    assert wait.apply_rules(500.0, 0) == 100.0


def test_config_rejects_invalid_probability_and_quiet_hour() -> None:
    """概率和勿扰时间应在配置加载阶段拒绝无效值。"""

    with pytest.raises(ValidationError):
        KFCConfig.ProactiveSection(trigger_probability=1.5)
    with pytest.raises(ValidationError):
        KFCConfig.ProactiveSection(quiet_hours_start="25:00")


def test_manifest_matches_registered_components_and_version_source() -> None:
    """manifest 应覆盖真实组件，且插件类不再维护重复版本字段。"""

    manifest = json.loads(
        (_ROOT / "plugins/kokoro_flow_chatter/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    plugin = KFCPlugin(KFCConfig())
    component_names = {component.name for component in plugin.get_components()}
    declared_names = {
        item["component_name"]
        for item in manifest["include"]
        if item["component_type"] != "config"
    }

    assert manifest["version"] == "2.2.0"
    assert "plugin_version" not in KFCPlugin.__dict__
    assert component_names == declared_names


@pytest.mark.asyncio
async def test_plugin_unload_cancels_runtime_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """插件卸载应取消延迟任务、周期调度和摘要任务。"""

    cancelled_task_ids: list[str] = []
    removed_schedule_ids: list[str] = []

    class _TaskManager:
        def cancel_task(self, task_id: str) -> bool:
            cancelled_task_ids.append(task_id)
            return True

    class _Scheduler:
        async def remove_schedule(self, schedule_id: str) -> bool:
            removed_schedule_ids.append(schedule_id)
            return True

        async def remove_schedule_by_name(self, task_name: str) -> bool:
            removed_schedule_ids.append(task_name)
            return True

    monkeypatch.setattr(
        "plugins.kokoro_flow_chatter.plugin.get_task_manager",
        lambda: _TaskManager(),
    )
    monkeypatch.setattr(
        "src.kernel.scheduler.get_unified_scheduler",
        lambda: _Scheduler(),
    )
    monkeypatch.setattr(SummaryService, "cancel_all", classmethod(lambda cls: 2))

    plugin = KFCPlugin(KFCConfig())
    plugin._scheduler_init_task_id = "init-task"
    plugin._proactive_schedule_id = "schedule-id"

    await plugin.on_plugin_unloaded()

    assert cancelled_task_ids == ["init-task"]
    assert removed_schedule_ids == ["schedule-id"]
    assert plugin._scheduler_init_task_id is None
    assert plugin._proactive_schedule_id is None


@pytest.mark.asyncio
async def test_summary_service_deduplicates_stream_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一聊天流只能同时存在一个摘要压缩任务。"""

    from src.kernel.concurrency import get_task_manager

    started = asyncio.Event()
    release = asyncio.Event()

    async def _fake_compress(*args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)
        started.set()
        await release.wait()

    monkeypatch.setattr(
        "plugins.kokoro_flow_chatter.services.summary_service.compress_history",
        _fake_compress,
    )
    SummaryService._task_ids.clear()
    session = cast(
        Any,
        SimpleNamespace(
            stream_id="stream-1",
            history_summary="",
            compress_round_count=0,
        ),
    )
    config = KFCConfig()
    chat_stream = cast(Any, SimpleNamespace())

    first = SummaryService.maybe_schedule_compression(session, config, chat_stream)
    await started.wait()
    second = SummaryService.maybe_schedule_compression(session, config, chat_stream)

    assert first is True
    assert second is False

    release.set()
    task_info = get_task_manager().get_task(SummaryService._task_ids["stream-1"])
    assert task_info.task is not None
    await task_info.task
    assert "stream-1" not in SummaryService._task_ids


@pytest.mark.asyncio
async def test_framework_compat_delegates_to_internal_managers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """兼容边界应以明确参数委托媒体和流循环管理器。"""

    media_calls: list[tuple[str, Any]] = []
    loop_calls: list[str] = []

    class _MediaManager:
        def skip_recognition_for_stream(
            self,
            stream_id: str,
            media_types: list[str] | None = None,
        ) -> None:
            media_calls.append(("skip", (stream_id, media_types)))

        def unskip_recognition_for_stream(self, stream_id: str) -> None:
            media_calls.append(("clear", stream_id))

    class _LoopManager:
        async def start_stream_loop(self, stream_id: str) -> None:
            loop_calls.append(stream_id)

    monkeypatch.setattr(
        "src.core.managers.media_manager.get_media_manager",
        lambda: _MediaManager(),
    )
    monkeypatch.setattr(
        "src.core.transport.distribution.stream_loop_manager.get_stream_loop_manager",
        lambda: _LoopManager(),
    )

    framework_compat.set_stream_recognition_skip("stream-1", ["image"])
    framework_compat.clear_stream_recognition_skip("stream-1")
    await framework_compat.start_stream_loop("stream-1")

    assert media_calls == [
        ("skip", ("stream-1", ["image"])),
        ("clear", "stream-1"),
    ]
    assert loop_calls == ["stream-1"]
