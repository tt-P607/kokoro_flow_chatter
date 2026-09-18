"""未安装 response_guard 时 KFC 的独立运行保证。

KFC 的响应守卫是可选增强：``guard_hook`` 只通过 Service API 查找
``response_guard:service:response_guard``，不导入该插件的任何模块。
本文件用真实（非 mock 的）注册表状态验证「未安装」这一场景：

- 签名不存在时守卫判定为未命中，不查询服务实例、不抛异常；
- 旧的 ``config.toml``（不含 guard 字段）仍可正常加载，字段取默认值；
- 真实用户配置文件在 guard 字段缺失时也能加载。
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plugins.kokoro_flow_chatter.config import KFCConfig  # noqa: E402
from plugins.kokoro_flow_chatter.runtime.guard_hook import (  # noqa: E402
    GUARD_SIGNATURE,
    check_response_guard,
    guard_available,
)
from src.core.components.registry import get_global_registry  # noqa: E402

_REAL_CONFIG_PATH = _ROOT / "config" / "plugins" / "kokoro_flow_chatter" / "config.toml"
"""用户的实际配置文件路径。"""

_REFUSAL = "抱歉，我无法提供这类内容。"


class _FakeResponse:
    """最小化模拟带 payloads/message/call_list 的响应对象。"""

    def __init__(self) -> None:
        """构造一条含模型层拒答特征的响应。"""
        self.payloads: list[object] = []
        self.message = _REFUSAL
        self.reasoning_content = "安全政策不允许此类内容，我必须拒绝。"
        self.reasoning_parts: list[object] = []
        self.call_list: list[object] = []


def _assert_guard_not_registered() -> None:
    """确认当前进程的组件注册表中确实没有 response_guard 的 Service。"""
    if GUARD_SIGNATURE in get_global_registry()._components:
        pytest.skip("当前进程已注册 response_guard 组件，无法模拟未安装场景")


async def test_guard_available_is_false_without_plugin() -> None:
    """插件未安装时守卫判定为不可用。"""
    _assert_guard_not_registered()
    assert guard_available() is False


async def test_check_returns_not_blocked_without_plugin() -> None:
    """插件未安装时检测结果恒为未命中，模型层拒答也不拦截。"""
    _assert_guard_not_registered()
    result = await check_response_guard(
        _FakeResponse(), request_name="kokoro_flow_chatter"
    )
    assert result.blocked is False
    assert result.evidence == ()


async def test_check_does_not_raise_without_plugin() -> None:
    """插件未安装时检测不抛异常，主循环不会被中断。"""
    _assert_guard_not_registered()
    for request_name in ("kokoro_flow_chatter", "", "unknown_chatter"):
        result = await check_response_guard(
            _FakeResponse(), request_name=request_name
        )
        assert result.blocked is False


async def test_check_tolerates_minimal_response_shape() -> None:
    """未安装时即使响应对象字段缺失也不报错。"""
    _assert_guard_not_registered()
    result = await check_response_guard(
        SimpleNamespace(), request_name="kokoro_flow_chatter"
    )
    assert result.blocked is False


def test_legacy_config_without_guard_fields_loads(tmp_path: Path) -> None:
    """只有 general.enabled 的旧配置可以加载，guard 字段取默认值。"""
    path = tmp_path / "config.toml"
    path.write_text("[general]\nenabled = true\n", encoding="utf-8")

    config = KFCConfig.load(path, auto_update=False)

    assert config.general.enabled is True
    assert config.general.guard_enabled is True
    assert config.general.guard_max_retries == 1


def test_auto_update_appends_missing_guard_fields(tmp_path: Path) -> None:
    """框架的自动更新会把缺失的 guard 字段补进配置文件。"""
    path = tmp_path / "config.toml"
    path.write_text("[general]\nenabled = true\n", encoding="utf-8")

    config = KFCConfig.load(path, auto_update=True)

    assert config.general.guard_enabled is True
    assert config.general.guard_max_retries == 1
    with path.open("rb") as handle:
        rendered = tomllib.load(handle)
    assert rendered["general"]["guard_enabled"] is True
    assert rendered["general"]["guard_max_retries"] == 1


def test_real_config_file_loads_without_guard_fields() -> None:
    """用户实际配置文件（不含 guard 字段）保持可加载。"""
    if not _REAL_CONFIG_PATH.exists():
        pytest.skip("本机不存在 KFC 配置文件，跳过真实配置回归")

    with _REAL_CONFIG_PATH.open("rb") as handle:
        raw = tomllib.load(handle)
    if "guard_enabled" in raw.get("general", {}):
        pytest.skip("用户配置已包含 guard 字段，不属于本次回归场景")

    config = KFCConfig.load(_REAL_CONFIG_PATH, auto_update=False)

    assert isinstance(config, KFCConfig)
    assert config.general.guard_enabled is True
    assert config.general.guard_max_retries == 1
