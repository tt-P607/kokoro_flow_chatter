# KFC 重构审计与执行手册

> 本文档是 [REFACTOR_PLAN.md](REFACTOR_PLAN.md) 的**配套执行版**。
> REFACTOR_PLAN.md 描述目标架构与重构思路，本文件聚焦于**已发现的具体问题、修复步骤、进度跟踪**。
> 所有修改必须严格遵守"项目代码规范摘要"小节。

---

## 项目代码规范摘要（始终遵守）

> 来源：[代码规范.md](../../代码规范.md) + [.github/copilot-instructions.md](../../.github/copilot-instructions.md) + [.github/prompts/plugin-dev.prompt.md](../../.github/prompts/plugin-dev.prompt.md)

### 强制要求

1. **类型注解**：所有函数、方法的参数与返回值必须有类型注解。
2. **文档字符串**：所有函数、类、文件必须有 docstring / 文件简介。
3. **PEP 8**：命名/分层风格与 `src/kernel/`、`src/core/`、`src/app/` 保持一致。
4. **Python ≥ 3.11**：依赖与运行使用 `uv`。
5. **异步任务**：统一通过 `task_manager`（`src/kernel/concurrency`），禁用 `asyncio.create_task()`。
6. **配置系统**：优先 `ConfigBase` / `SectionBase` + `config_section` 模式。
7. **数据访问**：优先 `CRUDBase` 与 `QueryBuilder`。
8. **测试覆盖**：新增 `src/` 代码必须补对应单元测试。

### 高频违规速查（动手前核对）

| 违规 | 正确做法 |
|------|---------|
| 插件直接 `from src.core.managers.*` 导入 | 使用 `src.app.plugin_system.api.*` |
| 覆盖 `BaseCommand.execute()` 方法 | 使用 `@cmd_route` 路由分发 |
| 在 `src/app/runtime/` 等模块写业务逻辑 | 业务逻辑放 `plugins/` |
| 用魔法字符串做逻辑判断 | 使用结构化 API 返回值类型判断 |
| `getattr(obj, "field", None)` 访问明确类型字段 | 直接属性访问 |
| 滥用 `try/except` 兜底 fallback | 修复根因，不要隐藏错误 |
| 直接 `response.payloads.append(...)` 绕过 `add_payload` | 使用 `add_payload` 或带保护的 helper |
| 在提交里出现明文密钥 / 数据库密码 | 严禁 |

### KFC 插件特有原则

- **桥接 `tool_result → user`**：任何向 `response.payloads` 追加 USER 之前，必须确保末尾不是 TOOL_RESULT。统一使用 [`services/context_bridge.py`](services/context_bridge.py) 中的 `ensure_tool_chain_closed` / `safe_add_payload`，禁止再手写 `if last == TOOL_RESULT: append(ASSISTANT(...))`。
- **chain 持久化格式约束**：写入 `session.chain_payloads` 时，含 `tool_calls` 的 assistant 条目，`text` 字段不得为空（用占位文本 `"好的。"` 兜底），否则跨 session 还原会触发链路错误。
- **`extra_payload` 直接 append**：在 [`runtime/orchestrator.py`](runtime/orchestrator.py) 中已用 `ensure_tool_chain_closed` 保护，**不要**改回普通 `add_payload`（会因末尾 USER 合并打乱 index 删除逻辑）。

---

## 一、已发现问题清单

### A. 链路与持久化（已修复）

- [x] **A1 [严重] `tool_result → user` 直连** — `LLMContextError: tool_result 后不能直接跟 user`
  - 根因 1：[`context/sources/history_source.py`](context/sources/history_source.py) `restore_chain_payloads` 在 assistant 条目 `text` 为空时跳过桥接 ASSISTANT。
  - 根因 2：[`runtime/turn_controller.py`](runtime/turn_controller.py) `commit_turn_decision` 把 `text=""` 的 assistant 条目写入 `session.chain_payloads`，跨 session 复现 bug。
  - 修复：占位 `"好的。"` 兜底 + chain 保存端兜底。
- [x] **A2 桥接逻辑分散在 3 处** — turn_controller 2 处、timeout_service 1 处重复 `if last==TOOL_RESULT: add(ASSISTANT)`。
  - 修复：抽出 [`services/context_bridge.py`](services/context_bridge.py)，三处统一调 `ensure_tool_chain_closed`。
- [x] **A3 orchestrator 直接 `payloads.append(extra_payload)`** — 绕过 `add_payload` 校验，依赖前面手插桥。
  - 修复：append 前调 `ensure_tool_chain_closed`。

### B. 违规导入框架内部（API 缺口，先记录不修）

- [ ] **B1** [`chatter.py:296`](chatter.py)、[`chatter.py:309`](chatter.py)、[`plugin.py:108`](plugin.py)：`from src.core.managers.media_manager` 调 `skip_vlm_for_stream`。
  - **缺口**：`src/app/plugin_system/api/media_api.py` 未暴露此能力。
  - **决策**：等框架补 API 后再迁移。
- [ ] **B2** [`handlers/proactive_handler.py:168`](handlers/proactive_handler.py)：`from src.core.transport.distribution.stream_loop_manager` 调 `start_stream_loop`。
  - **缺口**：`stream_api` 未暴露。
  - **决策**：等框架补 API。
- [ ] **B3** [`prompts/modules.py:10`](prompts/modules.py)：`from src.core.config import get_core_config`（已自带 TODO）。
  - **缺口**：等 `prompt_api.get_bot_personality()`。
- [ ] **B4** [`prompts/modules.py:11`](prompts/modules.py)：`from src.core.prompt import optional, wrap, min_len`（纯工具函数）。
  - **优先级**：低。

### C. 可在不动框架前提下修复的违规

- [x] **C1** [`session.py:323`](session.py) `from src.kernel.storage import JSONStore` → 改为 `from src.app.plugin_system.api.storage_api import JSONStore`。
- [x] **C2** [`parser.py:53-66`](parser.py) `_ensure_call_id` 三层 try/except 兜底 — **滥用 fallback 机制**。
  - 根因：`ToolCall` 是 dataclass(frozen=True)，普通 `setattr` 失败属于已知协议，应明确用 `object.__setattr__` 一次性设置；不应假设"如果都失败"。
  - 现在的 fallback 链是"凑合能跑"的代码气味，违反规范第 5 条。
  - **修复**：参数类型从 `Any` 改为 `ToolCall`，仅保留 `object.__setattr__` 单一路径。
- [ ] **C3** [`chatter.py`](chatter.py)、[`multimodal.py`](multimodal.py)、[`compressor.py`](compressor.py) 等 40+ 处 `getattr(msg, "field", None)` 访问明确类型字段（`Message` / `ChatStream` / `LLMResponse`）。
  - 违反高频违规速查第 5 条。
  - 部分场景确实是处理可能不存在的属性（如 `processed_plain_text`），但大多数是"防御性编程"过度。
  - 需要逐个核实字段是否在类型定义里存在。

### D. 架构与设计层面

- [x] **D1 桥接接口需提升到 services 公共层**。
  - 完成：`context_bridge.py` 提供 `ensure_tool_chain_closed` / `safe_add_payload`。
  - 完成 4.2：`MultimodalService.append_history_reference` 改为纯函数 `build_history_reference_payload`；`TimeoutService.build_timeout_result` 不再接受 `response`，也不再调用 `_close_pending_tool_chain`（调用方 turn_controller 已负责桥接保护）。
- [ ] **D2 `prepare_turn_input` 状态机隐式** — 4 路 if/elif/else（unread / pending_tool / waiting / fallback），通过 `has_pending_tool_results` 和 `is_waiting()` 隐式切换。
  - 风险：tool_result bug 长期不被发现的根本原因之一。
  - 建议：抽显式 `TurnTrigger`（参见 REFACTOR_PLAN.md 第 5 节）。
- [x] **D3 `session.chain_payloads` 持久化用 `dict` 列表** — 已引入 `domain.chain_entry.ChainEntry` 统一 schema：
  - 存储仍为 JSON 友好的 `list[dict]`，但产出与还原都走 `ChainEntry.user/assistant.to_dict()` / `from_dict()`；
  - 脘数据会在 `from_dict` 中被过滤（空 USER、不合法 role、缺 `name` 的 tool_calls），并自动修复存档中 `assistant text="" + tool_calls` 的历史占位；
  - `restore_chain_payloads`、`build_chain_assistant_entry`、`orchestrator` 中的 USER 写入均已访问 `ChainEntry`。
- [x] **D4 `TimeoutService` 与 `MultimodalService` 直接操作 `response.payloads`** — 已修正：两者现为纯函数，仅返回待追加的 payload，调用方统一负责 `add_payload` / `safe_add_payload`。

### E. 测试覆盖

- [ ] **E1 插件无任何单元测试** — 违反代码规范第 5 条。
  - 已补：`test_history_source.py`（A1 回归）4 例、`test_context_bridge.py` 5 例、`test_turn_controller_persistence.py` 4 例，合计 13 例全过。
  - 下一步：覆盖 parser/orchestrator 阶段主流程。

### F. 其他代码气味（低优先级）

- [ ] **F1** 大量运行时局部 `from ...` 导入（`chatter.py`、`orchestrator.py` 等）— 绕过 mypy/IDE 检查。
- [x] **F2** [`reply_json.py`](reply_json.py) — 未被任何 .py 引用，已删除。
- [x] **F3** [`llm_compat.py`](llm_compat.py) — 仅为向 `protocol/` 转发的 shim，未被引用，已删除；README 同步修正。

---

## 二、执行计划（按依赖顺序）

> **规则**：每完成一项，立即更新对应 checkbox 与"进度日志"。

### 阶段 1：链路稳定性（已完成）

- [x] 1.1 修复 A1（restore + 保存端兜底）
- [x] 1.2 修复 A2（抽 `context_bridge` helper）
- [x] 1.3 修复 A3（orchestrator 桥接保护）

### 阶段 2：消除明显违规（不需动框架）

- [x] 2.1 修复 C1：`session.py` 改用 `storage_api.JSONStore`
- [x] 2.2 修复 C2：`parser._ensure_call_id` 改为单一明确路径
- [x] 2.3 决定 F2：删除 `reply_json.py`
- [x] 2.4 决定 F3：删除 `llm_compat.py`（已被 `protocol/compat_adapter.py` 取代）

### 阶段 3：补单元测试（保护回归）

- [x] 3.1 决定测试目录位置（主仓 `test/plugins/kokoro_flow_chatter/` vs 插件内 `tests/`）→ 选择主仓，与 foxzone、default_chatter 一致
- [x] 3.2 写 `test_history_source.py`：覆盖 A1 回归、空 text + tool_calls 的情况、连续 tool_calls
- [x] 3.3 写 `test_context_bridge.py`：覆盖 `ensure_tool_chain_closed` / `safe_add_payload` 各种末尾状态
- [x] 3.4 写 `test_turn_controller_persistence.py`：覆盖 `commit_turn_decision` chain 保存的 assistant_text 兜底逻辑（已抽出 `build_chain_assistant_entry` 纯函数）

### 阶段 4：架构清理（可选 / 与 REFACTOR_PLAN.md 协同）

- [x] 4.1 D3：`ChainEntry` schema 化（已完成，见下方进度日志）
- [x] 4.2 D4：services 不再操作 `response.payloads`（已完成）
- [x] 4.3 C3：去除明显防御性 `getattr`（已按文件分批完成）
  - [x] chatter.py（19 处 → 1 处兜底保留）
  - [x] multimodal.py / session.py / prompts/builder.py / context/sources/{initial,scene,history}_source.py / compressor.py / runtime/{turn_controller,orchestrator,interrupt_controller}.py / handlers/proactive_handler.py
  - [x] 保留说明：`protocol/{response_normalizer,decision_parser,compat_adapter}.py`、`debug/log_formatter.py`、`compressor._msg_time`、`chat_stream.partner_name/group_name`、`msg.media`、`scheduler._running`、`services/context_bridge.payloads` 以 `getattr` 防御为附加价值（面向多 vendor / mixed type）不作改动
- [ ] 4.4 D2：抽 `TurnTrigger` 显式状态机

### 阶段 5：去除 getattr 滥用 / 局部导入

- [ ] 5.1 `chatter.py` 内的 getattr 清理
- [ ] 5.2 `multimodal.py` / `compressor.py` getattr 清理
- [ ] 5.3 集中顶部 import，消除所有运行时局部 import

### 阶段 6：API 缺口处理（需要框架协作 / 暂搁置）

- [ ] 6.1 B1 媒体 API：与框架协商 `media_api.skip_vlm_for_stream`
- [ ] 6.2 B2 流 API：与框架协商 `stream_api.start_stream_loop`
- [ ] 6.3 B3 prompt API：等待 `get_bot_personality()`

---

## 三、进度日志

| 日期 | 项 | 操作 | 验证方法 |
|------|----|------|----------|
| 2026-05-05 | A1 | restore_chain_payloads + commit_turn_decision 双端兜底 | `restore_chain_payloads([user, assistant{text=""+tool_calls}, user])` 角色为 `[user, assistant, tool_result, assistant, user]`，`validate_for_send` 通过 |
| 2026-05-05 | A2 | 抽 `services/context_bridge.py`；timeout_service / turn_controller 3 处替换 | `ensure_tool_chain_closed` 单元用例通过；turn_controller / timeout_service 无 lint 错误 |
| 2026-05-05 | A3 | orchestrator `extra_payload` append 前桥接 | 静态检查通过 |
| 2026-05-05 | C1 | `session.py` 改用 `storage_api.JSONStore` | grep 验证无 `from src.kernel.storage` 残留 |
| 2026-05-05 | C2 | `parser._ensure_call_id` 去除三层 fallback，改为 `object.__setattr__` 单一路径，参数类型从 `Any` 收紧为 `ToolCall` | `get_errors` 无报错 |
| 2026-05-05 | F2 | 删除 `reply_json.py`（未被任何 .py 引用） | grep `from \.reply_json\|import reply_json` 无命中 |
| 2026-05-05 | F3 | 删除 `llm_compat.py`，同步更新 README | grep `from \.llm_compat\|import llm_compat` 无命中 |
| 2026-05-05 | 3.2 | 新增 `test_history_source.py`（4 用例） | `pytest test/plugins/kokoro_flow_chatter/test_history_source.py` 4 passed |
| 2026-05-05 | 3.3 | 新增 `test_context_bridge.py`（5 用例） | `pytest test/plugins/kokoro_flow_chatter/test_context_bridge.py` 5 passed |
| 2026-05-05 | 3.4 | 从 `commit_turn_decision` 抽出 `build_chain_assistant_entry` 纯函数 + 4 测例 | `pytest test/plugins/kokoro_flow_chatter/` 13 passed |
| 2026-05-05 | 4.1 | 新增 `domain/chain_entry.py`（`ChainEntry` dataclass + `from_dict` / `to_dict`）；`history_source` / `turn_controller` / `orchestrator` 三处产出消费均接入 ChainEntry；补 `test_chain_entry.py` 9 例 | `pytest test/plugins/kokoro_flow_chatter/` 22 passed |
| 2026-05-05 | 4.2 | `MultimodalService.append_history_reference` → `build_history_reference_payload`；`TimeoutService.build_timeout_result` 去除 `response` 参数与 `_close_pending_tool_chain`；turn_controller 调用点同步更新 | 22 passed，`get_errors` 无报错 |
| 2026-05-05 | 4.3-1 | `chatter.py` getattr 清理：`format_message_line` 全量直接属性访问；`_extract_*` 系列借助 `Message.sender_id/message_id` 与 `ChatStream.bot_id/context.history_messages` 直接访问；`_extract_timestamp` 参数类型 `Any → Message`；保留第 146 行 `_is_reply_tool` `except Exception` 内 1 处兜底（面向第三方 LLMUsable 实现） | 22 passed |
| 2026-05-05 | 4.3-2 | 余下文件一批清理；session / prompts/builder / initial_source / scene_source / history_source / compressor / turn_controller / orchestrator / interrupt_controller / proactive_handler / multimodal 均改为直接属性访问。`KFCSession`、`Message`、`ChatStream`、`StreamContext`、`LLMResponse` 都是 dataclass / typed 定义，防御 getattr 为净负担。`protocol/{response_normalizer,decision_parser,compat_adapter}` 、`debug/log_formatter`、`compressor._msg_time`、`scheduler._running`、`services/context_bridge.safe_add_payload`、`chat_stream.partner_name/group_name`、`msg.media` 作为多 vendor / mixed type 边界保留 getattr | 22 passed，`get_errors` plugin 目录无报错 |

---

## 四、不要做的事

- ❌ **不要修改框架代码** (`src/kernel/`、`src/core/`、`src/app/`)，B 类问题先等 API 缺口被框架团队补齐。
- ❌ **不要"顺手"重构** REFACTOR_PLAN.md 描述的大型架构（决策 A/B/C），那是另一轮工作。
- ❌ **不要为了消除 getattr 而过度引入 TYPE_CHECKING 导入**，先确认字段确实存在再说。
- ❌ **不要在没有单元测试覆盖的情况下做大规模重构**（阶段 4 必须在阶段 3 之后）。
