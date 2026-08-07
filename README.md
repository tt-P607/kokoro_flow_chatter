# KokoroFlow Chatter (KFC)

*Kokoro (心) — 日语中"内心"的意思。*

**基于心理活动流的私聊特化聊天器** — Neo-MoFox 插件

---

## 概述

KFC 是一个面向私聊场景的 Chatter 插件。与传统聊天器不同，KFC 将 LLM 的每次决策与内心独白绑定，形成连续的心理活动流。对话历史与内心活动按时间线交织，让模型在回复时不仅能看到"说了什么"，还能"回想起"当时在想什么。

### 核心能力

- **心理活动流**：每次回复附带内心独白（情绪、期待），形成可回顾的心理时间线
- **近期记忆压缩**：自动将近期对话压缩为第一人称叙事摘要，长期对话不丢失上下文
- **完整上下文快照**：每次成功发送 LLM 请求前把完整 payload 链（含工具调用与结果）持久化到会话 JSON，重启后恢复，消除重启导致的上下文连续性损失
- **私人备忘录**：LLM 可自主记录带过期时间的待办/提醒，自动过期清理
- **等待与超时**：回复后进入等待状态，超时后智能决定追问、继续或结束
- **主动发起**：沉默超过阈值后有概率主动发起对话，支持深夜静默和模型预约
- **消息积累窗口**：连发消息在窗口内合并后统一处理，避免碎片化响应
- **生成打断**：LLM 生成期间检测到新消息时取消当前请求，合并新消息重新处理
- **原生多模态**：图片直接进入 LLM 上下文，无需额外处理
- **回复节奏**：回复拆分为短句，模拟打字节奏逐条发送
- **第三方上下文注入**：通过 `on_prompt_build` 事件接收其他插件的上下文贡献

---

## 架构

### 决策流程

KFC 通过原生 Tool Calling 驱动对话，所有行为通过工具调用完成：

```
收到消息 → 构建上下文 → LLM 决策 → 工具调用
                                        │
                    ┌───────────────────┼───────────────────┐
                    ↓                   ↓                   ↓
               kfc_reply           do_nothing         schedule_proactive
               (发送消息)          (选择沉默)          (预约主动思考)
                    │                   │
                    ↓                   ↓
              设置等待状态          设置等待状态
                    │                   │
                    ↓                   ↓
              等待用户回复 ←── 超时 → 主动续话
```

### 核心动作

| 动作 | 用途 |
|------|------|
| `kfc_reply` | 发送消息，携带内心独白、情绪、预期反应、等待时长 |
| `do_nothing` | 选择不回复，设置等待时长 |
| `schedule_proactive` | 预约下一次主动思考时间 |
| `kfc_memo` | 写入或刷新一条带过期时间的私人备忘录 |
| `kfc_memo_delete` | 删除指定的备忘录 |

同时自动注册框架中所有第三方工具（Action / Tool）。

### 上下文系统

每轮 LLM 请求的上下文由多个来源组合而成：

| 来源 | 内容 |
|------|------|
| 系统提示词 | 人设、行为规范、场景状态 |
| 近期记忆摘要 | 自动压缩的对话叙事（第一人称） |
| 对话链 | 最近的 USER/ASSISTANT 对话记录 |
| 融合叙事 | 聊天记录与内心独白按时间线交织 |
| 心理活动流 | 最近的内心事件（等待、超时、打断等） |
| 私人备忘录 | 当前有效的备忘条目 |
| 第三方注入 | 其他插件通过 `on_prompt_build` 提供的上下文 |

### 近期记忆压缩

对话轮数达到阈值后，自动将近期对话压缩为叙事摘要：

- 使用独立的压缩模型（`compress_model_task`），不影响主对话
- 以第一人称书写，注入后续每轮上下文
- 摘要生成后立即生效，无需重启

### 私人备忘录

LLM 可自主管理的中短期提醒系统：

- 写入：通过 `kfc_memo` 工具，LLM 自行判断时机
- 过期：LLM 设定过期时长（1 小时 ~ 14 天），到期自动清理
- 上限：单流最多 10 条
- 渲染：注入到用户提示词末尾，不进持久化对话链

### 主动发起

沉默超过阈值后，有概率主动发起对话。会先检查：

- 上次是否自己发的最后一条（对方没回就不再追）
- 上次是否以"晚安"等结束语收尾
- 当前是否在深夜静默时段

模型也可通过 `schedule_proactive` 预约未来的主动思考时间。

---

## 文件结构

分层依赖严格单向：`runtime → execution → protocol → domain`，
`context` / `services` / `prompts` 作为被调用的能力层，`domain` 与
`models` 不依赖任何上层。

```
kokoro_flow_chatter/
├── manifest.json              # 插件元数据
├── plugin.py                  # 插件入口：组件注册、调度任务、会话存储
├── config.py                  # 配置定义
├── chatter.py                 # 组件门面：实现框架契约，向 runtime 暴露能力
├── models.py                  # 共享数据模型：常量、事件类型、备忘、等待状态
├── mental_log.py              # 心理活动流容器
├── session.py                 # 会话状态与持久化存储
├── compressor.py              # 近期记忆压缩
├── multimodal.py              # 原生多模态图片处理
├── framework_compat.py        # 框架未公开能力的兼容边界
│
├── domain/                    # 领域模型（纯数据，无 IO）
│   ├── decision.py            # 决策对象
│   ├── chain_entry.py         # 对话链条目
│   └── turn_trigger.py        # 回合触发分类
│
├── protocol/                  # 协议层（纯函数，无副作用）
│   ├── response_normalizer.py # 响应标准化（含 provider 兼容）
│   ├── tool_call_adapter.py   # 工具名/参数归一化的唯一来源
│   └── decision_parser.py     # 执行结果 → Decision 收敛
│
├── execution/                 # 执行层（唯一产生对外副作用）
│   ├── runner.py              # 单轮执行入口：解析 → 执行 → 收敛
│   └── decision_executor.py   # 控制动作解释与第三方工具调度
│
├── runtime/                   # 运行时（对话主循环及配套）
│   ├── orchestrator.py        # 主循环编排
│   ├── turn_controller.py     # 回合输入准备与决策提交
│   ├── context_builder.py     # 初始请求构建
│   ├── model_setup.py         # 模型集解析
│   ├── payload_hygiene.py     # 上下文链清理（孤立结果 / 残留提醒）
│   ├── summary_sync.py        # 记忆摘要热更新
│   ├── input_status.py        # 「正在输入」状态上报
│   ├── request_view.py        # 一次性发送视图
│   ├── interrupt_controller.py# 可打断的 LLM 调用
│   ├── phase_machine.py       # 上下文链角色相位
│   └── unread_policy.py       # 未读消息优先级策略
│
├── context/                   # 上下文层（规划 + 渲染）
│   ├── planner.py             # 上下文规划（产出纯数据）
│   ├── renderer.py            # payload 组装
│   ├── types.py               # 上下文类型定义
│   └── sources/               # 各上下文来源
│       ├── history_source.py  # 历史 / 摘要 / 融合叙事
│       ├── initial_source.py  # 启动时的系统模板变量
│       ├── memo_source.py     # 备忘录
│       └── plugin_source.py   # 第三方注入
│
├── services/                  # 运行时服务（带状态副作用）
│   ├── proactive_service.py   # 主动发起：预约管理 + 触发判定
│   ├── timeout_service.py     # 等待超时：判定 + 状态推进
│   └── summary_service.py     # 摘要压缩任务调度与去重
│
├── actions/                   # KFC 专属动作
│   ├── reply.py               # kfc_reply（含元数据 schema 工具）
│   ├── do_nothing.py          # do_nothing
│   ├── pass_and_wait.py       # pass_and_wait
│   ├── memo.py                # kfc_memo / kfc_memo_delete
│   └── schedule_proactive.py  # schedule_proactive
│
├── prompts/                   # 提示词
│   ├── templates.py           # 静态模板文本
│   └── modules.py             # 模板注册与动态提示词构建
│
├── handlers/                  # 事件处理
│   ├── proactive_handler.py   # 主动发起事件
│   └── voice_call_history_handler.py  # 通话历史回填
│
├── debug/                     # 调试工具
│   └── log_formatter.py       # 提示词面板与决策摘要
│
└── test/                      # 测试
    ├── test_kfc_refactor_protocol.py    # 核心协议
    └── test_kfc_lifecycle_and_config.py # 生命周期与配置
```

---

## 配置

配置文件：`config/plugins/kokoro_flow_chatter/config.toml`（首次运行自动生成）

### `[general]` 基础

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enabled` | `true` | 启用插件 |
| `model_task` | `"actor"` | LLM 模型任务名 |
| `models` | `[]` | 指定模型列表（优先级高于 model_task） |
| `temperature` | `0.7` | 温度参数；仅在 `models` 非空时生效，范围 0~2 |
| `max_tokens` | `8000` | 最大输出 token；仅在 `models` 非空时生效 |
| `native_multimodal` | `false` | 图片直接进 LLM 上下文 |
| `max_images_per_payload` | `4` | 单次最多图片数 |
| `blocked_tools` | `[]` | 屏蔽的工具列表 |
| `max_follow_up_retries` | `3` | 工具失败最大续轮次数 |

### `[wait]` 等待

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enabled` | `true` | 启用等待机制 |
| `min_seconds` | `10.0` | 最小等待秒数 |
| `max_seconds` | `600.0` | 最大等待秒数 |
| `max_consecutive_timeouts` | `3` | 连续超时上限 |

### `[proactive]` 主动发起

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enabled` | `true` | 启用主动发起 |
| `silence_threshold` | `7200` | 沉默阈值（秒） |
| `trigger_probability` | `0.3` | 触发概率 |
| `min_interval` | `1800` | 最小间隔（秒） |
| `quiet_hours_start` | `"23:00"` | 勿扰开始 |
| `quiet_hours_end` | `"07:00"` | 勿扰结束 |

### `[reply]` 回复

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `typing_chars_per_sec` | `15.0` | 打字速度（字/秒） |
| `typing_delay_min` | `0.8` | 最小延迟（秒） |
| `typing_delay_max` | `4.0` | 最大延迟（秒） |

### `[prompt]` 提示词

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `max_log_entries` | `50` | 活动流最大条目数 |
| `max_context_payloads` | `20` | 上下文最大 payload 数 |
| `compress_every_n_rounds` | `50` | 每 N 轮触发记忆压缩 |
| `compress_days_window` | `3.0` | 压缩覆盖天数 |
| `min_compress_interval_minutes` | `120.0` | 压缩最小间隔（分钟） |
| `compress_min_chars` | `800` | 摘要最小字数 |
| `compress_max_chars` | `1200` | 摘要最大字数 |
| `compress_model_task` | `"actor"` | 压缩使用的模型任务（独立于主对话） |

### `[buffer]` 打断

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `interrupt_enabled` | `true` | 启用生成打断 |
| `interrupt_poll_seconds` | `0.5` | 打断检测间隔（秒） |
| `interrupt_cooldown` | `3.0` | 打断后冷却窗口基准值（秒），连续打断递增 1/2 |
| `max_consecutive_interrupts` | `3` | 连续打断次数上限 |

### `[debug]` 调试

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `show_prompt` | `false` | 显示完整提示词 |
| `show_response` | `true` | 显示响应摘要 |

### `[snapshot]` 完整上下文快照

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enabled` | `true` | 启用完整上下文快照。每次成功发送 LLM 请求前把主链 payload 持久化到会话 JSON（`data/kokoro_flow_chatter/sessions/<stream_id>.json`）；重启后首次 `execute()` 启动时恢复，保证重启前后上下文一致 |

---

## 组件签名

- `kokoro_flow_chatter:chatter:kokoro_flow_chatter`
- `kokoro_flow_chatter:action:kfc_reply`
- `kokoro_flow_chatter:action:do_nothing`
- `kokoro_flow_chatter:action:pass_and_wait`
- `kokoro_flow_chatter:action:schedule_proactive`
- `kokoro_flow_chatter:action:kfc_memo`
- `kokoro_flow_chatter:action:kfc_memo_delete`
- `kokoro_flow_chatter:event_handler:kfc_proactive_handler`
- `kokoro_flow_chatter:event_handler:kfc_voice_call_history_handler`

## 生命周期与数据

- 插件加载后注册提示词模板，并通过 TaskManager 等待统一 Scheduler 启动。
- 主动发起检查使用名为 `kfc_proactive_check` 的周期调度；插件卸载时会移除。
- 近期摘要按聊天流去重调度；同一流不会并发启动多份压缩任务，卸载时会取消残留任务。
- 会话状态保存在 `data/kokoro_flow_chatter/sessions/`，按 `stream_id` 隔离；同目录下的 `_index.json` 维护 `stream_id` 与账号的可读映射。
- **单一 JSON 承载全部会话数据**：`mental_log`（日记）、`history_summary`（近期记忆摘要）、`memos`（备忘录）、`context_snapshot`（完整上下文快照）等所有跨轮状态都在同一个 `<stream_id>.json` 文件里，不设额外存储目录。
- 禁用 `[general].enabled` 后不注册 Chatter，由框架为私聊流选择其他可用 Chatter。

## 自动测试

在仓库根目录执行：

```bash
uv run ruff check plugins/kokoro_flow_chatter
uv run pytest plugins/kokoro_flow_chatter/test -q
```

测试覆盖对话相位、上下文规划、工具执行、未读策略、配置边界、manifest 组件一致性、Scheduler 卸载清理、摘要任务去重和框架兼容边界。

## 故障排查

### 插件启用但没有接管私聊

1. 检查 `[general].enabled` 是否为 `true`。
2. 检查 manifest 中是否启用了 `kokoro_flow_chatter` Chatter。
3. 确认目标流是私聊；KFC 不接管群聊。
4. 查看日志中是否出现“模型配置错误”或 Chatter 绑定恢复信息。

### 主动发起没有触发

1. 检查 `[proactive].enabled`、`silence_threshold`、`trigger_probability` 和 `min_interval`。
2. 沉默触发受勿扰时间限制；模型预约不受勿扰时间限制。
3. 查看 Scheduler 是否已启动，以及 `kfc_proactive_check` 是否注册成功。
4. 冷启动流需要框架 `StreamLoopManager.start_stream_loop()` 能力；失败会记录明确警告。

### 原生多模态仍出现 VLM 描述

1. 检查 `[general].native_multimodal` 是否开启，主模型是否支持图片输入。
2. 新用户第一条消息可能早于 KFC 注册流级识别跳过；后续消息会使用跳过设置。
3. 流级识别跳过当前由框架 `MediaManager` 提供，但尚未暴露为插件公共 API；KFC 通过 `framework_compat.py` 集中隔离该内部边界。
4. 表情包按设计仍走 VLM 文字描述，以复用其哈希缓存，属预期行为。

### 近期摘要不生成

1. 首次有效对话会尝试生成空摘要；后续按 `compress_every_n_rounds` 触发。
2. 检查 `compress_model_task` 是否存在可用模型。
3. 检查 `min_compress_interval_minutes` 是否阻止了短时间重复压缩。
4. 同一流已有压缩任务运行时，新请求会被去重而不是重复启动。

## 人工验证清单

1. 私聊发送文本和图片，确认 KFC 接管且多模态配置符合预期。
2. 让模型分段回复，确认消息顺序、打字延迟和引用回复正常。
3. 在模型生成期间连续发送新消息，确认旧请求被打断且消息合并处理。
4. 测试 `do_nothing`、`pass_and_wait`、等待超时和最大连续超时。
5. 创建、覆盖、取消主动预约，并验证热流和冷启动流触发。
6. 达到摘要条件后确认摘要写入会话文件，重启后仍可恢复。
7. 发布 `voice_call.ended` 事件，确认通话内容以一对 chain entry 回填。
8. 禁用、重载、卸载插件，确认无残留 Scheduler 或摘要后台任务。

---

## 安装

将 `kokoro_flow_chatter/` 放入 Neo-MoFox 的 `plugins/` 目录，首次启动自动生成配置文件。

**要求**：Neo-MoFox >= 1.0.0 · Python >= 3.11

---

## 许可证

与 Neo-MoFox 主项目保持一致。
