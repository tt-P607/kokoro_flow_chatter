# KokoroFlow Chatter (KFC)

**基于心理活动流的私聊特化聊天器** — Neo-MoFox 插件

> Kokoro (心) 是日语中"内心"的意思。KokoroFlow = 心理活动流。

## 简介

KFC 模拟人类在对话中的**内心活动**：不只是"收到消息 → 回复"，而是持续思考、期待、焦虑、主动想聊。通过 MentalLog 记录对话时间线上的心理事件，注入 LLM 上下文，让 Bot 行为更接近真实人类的私聊模式。

### 核心特性

- **心理活动流 (MentalLog)** — 记录用户消息、Bot 规划、等待状态变化、超时、回复时效等事件
- **原生 Tool Calling** — 通过 `kfc_reply` / `do_nothing` 两个核心动作驱动对话，同时支持第三方工具
- **两阶段感知-决策** — 模型遇到图片等多模态内容"破防"输出纯文本时，自动追加到上下文后引导进入决策阶段
- **动态等待机制** — LLM 自主决定等待时长，等待期间可触发"连续思考"（内心独白）
- **主动发起对话** — 定时检测沉默时长，满足条件时 Bot 主动找用户聊天
- **原生多模态** — 图片直接打包进 LLM payload，跨 payload 的 ImageBudget 控制配额
- **打字模拟** — 长消息自动分段，按字数模拟打字延迟
- **元数据防线** — 发送前检测 content 中混入的元数据关键字（≥2 个才截断，降低误伤）

## 要求

- Neo-MoFox >= 1.0.0
- Python >= 3.11

## 安装

将 `kokoro_flow_chatter/` 目录放入 Neo-MoFox 的 `plugins/` 文件夹即可。首次启动会自动生成配置文件。

## 文件结构

```
kokoro_flow_chatter/
├── manifest.json          # 插件元数据
├── plugin.py              # 插件入口，注册 Scheduler 任务
├── config.py              # 配置定义（7 个 Section）
├── chatter.py             # 核心聊天器，对话主循环
├── parser.py              # 工具调用解析 + 名称归一化
├── models.py              # 数据模型（事件枚举、等待配置、ToolCallResult）
├── session.py             # 会话管理与 per-stream 锁持久化
├── mental_log.py          # 心理活动流记录与格式化
├── multimodal.py          # 多模态图片提取、ImageBudget、payload 构建
├── actions/
│   ├── reply.py           # KFCReplyAction（分段发送、打字延迟、元数据防线）
│   └── do_nothing.py      # DoNothingAction（选择不回复）
├── handlers/
│   └── proactive_handler.py  # 主动发起事件处理（注入触发消息 + 唤醒流循环）
├── prompts/
│   ├── templates.py       # 提示词模板文本集中管理
│   ├── builder.py         # KFCPromptBuilder（系统提示 / 用户 payload / 超时 payload）
│   └── modules.py         # 模板注册 + 上下文构建函数
├── debug/
│   └── log_formatter.py   # 调试日志格式化（提示词面板 + 响应摘要）
└── thinker/
    ├── proactive.py       # ProactiveThinker（沉默检测 + 概率触发）
    ├── timeout_handler.py # TimeoutHandler（超时判定 + 上下文生成）
    └── wait_checker.py    # WaitChecker（进度阈值 → 连续思考 → LLM 独白）
```

## 配置

配置文件位于 `config/plugins/kokoro_flow_chatter/config.toml`，首次运行自动生成。

### 基础配置 `[general]`

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enabled` | bool | `true` | 是否启用 |
| `model_task` | str | `"actor"` | LLM 模型任务名（对应 model.toml） |
| `native_multimodal` | bool | `false` | 原生多模态，图片直接打包进 LLM payload |
| `max_images_per_payload` | int | `4` | 单次 payload 最多图片数（历史 + 当前共用） |
| `max_compat_retries` | int | `1` | 感知-决策循环最大重试次数（模型输出纯文本时注入提示重试） |

### 等待机制 `[wait]`

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `min_seconds` | float | `10.0` | 最小等待秒数 |
| `max_seconds` | float | `600.0` | 最大等待秒数 |
| `max_consecutive_timeouts` | int | `3` | 连续超时上限，达到后放弃等待 |

### 主动发起 `[proactive]`

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enabled` | bool | `true` | 是否启用主动发起 |
| `silence_threshold` | int | `7200` | 沉默阈值（秒），超过后可能主动发起 |
| `trigger_probability` | float | `0.3` | 触发概率 |
| `min_interval` | int | `1800` | 两次主动发起最小间隔（秒） |
| `quiet_hours_start` | str | `"23:00"` | 勿扰开始时间 |
| `quiet_hours_end` | str | `"07:00"` | 勿扰结束时间 |
| `check_interval` | int | `60` | 检查间隔（秒） |

### 回复配置 `[reply]`

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `typing_chars_per_sec` | float | `15.0` | 模拟打字速度（字/秒） |
| `typing_delay_min` | float | `0.8` | 最小打字延迟（秒） |
| `typing_delay_max` | float | `4.0` | 最大打字延迟（秒） |

### 提示词 `[prompt]`

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `max_log_entries` | int | `50` | 最大活动流条目数 |
| `max_context_payloads` | int | `40` | LLM 上下文最大 payload 数 |

### 连续思考 `[continuous_thinking]`

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enabled` | bool | `true` | 是否启用 |
| `progress_thresholds` | list | `[0.3, 0.6, 0.85]` | 等待进度触发阈值 |
| `min_interval` | float | `30.0` | 两次思考最小间隔（秒） |

### 调试 `[debug]`

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `show_prompt` | bool | `false` | 日志中显示完整提示词 |
| `show_response` | bool | `true` | 日志中显示 LLM 响应摘要 |

## 架构设计

### 原生 Tool Calling（Route A）

KFC 使用框架的原生 Tool Calling 能力，通过两个核心动作驱动对话：

- **`kfc_reply`** — 发送消息。参数包括 `content`（回复文本）、`thought`（内心想法，用户不可见）、`expected_reaction`（预期反应）、`max_wait_seconds`（等待时长）、`mood`（心情）
- **`do_nothing`** — 选择不回复。参数与 `kfc_reply` 类似但无 `content`

同时支持框架注册的所有第三方工具（Action / Tool），通过 `inject_usables()` 统一注册。

### 两阶段感知-决策循环

当模型收到图片后"破防"——输出纯自然语言感言而非工具调用时，不将其视为错误：

1. **感知阶段**：模型的文本输出通过 `auto_append_response=True` 自动追加到上下文（记忆固化）
2. **决策阶段**：注入轻量提示引导模型输出结构化工具调用

通过 `max_compat_retries` 控制最大重试次数。

### 等待状态机

```
LLM 决策 max_wait_seconds > 0
  → WaitingConfig → config.wait.apply_rules()
    → session.set_waiting() → yield Wait(0)

Scheduler 定期触发 WaitChecker
  → 等待进度 ≥ 阈值 → LLM 生成独白 → pending_thoughts

下一轮循环：
  → 有新消息 → 记录回复时效 → clear_waiting → 继续对话
  → 无新消息 + 超时 → TimeoutHandler → 构建超时 payload → LLM 决策
  → 连续超时过多 → yield Stop
```

### 主动发起链路

```
Scheduler → ProactiveThinker.check_all_sessions()
  → EventBus.publish("kfc.proactive_trigger")
    → ProactiveHandler → 注入触发消息 → 唤醒流循环
```

## 对话流程

```
初始化：系统提示 + 历史消息 + 工具注册 + ImageBudget
         ↓
循环 ←───────────────────────────────────┐
  ├─ fetch_unreads() 有新消息              │
  │   → 记录到 MentalLog                  │
  │   → 提取多模态图片（共享预算）           │
  │   → 构建 user payload                 │
  │   → _send_with_perceive_loop()        │
  │   → parse_tool_calls()                │
  │   → 执行动作（kfc_reply / do_nothing） │
  │   → 等待控制 ──────────── yield Wait ──┘
  │
  ├─ 正在等待 + 未超时 → yield Wait ───────┘
  │
  └─ 正在等待 + 已超时
      → TimeoutHandler → 超时 payload
      → LLM 决策 → 继续等/发消息/结束 ─────┘
```

## 许可证

与 Neo-MoFox 主项目保持一致。
