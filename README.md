# KokoroFlow Chatter (KFC)

**基于心理活动流的私聊特化聊天器** — Neo-MoFox 插件

> Kokoro (心) 是日语中"内心"的意思。KokoroFlow = 心理活动流。

## 简介

KFC 模拟人类在对话中的**内心活动**：不只是"收到消息 → 回复"，而是持续思考、期待、焦虑、主动想聊。通过 MentalLog 记录对话时间线上的心理事件，注入 LLM 上下文，让 Bot 行为更接近真实人类的私聊模式。

### 核心特性

- **心理活动流 (MentalLog)** — 记录用户消息、Bot 规划、等待状态变化、超时、回复时效等事件
- **动态等待机制** — LLM 自主决定等待时长，等待期间可触发"连续思考"（内心独白）
- **主动发起对话** — 定时检测沉默时长，满足条件时 Bot 主动找用户聊天
- **双模式运行** — Unified（单次调用）和 Split（决策+回复两步）可切换
- **原生多模态** — Unified 模式下可直接将图片打包进 LLM payload
- **JSON 嵌入式工具调用** — 不依赖原生 Tool Calling，第三方工具通过提示词注入
- **打字模拟** — 长消息自动分段，按字数模拟打字延迟
- **三层防泄漏** — JSON 结构隔离 → 正则清洗 → 发送前最后防线

## 要求

- Neo-MoFox >= 1.0.0
- Python >= 3.11
- 额外依赖：`json-repair >= 0.57.1`

## 安装

将 `kokoro_flow_chatter/` 目录放入 Neo-MoFox 的 `plugins/` 文件夹即可。首次启动会自动生成配置文件。

## 文件结构

```
kokoro_flow_chatter/
├── manifest.json        # 插件元数据
├── plugin.py            # 插件入口，注册定时任务
├── config.py            # 配置定义（7 个 Section）
├── chatter.py           # 核心聊天器，对话主循环
├── models.py            # 数据模型（事件枚举、等待配置、策略结果）
├── session.py           # 会话管理与持久化
├── mental_log.py        # 心理活动流记录与格式化
├── multimodal.py        # 多模态图片提取与 payload 构建
├── actions/
│   └── reply.py         # 回复动作（分段发送、打字延迟、防泄漏）
├── prompts/
│   ├── templates.py     # 提示词模板
│   ├── builder.py       # 提示词构建器
│   └── modules.py       # 模板注册
├── strategies/
│   ├── base.py          # 策略协议定义
│   ├── unified.py       # Unified 策略（单次调用）
│   └── split.py         # Split 策略（两步调用）
└── thinker/
    ├── proactive.py     # 主动发起检测
    ├── wait_checker.py  # 连续思考检测
    └── timeout.py       # 超时处理
```

## 配置

配置文件位于 `config/plugins/kokoro_flow_chatter/config.toml`，首次运行自动生成。

### 基础配置 `[general]`

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enabled` | bool | `true` | 是否启用 |
| `mode` | str | `"unified"` | 执行模式：`unified`（单次调用）/ `split`（规划+回复） |
| `model_task` | str | `"actor"` | LLM 模型任务名（对应 model.toml） |
| `native_multimodal` | bool | `false` | 原生多模态（仅 unified 模式） |
| `max_images_per_payload` | int | `4` | 单次最多图片数 |

### 等待机制 `[wait]`

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `multiplier` | float | `1.0` | 等待时长倍率 |
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
| `min_affinity` | float | `0.3` | 最低好感度阈值 |
| `check_interval` | int | `60` | 检查间隔（秒） |

### 回复配置 `[reply]`

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `typing_chars_per_sec` | float | `15.0` | 模拟打字速度（字/秒） |
| `typing_delay_min` | float | `0.8` | 最小打字延迟（秒） |
| `typing_delay_max` | float | `4.0` | 最大打字延迟（秒） |
| `max_segment_length` | int | `200` | 分段长度上限 |
| `enable_typo` | bool | `false` | 是否启用错字生成 |

### 提示词 `[prompt]`

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `log_format` | str | `"narrative"` | 活动流格式：`narrative`（叙事）/ `table`（表格） |
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

## 两种模式

### Unified 模式（默认）

单次 LLM 调用，模型以 JSON 一次性返回思考、动作、情绪等全部信息。

- ✅ 延迟低（一次调用）
- ✅ 支持原生多模态
- ✅ 支持第三方工具调用
- ⚠️ 要求模型有较强的 JSON 输出能力

### Split 模式

两步调用：决策步用轻量模型（`sub_actor`）生成 JSON 决策，回复步用主模型（`actor`）生成自然语言。

- ✅ 决策与表达分离，各用最合适的模型
- ✅ JSON 解析失败时更宽容（将文本当回复）
- ⚠️ 延迟较高（两次调用）
- ⚠️ 不支持原生多模态和第三方工具

## 对话流程

```
用户消息 → 构建上下文（历史 + 心理活动流 + 工具描述）
         → LLM 调用（返回 JSON 决策）
         → 解析 StrategyResult
         → 执行动作（回复 / 等待 / 什么都不做）
         → 等待期间可触发连续思考
         → 超时或新消息到达 → 回到循环
```

## 许可证

与 Neo-MoFox 主项目保持一致。
