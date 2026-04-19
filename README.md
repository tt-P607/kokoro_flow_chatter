# KokoroFlow Chatter (KFC)

*Kokoro (心) — 日语中"内心"的意思。*

**基于心理活动流的私聊特化聊天器** — Neo-MoFox 插件

---

## 概述

KFC 是一个面向私聊场景的 Chatter 插件，核心设计是将 LLM 的每次决策与内心独白（MentalLog）绑定，形成连续的心理活动流。对话历史与内心活动按时间线交织，让模型在回复时不仅能看到说了什么，还能"回想起"当时在想什么。

**主要能力**

- 每次回复附带内心独白，记录当前情绪与期待
- 等待超时后分析消息类型，决定追问、继续等或结束
- 沉默超过阈值后有概率主动发起对话，深夜自动静默
- 多条连发消息在积累窗口内合并后统一处理
- LLM 生成期间若检测到新消息，取消当前请求并重新处理
- 原生多模态支持，图片直接进 LLM 上下文
- 回复拆分为短句模拟打字节奏逐条发送

---

## 架构

### 原生 Tool Calling

KFC 通过框架的原生 Tool Calling 能力驱动对话，两个核心动作：

| 动作 | 用途 |
|------|------|
| `kfc_reply` | 发送消息，携带 `content`、`thought`、`expected_reaction`、`max_wait_seconds`、`mood` |
| `do_nothing` | 选择不回复，携带 `thought`、`max_wait_seconds` |
| `schedule_proactive` | 预约下一次主动思考时间，携带 `delay_seconds`、`reason` |

同时自动注册框架中所有第三方工具（Action / Tool），如 `send_emoji`、`update_impression` 等。

### 消息积累窗口

检测到第一条新消息后，KFC 等待一个固定窗口（默认 1.5 秒）以收集连发的多条消息，再统一提交给 LLM。最大积累时长默认为 5 秒，防止无限延迟：

```
用户消息 1 → 启动积累窗口 (1.5s)
用户消息 2 → 窗口刷新
用户消息 3 → 窗口结束 → 合并三条消息 → LLM
```

### LLM 生成打断

LLM 生成期间，每 0.5 秒检测一次是否有新消息到达。检测到打断时：

```
LLM 正在生成 ──────────────────→
               ↑ 新消息到达
               │ 取消当前请求
               ↓
         记录打断事件到 MentalLog
               ↓
         合并新消息重新提交 LLM
```

打断事件会写入 MentalLog，LLM 在后续回复中可以感知到"被打断"这件事。

### 融合叙事

聊天记录与内心独白按时间交织，形成统一时间线：

```
[21:30:18] 你回复：刚才那张明明还没发出去嘛♪
[22:17:10] 你回复：突然戳我一下，是想提醒我该去梦里找你了吗♪
[22:18:06] 言柒说：[图片]
[22:18:22] 你回复：怎么又在发牛奶呀♪
[22:23:07] 你回复：总是这样戳我，是在怪我吗♪
[22:23:09] （你的内心：柒柒又戳我了，看来这瓶牛奶并没有让他安静下来……）
```

LLM 在回顾历史时不仅看到"说了什么"，还能想起"当时在想什么"。

### 主动发起

沉默超过阈值后，有概率主动找你聊天。但会先想一想：

- 上次是谁发的最后一条？如果是自己发的对方没回，就不再追了
- 上次怎么结束的？如果说了晚安，就不再打扰
- 深夜时段自动静默，不打扰你休息

**模型预约**：模型可以调用 `schedule_proactive` 工具预约下一次主动思考时间。预约存在时，沉默条件检测暂停，只在预约时间到达时触发，赋予模型真正的主动性——而不只是被动地等条件触发。

---

## 文件结构

```
kokoro_flow_chatter/
├── manifest.json            # 插件元数据
├── plugin.py                # 插件入口，注册调度任务
├── config.py                # 配置定义（7 个 Section）
├── chatter.py               # 核心对话循环
├── parser.py                # Tool Calling 解析 + 打字延迟
├── models.py                # 数据模型（事件枚举、等待配置）
├── session.py               # 会话状态持久化
├── mental_log.py            # 心理活动流记录与格式化
├── multimodal.py            # 图片提取与 ImageBudget
├── actions/
│   ├── reply.py             # kfc_reply 动作
│   ├── do_nothing.py        # do_nothing 动作
│   └── schedule_proactive.py # schedule_proactive 动作（预约主动思考）
├── handlers/
│   └── proactive_handler.py # 主动发起事件处理
├── prompts/
│   ├── templates.py         # 提示词模板集中管理
│   ├── builder.py           # 提示词构建（系统 / 用户 / 超时）
│   └── modules.py           # 模板注册与上下文构建
├── debug/
│   └── log_formatter.py     # 调试日志美化输出
└── thinker/
    ├── proactive.py         # 沉默检测与概率触发
    └── timeout_handler.py   # 超时判定与上下文生成
```

---

## 配置

配置文件：`config/plugins/kokoro_flow_chatter/config.toml`（首次运行自动生成）

### 基础 `[general]`

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enabled` | `true` | 启用插件 |
| `model_task` | `"actor"` | LLM 模型任务名 |
| `use_tool_calling` | `true` | 回复模式：`true` = 工具调用（新模型），`false` = JSON 解析（旧模型） |
| `native_multimodal` | `false` | 图片直接打包进 LLM payload |
| `max_images_per_payload` | `4` | 单次最多图片数 |
| `max_compat_retries` | `1` | 感知-决策重试次数（JSON 解析模式） |

### 等待 `[wait]`

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `min_seconds` | `10.0` | 最小等待秒数 |
| `max_seconds` | `600.0` | 最大等待秒数 |
| `max_consecutive_timeouts` | `3` | 超时上限，达到后停止等待 |

### 主动发起 `[proactive]`

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enabled` | `true` | 启用主动发起 |
| `silence_threshold` | `7200` | 沉默阈值（秒） |
| `trigger_probability` | `0.3` | 触发概率 |
| `min_interval` | `1800` | 最小间隔（秒） |
| `quiet_hours_start` | `"23:00"` | 勿扰开始 |
| `quiet_hours_end` | `"07:00"` | 勿扰结束 |
| `check_interval` | `60` | 检查间隔（秒） |

### 回复 `[reply]`

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `typing_chars_per_sec` | `15.0` | 打字速度（字/秒） |
| `typing_delay_min` | `0.8` | 最小延迟（秒） |
| `typing_delay_max` | `4.0` | 最大延迟（秒） |

### 提示词 `[prompt]`

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `max_log_entries` | `50` | 活动流最大条目数 |
| `max_context_payloads` | `20` | LLM 上下文最大 payload 数 |
| `warmup_rounds` | `3` | 热启动历史轮次数 |

### 消息缓冲与打断 `[buffer]`

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `accumulate_window` | `1.5` | 消息积累窗口（秒） |
| `accumulate_max_window` | `5.0` | 最大积累时长（秒），防止无限延迟 |
| `interrupt_enabled` | `true` | 启用 LLM 生成打断 |
| `interrupt_poll_seconds` | `0.5` | 打断检测轮询间隔（秒） |

### 调试 `[debug]`

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `show_prompt` | `false` | 显示完整提示词 |
| `show_response` | `true` | 显示响应摘要 |

---

## 安装

将 `kokoro_flow_chatter/` 目录放入 Neo-MoFox 的 `plugins/` 文件夹。首次启动自动生成配置。

**要求**：Neo-MoFox >= 1.0.0 · Python >= 3.11

---

## 许可证

与 Neo-MoFox 主项目保持一致。
