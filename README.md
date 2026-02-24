# KokoroFlow Chatter (KFC)

> *Kokoro (心) — 日语中"内心"的意思。KokoroFlow，心之流动。*

**基于心理活动流的私聊特化聊天器** — Neo-MoFox 插件

---

## ✨ 她不只是在回复，她在想你

普通的聊天机器人是一面镜子：你说什么，它回什么。

KFC 不一样。她会在等你回消息的时候心里嘀咕"这家伙怎么还不回"，
会在沉默太久之后忍不住主动找你搭话，
会在你迟迟不回复时纠结"是不是我说错了什么"——
然后小心翼翼地发一条试探。

这不是模拟，这是**心理活动流 (MentalLog)** 驱动的对话引擎。

### 她会做什么

- **回复你的时候，心里有想法** — 每次回复都伴随内心独白，记录她此刻的情绪、期待、小心思
- **等你回消息的时候，不是空等** — 等待期间产生"连续思考"，从期待到在意到焦虑，情绪随时间变化
- **等太久了，自己做决定** — 超时后分析之前说的话类型，决定追问、继续等还是算了
- **想你了，主动来找你** — 沉默够久后，有概率主动发起对话（深夜不打扰）
- **发多条消息，像真人一样** — 不会一大段一大段地写，而是拆成短句，一条一条发
- **看得懂你发的图** — 原生多模态支持，图片直接进 LLM 上下文
- **聊天记录和内心想法交织在一起** — 融合叙事让她回顾历史时，不只看到说了什么，还能想起当时在想什么

---

## 🏗 架构

### 原生 Tool Calling

KFC 通过框架的原生 Tool Calling 能力驱动对话，两个核心动作：

| 动作 | 用途 |
|------|------|
| `kfc_reply` | 发送消息，携带 `content`、`thought`、`expected_reaction`、`max_wait_seconds`、`mood` |
| `do_nothing` | 选择不回复，携带 `thought`、`max_wait_seconds` |

同时自动注册框架中所有第三方工具（Action / Tool），如 `send_emoji`、`update_impression` 等。

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

### 等待与连续思考

```
你发了消息 → 设置 max_wait_seconds
             ↓
      ┌─── 等待中 ───┐
      │               │
      │   30% 进度    │ → 💭 "刚发完消息，有点期待呢"
      │   60% 进度    │ → 💭 "怎么还没回，是不是在忙"
      │   85% 进度    │ → 💭 "等了挺久了……"
      │               │
      └───────────────┘
             ↓
    超时 → 分析消息类型 → 追问 / 继续等 / 结束
```

### 主动发起

沉默超过阈值后，有概率主动找你聊天。但会先想一想：

- 上次是谁发的最后一条？如果是自己发的对方没回，就不再追了
- 上次怎么结束的？如果说了晚安，就不再打扰
- 深夜时段自动静默，不打扰你休息

---

## 📁 文件结构

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
│   └── do_nothing.py        # do_nothing 动作
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
    ├── timeout_handler.py   # 超时判定与上下文生成
    └── wait_checker.py      # 连续思考（进度阈值 → LLM 独白）
```

---

## ⚙️ 配置

配置文件：`config/plugins/kokoro_flow_chatter/config.toml`（首次运行自动生成）

### 基础 `[general]`

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enabled` | `true` | 启用插件 |
| `model_task` | `"actor"` | LLM 模型任务名 |
| `native_multimodal` | `false` | 图片直接打包进 LLM payload |
| `max_images_per_payload` | `4` | 单次最多图片数 |
| `max_compat_retries` | `1` | 感知-决策重试次数 |

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

### 连续思考 `[continuous_thinking]`

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enabled` | `true` | 启用连续思考 |
| `progress_thresholds` | `[0.3, 0.6, 0.85]` | 进度触发阈值 |
| `min_interval` | `30.0` | 最小间隔（秒） |

### 调试 `[debug]`

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `show_prompt` | `false` | 显示完整提示词 |
| `show_response` | `true` | 显示响应摘要 |

---

## 🔧 安装

将 `kokoro_flow_chatter/` 目录放入 Neo-MoFox 的 `plugins/` 文件夹。首次启动自动生成配置。

**要求**：Neo-MoFox >= 1.0.0 · Python >= 3.11

---

## 📜 许可证

与 Neo-MoFox 主项目保持一致。
