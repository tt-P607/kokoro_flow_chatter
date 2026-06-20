# KokoroFlow Chatter (KFC)

*Kokoro (心) — 日语中"内心"的意思。*

**基于心理活动流的私聊特化聊天器** — Neo-MoFox 插件

---

## 概述

KFC 是一个面向私聊场景的 Chatter 插件。与传统聊天器不同，KFC 将 LLM 的每次决策与内心独白绑定，形成连续的心理活动流。对话历史与内心活动按时间线交织，让模型在回复时不仅能看到"说了什么"，还能"回想起"当时在想什么。

### 核心能力

- **心理活动流**：每次回复附带内心独白（情绪、期待），形成可回顾的心理时间线
- **近期记忆压缩**：自动将近期对话压缩为第一人称叙事摘要，长期对话不丢失上下文
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

```
kokoro_flow_chatter/
├── manifest.json              # 插件元数据
├── plugin.py                  # 插件入口
├── config.py                  # 配置定义
├── chatter.py                 # 聊天器门面
├── compressor.py              # 近期记忆压缩
├── session.py                 # 会话状态持久化
├── mental_log.py              # 心理活动流
├── models.py                  # 共享数据模型
├── multimodal.py              # 多模态图片处理
│
├── domain/                    # 领域模型
│   ├── decision.py            # 决策对象
│   ├── scene_state.py         # 场景状态
│   ├── chain_entry.py         # 对话链条目
│   └── turn_trigger.py        # 回合触发分类
│
├── runtime/                   # 运行时主循环
│   ├── orchestrator.py        # 主循环编排
│   ├── turn_controller.py     # 回合准备与提交
│   ├── phase_machine.py       # 对话阶段状态机
│   ├── request_view.py        # LLM 请求视图
│   ├── interrupt_controller.py # 生成打断控制
│   ├── message_buffer.py      # 消息积累窗口
│   └── unread_policy.py       # 未读消息过滤
│
├── context/                   # 上下文构建
│   ├── planner.py             # 上下文规划
│   ├── renderer.py            # 上下文渲染
│   ├── types.py               # 上下文类型定义
│   └── sources/               # 各上下文来源
│       ├── history_source.py  # 历史/摘要/叙事
│       ├── scene_source.py    # 场景状态
│       ├── plugin_source.py   # 第三方注入
│       ├── memo_source.py     # 备忘录
│       └── initial_source.py  # 初始上下文
│
├── protocol/                  # 协议层
│   ├── compat_adapter.py      # Provider 兼容
│   ├── response_normalizer.py # 响应标准化
│   ├── decision_parser.py     # 决策解析
│   └── tool_call_adapter.py   # 工具调用适配
│
├── actions/                   # KFC 专属动作
│   ├── reply.py               # kfc_reply
│   ├── do_nothing.py          # do_nothing
│   ├── memo.py                # kfc_memo / kfc_memo_delete
│   ├── schedule_proactive.py  # schedule_proactive
│   └── pass_and_wait.py       # pass_and_wait
│
├── prompts/                   # 提示词
│   ├── templates.py           # 模板定义
│   ├── builder.py             # 提示词构建
│   └── modules.py             # 模板注册
│
├── services/                  # 服务层
│   ├── summary_service.py     # 摘要压缩调度
│   ├── timeout_service.py     # 超时处理
│   └── proactive_service.py   # 主动发起调度
│
├── thinker/                   # 思考决策
│   ├── proactive.py           # 沉默检测与触发
│   └── timeout_handler.py     # 超时判定
│
├── handlers/                  # 事件处理
│   ├── proactive_handler.py   # 主动发起事件
│   └── voice_call_history_handler.py
│
├── execution/                 # 执行层
│   └── decision_executor.py   # 决策执行
│
├── debug/                     # 调试工具
│   └── log_formatter.py       # 日志美化
│
└── test/                      # 测试
    └── test_kfc_refactor_protocol.py
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
| `temperature` | `0.9` | 温度参数 |
| `max_tokens` | `4096` | 最大输出 token |
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

### `[buffer]` 消息缓冲

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `accumulate_window` | `1.5` | 消息积累窗口（秒） |
| `accumulate_max_window` | `5.0` | 最大积累时长（秒） |
| `interrupt_enabled` | `true` | 启用生成打断 |
| `interrupt_poll_seconds` | `0.5` | 打断检测间隔（秒） |

### `[debug]` 调试

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `show_prompt` | `false` | 显示完整提示词 |
| `show_response` | `true` | 显示响应摘要 |

---

## 安装

将 `kokoro_flow_chatter/` 放入 Neo-MoFox 的 `plugins/` 目录，首次启动自动生成配置文件。

**要求**：Neo-MoFox >= 1.0.0 · Python >= 3.11

---

## 许可证

与 Neo-MoFox 主项目保持一致。
