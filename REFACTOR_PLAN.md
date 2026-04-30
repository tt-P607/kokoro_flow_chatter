# KFC 重构计划

## 目的

这份文档用于把 KokoroFlowChatter 的重构任务交接给下一个 agent。

目标不是继续在现有结构上打补丁，而是把 KFC 从“单文件臃肿总控 + 多条隐式注入链 + 多套决策协议并存”的状态，重构为一个边界清晰、可维护、可继续扩展的私聊关系驱动聊天器。

本文档包含：

- 当前实现的关键问题
- 推荐的目标架构
- 关键状态与模块拆分建议
- 分阶段迁移步骤
- 风险点与验收标准
- 给下一个 agent 的具体执行建议

---

## 当前问题总结

### 1. 总控文件职责过重

当前 `chatter.py` 同时负责：

- 模型选择与 provider 兼容前处理
- 初始上下文构建
- 历史消息装配
- 消息积累窗口
- 生成打断
- 感知重试与 compat 补救
- 工具解析前后的桥接
- session 持久化更新
- 等待控制
- 近期摘要触发
- 多模态注入

这使得任何一个问题，例如：

- 场景脑补错误
- provider 响应格式漂移
- tool chain 闭合问题
- 多模态上下文污染

都会变成全文件追线，而不是局部修复。

### 2. 上下文来源过多，且没有统一模型

KFC 当前上下文主要来自：

- 系统 prompt
- actor reminder
- 融合叙事文本
- chain payloads
- history_summary
- on_prompt_build 的额外注入
- timeout / proactive 的特殊 prompt
- 多模态图片内容

这些来源在实现中是分散追加的，而不是先形成结构化上下文模型再统一渲染。

后果：

- 很难判断某段行为到底来自哪个上下文源
- 很难控制优先级和冲突关系
- 很容易出现模型在证据不足时自行补足场景

### 3. 缺少显式场景状态，导致模型自由脑补

KFC 有关系感、等待感、生活连续性、主动联系、近期记忆，但没有明确的 `SceneState`。

因此模型会把：

- 私聊语境
- 连续关系
- 夜晚 / 沉默 / 等待
- 生活化叙事

自动组合成默认私人场景，例如：

- 在家里
- 在房间里
- 在床上 / 沙发上
- 拿着手机聊天

这不是单条提示词问题，而是世界建模方式的问题。

### 4. 决策协议过多

当前同时存在：

- 原生 tool calling
- 旧 JSON 回复模式
- 正文里的 compat JSON tool_calls
- 纯文本感知后补一轮决策

这意味着 KFC 实际在维护多套“模型输出 -> 插件内部动作”的转换协议。

这类复杂度不应该散落在主流程中。

### 5. provider 兼容逻辑侵入业务层

`llm_compat.py` 现在处理：

- DeepSeek thinking 关闭
- reasoning_content 回退
- compat JSON 解析
- last assistant payload 同步
- 未发送草稿重写

这说明业务层和 provider 适配层边界没有切清楚。

### 6. 事件注入链不统一

KFC 目前有几种不同注入方式：

- `kfc_system_prompt` 的模板构建事件
- `kfc_user_prompt` 的手工 on_prompt_build 事件
- actor reminder
- 临时 extra user payload
- timeout / proactive 的单独 user payload

这些都是“上下文进入模型”的入口，但没有一个总线去描述它们的顺序、作用域、生命周期和冲突规则。

更严重的是，部分第三方插件为了让信息“跨轮保留”，只能把内容注入到 `USER` payload。

这样虽然在技术上能进入上下文，但在语义上会产生严重混淆：

- 模型会把这段内容理解为“用户当前说的话”
- 模型会把这段内容误判为“用户状态”而不是“bot 自身状态”
- 第三方插件实际获得了绕过上下文分层、直接污染主提示词的能力

这不是文案问题，而是缺少“上下文归属模型”和“第三方贡献协议”。

---

## 重构目标

### 核心目标

把 KFC 重构为：

- 一个以私聊关系状态为中心的聊天器
- 一个以结构化上下文为核心的 prompt 系统
- 一个只维护单一内部决策协议的执行系统
- 一个能显式建模场景状态、避免默认脑补的架构

### 明确架构决策

以下三条不再作为开放讨论项，而是本次重构的固定方向。

#### 决策 A：多模态降级为可选外围能力

保留多模态能力，但不再把它视为 KFC 的核心路径。

重构后应满足：

- 多模态默认作为可选服务存在，而不是主流程骨架的一部分
- 文本私聊主链在关闭多模态时必须完整独立运行
- 多模态只作为“输入证据增强”进入上下文，不负责塑造默认生活场景
- 不再为了多模态能力把历史图片、预算控制、预加载逻辑深嵌进主总控

#### 决策 B：删除旧 JSON 回复兼容

旧的 JSON 回复模式不再保留。

重构后应满足：

- `use_tool_calling = false` 这条正式模式被移除
- `reply_json.py` 及其主流程依赖不再作为正式兼容路径保留
- KFC 唯一正式决策入口为：`tool calling -> normalize -> Decision`
- 允许保留 provider 侧的 tool-call 兼容适配，例如正文里携带 `tool_calls` JSON 的场景，但这不属于旧 JSON 回复模式

#### 决策 C：第三方扩展必须走“工具贡献 / 上下文贡献 / 状态提交”三分模型

第三方扩展不再允许直接把“要长期生效的信息”伪装成普通 `USER` 文本注入主链。

重构后应满足：

- 第三方工具注册与第三方上下文注入必须分离
- 第三方上下文必须以结构化 `ContextContribution` 提交，而不是裸文本拼接
- 需要跨轮保留的信息必须通过 `StatePatch` / session store 持久化，而不是依赖历史 `USER` 文本残留
- 即使底层最终仍以 `USER payload` 承载部分上下文，那也只能是传输实现细节，不能代表语义归属
- KFC 必须能区分 `self_state`、`user_state`、`relationship_state`、`scene_evidence`、`notice`
- 第三方插件默认无权直接写高优先级 system/policy 区域

### 非目标

本轮不要追求：

- 一次性改完所有提示词文案
- 一次性换掉全部功能
- 一次性移除所有 provider 兼容逻辑

重点应该是先把架构骨架搭对，再逐步迁移行为。

---

## 推荐目标架构

建议采用“模块化单体”而不是分散式架构。

### 顶层分层

建议拆成以下几层：

1. `runtime/`
负责对话生命周期编排，不包含复杂业务逻辑。

2. `domain/`
负责会话聚合、关系状态、等待状态、场景状态、决策对象。

3. `context/`
负责把各种上下文源先变成结构化块，再渲染给模型。

4. `protocol/`
负责把模型输出统一规范化为内部 `Decision`。

5. `execution/`
负责执行动作、生成 ToolResult、更新状态。

6. `services/`
负责主动发起、超时、可选多模态、摘要压缩等外围能力。

7. `extensions/`
负责第三方工具提供、上下文贡献、状态补丁接入协议。

### 推荐目录结构

建议目标结构如下：

```text
kokoro_flow_chatter-main/
├── plugin.py
├── config.py
├── manifest.json
├── README.md
├── REFACTOR_PLAN.md
├── runtime/
│   ├── orchestrator.py
│   ├── turn_controller.py
│   ├── interrupt_controller.py
│   └── message_buffer.py
├── domain/
│   ├── session.py
│   ├── scene_state.py
│   ├── relationship_state.py
│   ├── waiting_state.py
│   ├── decision.py
│   ├── events.py
│   └── mental_log.py
├── context/
│   ├── planner.py
│   ├── renderer.py
│   ├── ownership.py
│   ├── sources/
│   │   ├── system_source.py
│   │   ├── reminder_source.py
│   │   ├── history_source.py
│   │   ├── summary_source.py
│   │   ├── scene_source.py
│   │   ├── plugin_source.py
│   │   ├── proactive_source.py
│   │   └── timeout_source.py
│   └── schemas.py
├── protocol/
│   ├── response_normalizer.py
│   ├── tool_call_adapter.py
│   ├── compat_adapter.py
│   └── decision_parser.py
├── execution/
│   ├── decision_executor.py
│   ├── action_dispatcher.py
│   ├── result_committer.py
│   └── typing_simulator.py
├── services/
│   ├── proactive_service.py
│   ├── timeout_service.py
│   ├── multimodal_service.py
│   └── summary_service.py
├── extensions/
│   ├── tool_provider.py
│   ├── context_provider.py
│   ├── state_patch.py
│   └── registry.py
├── prompts/
│   ├── templates.py
│   ├── registry.py
│   └── policies.py
├── actions/
│   ├── reply.py
│   ├── do_nothing.py
│   └── schedule_proactive.py
└── handlers/
    └── proactive_handler.py
```

说明：

- 不要求一步到位迁移到这个目录
- 但迁移方向应当以此为准

---

## 核心重构思路

### 1. 把 `chatter.py` 缩成真正的 orchestrator

未来的 `chatter.py` 或 `runtime/orchestrator.py` 只负责：

- 激活 stream
- 读取 session
- 拉取未读
- 选择触发原因
- 调用上下文规划器
- 调用模型
- 调用决策规范化
- 调用执行器
- 提交结果

它不应该再负责：

- 自己拼 prompt 文本
- 自己解析 provider 差异
- 自己决定怎样写回每一种 session 字段

### 2. 引入显式的 `Decision` 作为唯一内部协议

当前模型输出路径太多，但重构后只保留一个正式入口。

建议引入统一的内部决策对象，例如：

```python
@dataclass
class Decision:
    thought: str
    mood: str
    expected_reaction: str
    visible_reply_segments: list[str]
    wait_seconds: float
    should_reply: bool
    should_wait: bool
    should_end_turn: bool
    third_party_calls: list[ToolCallSpec]
    proactive_schedule: ProactiveSchedule | None
```

所有输入都必须先归一化到这个对象：

- 原生 tool calling
- provider 兼容型 tool_calls 文本
- 纯文本感知后的补发

主流程后面一律只认 `Decision`，不再直接依赖 `call_list` 或原始 `message`。

### 3. 显式建模 `SceneState`

这是这次重构里必须新增的关键状态。

建议至少包含：

```python
@dataclass
class SceneState:
    certainty: Literal["unknown", "weak", "confirmed"]
    location_type: str
    social_channel: str
    device_assumption_allowed: bool
    evidence: list[SceneEvidence]
```

设计原则：

- 默认场景必须是 `unknown`
- 没有证据时，不能渲染为具体生活场景
- `platform/chat_type` 只能影响社交礼仪，不得直接推导为物理环境
- “会用手机拍照”是能力，不是当前场景

建议把场景信息分为三类：

1. `channel metadata`
例如平台、chat_type、bot_id

2. `scene evidence`
来自用户消息、历史消息、工具结果、显式设定

3. `scene inference`
只允许基于 evidence 生成，且必须带置信度

### 4. 上下文必须先结构化，再渲染

不要再让 prompt builder 直接到处字符串拼接。

建议先产出：

```python
ContextPlan(
    system_blocks=[...],
    policy_blocks=[...],
    reminder_blocks=[...],
    self_state_blocks=[...],
    user_state_blocks=[...],
    relationship_blocks=[...],
    scene_blocks=[...],
    history_blocks=[...],
    summary_blocks=[...],
    user_blocks=[...],
    transient_blocks=[...],
)
```

然后由 `ContextRenderer` 再决定如何渲染进：

- system payload
- user payload
- transient user payload

这样做的好处：

- 可以明确每一块的来源和作用域
- 可以控制优先级和覆盖关系
- 可以更容易调试“某条描述是从哪来的”

这里必须进一步明确一个原则：

- payload role 不等于语义归属

也就是说，某段文本即便最终通过 `USER payload` 发送给模型，也不代表它在语义上属于“用户当前发言”。

KFC 应显式维护至少这些归属类别：

1. `policy`
协议、硬约束、格式要求；默认只允许 core / KFC 自己写入。

2. `self_state`
bot 自身能力、装备、模式、内部状态。

3. `user_state`
用户明确表达或工具明确验证过的用户信息。

4. `relationship_state`
跨轮关系结论、互动阶段、长期约定。

5. `scene_evidence`
场景证据，只能写观察到或确认到的事实。

6. `notice`
本轮临时提醒，不自动升级为长期状态。

第三方插件如果要提供内容，必须先声明它属于哪个归属区，而不是直接拼接成自由文本。

### 4.1 第三方扩展接入协议

第三方扩展建议强制拆成三条链：

1. `ToolProvider`
只负责注册工具，不直接修改 prompt。

2. `ContextProvider`
只负责提交结构化上下文贡献，例如：

```python
@dataclass
class ContextContribution:
    source: str
    owner: Literal[
        "policy",
        "self_state",
        "user_state",
        "relationship_state",
        "scene_evidence",
        "notice",
    ]
    scope: Literal["turn", "session", "persistent"]
    priority: int
    ttl_turns: int | None
    content: str
    evidence_only: bool = False
```

3. `StatePatch`
只负责申请持久化修改，例如：

```python
@dataclass
class StatePatch:
    source: str
    target: Literal["self_state", "user_state", "relationship_state", "scene_state"]
    op: Literal["set", "merge", "append", "remove"]
    path: str
    value: Any
    reason: str
```

这样做的意义是：

- 工具结果不会直接升级成高优先级 prompt
- 第三方上下文不会再偷偷冒充用户输入
- 持久化行为从“靠历史文本残留”变成“显式状态提交”

### 4.2 工具结果不能直接变成高优先级提示词

建议统一要求：

`tool result -> observation normalize -> optional state patch -> next turn render`

而不是：

`tool result -> 直接拼进 system/user 提示词`

否则第三方工具只要返回一段强指令文本，就等于绕过了 KFC 的上下文和权限模型。

### 5. 区分“持久状态”和“瞬时触发原因”

当前 timeout、proactive、interrupt、new unread 都在主循环里混在一起。

建议统一抽象成：

```python
TurnTrigger = NewUnread | TimeoutExpired | ProactiveWake | FollowupToolResult
```

每种触发原因都走同一条 orchestrator 主链，但由不同的 context source 注入额外上下文。

这样：

- 超时逻辑不会再直接决定 prompt 文案
- 主动发起不会再单独伪造一条特殊消息作为“伪用户输入”
- follow-up 工具续轮和正常用户消息轮次能共享更多逻辑

### 6. 把 provider 兼容隔离到 `protocol/compat_adapter.py`

兼容逻辑应该只做这些事：

- 调整请求参数
- 从 provider 响应中提取标准内容
- 做兼容型 tool_calls 解析
- 修复响应链结构一致性

它不应该决定业务行为。

也就是说：

- `llm_compat.py` 应当只产出“标准响应视图”
- 不应继续夹带 KFC 特定业务状态

### 7. 工具执行与状态提交分离

建议把：

- “执行动作”
- “把执行结果写回 session / mental log / chain / summary 触发器”

拆成两个模块。

当前做法的问题是：

- tool 执行副作用和状态写回顺序高度耦合
- 出错时很难做补偿
- 不利于以后做事件回放和调试审计

对第三方扩展来说，这里还要再补一条：

- 工具执行完成不等于状态自动生效，任何跨轮影响都必须经过状态提交层审核

---

## 当前文件到目标模块的建议映射

### 应保留但缩职责的文件

- `plugin.py`
- `config.py`
- `actions/reply.py`
- `actions/do_nothing.py`
- `actions/schedule_proactive.py`
- `handlers/proactive_handler.py`

### 应拆分的文件

#### `chatter.py`

建议拆成：

- `runtime/orchestrator.py`
- `runtime/message_buffer.py`
- `runtime/interrupt_controller.py`
- `runtime/turn_controller.py`

#### `prompts/builder.py`

建议拆成：

- `context/planner.py`
- `context/renderer.py`
- `context/sources/history_source.py`
- `context/sources/scene_source.py`
- `context/sources/reminder_source.py`
- `context/sources/plugin_source.py`

#### `on_prompt_build` / extra user payload 注入链

建议从“任意插件可直接拼接 raw 文本”改成：

- `extensions/context_provider.py`
- `extensions/state_patch.py`
- `context/sources/plugin_source.py`

兼容期内可以保留旧事件，但只允许它们产出结构化贡献对象，不再允许直接返回“视为用户发言”的裸文本。

#### `parser.py`

建议拆成：

- `protocol/decision_parser.py`
- `execution/action_dispatcher.py`
- `execution/typing_simulator.py`

#### `reply_json.py`

建议删除，不再保留旧 JSON 回复模式。

#### `llm_compat.py`

建议拆成：

- `protocol/compat_adapter.py`
- `protocol/response_normalizer.py`

#### `multimodal.py`

建议迁移为：

- `services/multimodal_service.py`

并明确降级为可选外围能力，不再成为核心主流程的一部分。

#### `compressor.py`

建议迁移为：

- `services/summary_service.py`

#### `thinker/proactive.py`

建议迁移为：

- `services/proactive_service.py`

#### `thinker/timeout_handler.py`

建议迁移为：

- `services/timeout_service.py`

---

## 推荐迁移顺序

不要直接大改所有文件。建议分 6 个阶段推进。

### 阶段 0：冻结当前行为，补回归测试

目的：先给重构留安全网。

必须保证以下行为有测试：

- actor reminder 以标准 reminder 语义进入上下文
- DeepSeek compat JSON 能转为 tool_calls
- 纯文本感知草稿不会污染 assistant 历史
- tool_result 链闭合正确
- 缺失 call_id 时本地补齐
- KFC system prompt 不应把通道参数渲染成具体设备场景
- 第三方插件注入的 bot 自身状态不会被模型误判为 user_state

如果要继续重构，必须先保住这些。

### 阶段 1：删除旧 JSON 回复模式，确立唯一正式协议

这一阶段先做架构收敛，而不是继续双轨并行。

动作：

- 删除 `use_tool_calling = false` 的正式支持路径
- 删除 `reply_json.py` 在主流程中的职责
- 清理 system prompt / parser / config 中围绕旧 JSON 模式的分支
- 明确 KFC 的唯一正式协议为原生 tool calling

成功标志：

- 主流程不再存在旧 JSON 回复模式分支
- 配置和文档不再把旧 JSON 模式视为正式能力

### 阶段 2：引入统一的 `Decision`

先不改主流程，只在现有实现旁边新增：

- `Decision`
- `DecisionParser`
- `ResponseNormalizer`

让旧主流程仍然工作，但逐步把输出统一到 `Decision`。

成功标志：

- 主流程后半段不再直接依赖原始 `call_list` 结构

### 阶段 3：抽出上下文规划器

新增：

- `ContextPlan`
- `ContextPlanner`
- `ContextRenderer`

并在这一阶段同步建立：

- `ContextContribution`
- `StatePatch`
- plugin source 的统一接入点

先把 `build_system_prompt()` 和 `build_user_payload()` 内部逻辑迁出去，`prompts/builder.py` 暂时只做门面层。

成功标志：

- prompt builder 不再负责直接遍历历史消息和 mental log
- 第三方扩展不再直接拼 raw extra user payload

### 阶段 4：引入显式 `SceneState`

这一阶段是解决“默认在家里 / 默认在手机上 / 默认室内场景”的关键。

动作：

- 在 session 或 domain 层新增 `SceneState`
- 定义场景证据来源
- 改写 context planner，让 scene block 只使用 evidence，而不依赖自由脑补

成功标志：

- 没有证据时，渲染出的场景说明是“未知 / 未提供”，而不是任何具体生活环境

### 阶段 5：拆解 `chatter.py`

把当前主流程拆成：

- 拉消息
- 生成 turn trigger
- 上下文规划
- 调模型
- 标准化决策
- 执行动作
- 提交状态

每个环节只做一件事。

成功标志：

- `chatter.py` 从超大总控变成薄 orchestrator

### 阶段 6：整理外围服务与收尾清理

把：

- proactive
- timeout
- multimodal
- summary compress

迁成稳定的服务模块。

其中多模态必须满足：

- 默认不成为文本主链依赖
- 关闭时不影响私聊文本行为
- 开启时只作为附加证据输入，而不是场景主导来源

成功标志：

- 主流程对这些服务只通过清晰接口调用
- 旧 JSON 回复模式及相关残留配置已清理完毕

---

## 下一个 agent 的首批任务建议

建议下一个 agent 不要先改 prompt，而是先做结构化骨架。

推荐执行顺序：

1. 删除旧 JSON 回复模式的主流程入口与配置入口
2. 新建 `Decision` 与 `ResponseNormalizer`
3. 把 `llm_compat.py` 里与“响应结构标准化”相关的逻辑迁过去
4. 新建 `ContextPlan`、`ContextPlanner`、`ContextContribution`、`StatePatch`
5. 把 `prompts/builder.py` 中的 fused narrative 构建逻辑迁到 context 层，并建立 plugin source 接入点
6. 新增 `SceneState`，并让系统 prompt 从“直接渲染场景描述”改为“渲染场景证据说明”

在这个阶段不要做的事：

- 不要先大改人格提示词
- 不要先改主动发起文案
- 不要先扩展多模态能力
- 不要同时改协议和行为策略
- 不要继续新增“为了持久化而塞进 USER 文本”的插件约定

---

## 建议保留的设计

以下设计本身是有价值的，不建议一刀切删掉：

### 1. MentalLog 机制

“内心活动流”是 KFC 的识别性能力，问题不在它存在，而在它与历史、摘要、场景的耦合方式。

### 2. 主动预约能力

`schedule_proactive` 是一个很好的能力设计，因为它让“主动联系”变成了模型显式决策，而不是纯概率随机。

### 3. 消息积累窗口

私聊连续发消息合并处理是合理的，不建议删，只建议把实现挪出主总控。

### 4. 生成期间打断

这是 KFC 非常实用的能力，也应当保留，但要从 `chatter.py` 独立出去。

### 5. 多模态能力本身

多模态能力可以保留，但只应保留为可选外围服务，不应再作为主流程的一等公民。

---

## 建议弱化或重写的设计

### 1. 融合叙事的“生活场景暗示性”

融合叙事应该保留，但不要再让它自然语言地过于强烈地暗示环境连续性。

更好的方式是：

- 让它提供“发生了什么”的时间线
- 不负责推导“现在身处怎样的实体场景”

### 2. prompt 内的软约束过多

现在很多行为依靠 prompt 里的“不要这样做”来限制。

这类事情更适合：

- 用结构化状态限制
- 用 context planner 控制输入
- 用 decision normalizer 过滤输出

### 3. 第三方插件直接注入 raw USER 文本

这类设计必须弱化甚至移除。

因为它把两个完全不同的问题混成了一件事：

- 如何把内容送进本轮模型上下文
- 这段内容在语义上到底属于谁

更好的方式是：

- 第三方上下文只提交结构化贡献
- 跨轮保留走 `StatePatch`
- 由 `ContextPlanner` 统一决定渲染位置

### 4. 删除旧 JSON 模式

这里不再是“长期共存”的讨论项，而是明确删除项。

重构目标应直接删除旧 JSON 回复模式，只保留原生 tool calling 及必要的 provider tool-call 兼容适配。

### 5. 多模态作为默认主路径

多模态不应再和文本私聊主链平级。

更好的方式是：

- 让多模态成为可选服务
- 仅在启用且模型支持时参与上下文装配
- 在设计上保证“没有多模态时，KFC 仍然是完整产品”

---

## 验收标准

重构完成后，至少要满足以下条件：

### 架构层

- `chatter.py` 不再是超大总控文件
- prompt 组装逻辑不再到处直接拼文本
- provider 兼容逻辑不再侵入业务流程
- 工具执行与状态提交分离
- 旧 JSON 回复模式及其配置入口已删除
- 多模态已降级为可选外围服务
- 第三方工具、第三方上下文、第三方状态提交三条链已拆分
- payload role 与语义归属已显式区分

### 行为层

- 无场景证据时，不默认渲染为“在家 / 在房间 / 在床上 / 在沙发上 / 拿着手机”
- actor reminder 的能力描述不自动升级为当前现实场景
- timeout、proactive、follow-up 都走统一 turn 流程
- DeepSeek compat 行为继续正确
- 关闭多模态时，文本主链行为不受影响
- 第三方插件提供的 bot 自身状态不会再被模型误判为用户状态
- 第三方工具结果不会直接变成高优先级提示词

### 可维护性层

- 新增一个上下文源时，不需要修改 4 个以上核心文件
- 新增一个 provider 兼容分支时，不需要修改业务层主循环
- 能通过日志快速定位每一段上下文来自哪个 source
- 能追踪每条第三方贡献的 owner、scope、priority、ttl

---

## 建议补充的测试

建议新增或补齐以下测试：

1. `SceneState` 在无证据时渲染为 unknown
2. `ContextPlanner` 能区分 reminder、history、scene、summary 的来源
3. `ResponseNormalizer` 能统一处理原生 tool call 与 provider 兼容型 tool_calls
4. `DecisionExecutor` 执行 reply / do_nothing / third-party calls 的顺序正确
5. timeout 触发时不会伪造额外生活场景
6. proactive 唤醒时只带触发原因，不隐式带室内环境暗示
7. 多模态关闭时，文本私聊主链仍可完整运行
8. 第三方 `ContextContribution(owner=self_state)` 不会被渲染成 user_state
9. 第三方工具结果只有经过 normalize 和 state patch 后才能跨轮生效
10. 旧 `on_prompt_build` 兼容层不会再直接产出 raw extra user payload

---

## 最后判断

KFC 现在的问题不是“功能太多”，而是“同一份功能被塞进了错误的边界”。

它本来应该是：

- 私聊关系状态驱动
- 多来源上下文规划
- 单一决策协议
- 明确场景状态建模

但当前实现更像：

- 一个不断长大的总控文件
- 多条字符串注入链叠加
- 多套协议并存
- 缺少显式 SceneState 的隐式世界模拟

因此重构方向必须是：

先整理边界，再整理 prompt，最后再整理行为细节。

如果顺序反过来，只会继续在旧结构上打补丁。