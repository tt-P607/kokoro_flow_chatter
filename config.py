"""KokoroFlowChatter 配置定义。

定义插件所有可配置参数，基于 Pydantic + TOML 热重载。
通过 @config_section 划分为语义清晰的 Section。
"""

from __future__ import annotations

from typing import ClassVar

from src.app.plugin_system.base import BaseConfig, Field, SectionBase, config_section


class KFCConfig(BaseConfig):
    """KokoroFlowChatter 配置。"""

    name: ClassVar[str] = "config"
    description: ClassVar[str] = "KokoroFlowChatter 配置"

    @config_section("general")
    class GeneralSection(SectionBase):
        """基础配置。"""

        enabled: bool = Field(default=True, description="是否启用")
        model_task: str = Field(
            default="actor",
            description="LLM 模型名称（对应 model.toml 中的 task），models 为空时使用",
        )
        models: list[str] = Field(
            default_factory=list,
            description="指定 LLM 模型列表（对应 model.toml 中的 name）。非空时覆盖 model_task，多个模型按顺序 fallback",
        )
        temperature: float = Field(
            default=0.7,
            description="模型温度，仅在 models 非空时生效",
            ge=0.0,
            le=2.0,
        )
        max_tokens: int = Field(
            default=8000,
            description="最大输出 token 数，仅在 models 非空时生效",
            ge=1,
            le=128000,
        )
        native_multimodal: bool = Field(
            default=False,
            description=(
                "原生多模态模式。启用后，图片直接打包进 LLM payload，"
                "由主模型在对话上下文中理解图片内容并做出响应。"
                "需确保 model_task 配置的模型支持多模态输入。"
            ),
        )
        custom_decision_prompt: str = Field(
            default="",
            description=(
                "自定义决策提示词。用于指导 KFC 的决策行为，"
                "会被注入到系统提示词的安全准则之后。留空则不生效。"
            ),
        )
        blocked_tools: list[str] = Field(
            default_factory=lambda: [
                "send_text",
                "pass_and_wait",
                "stop_conversation",
            ],
            description=(
                "需要从工具列表中屏蔽的工具末段名称（不含组件类型前缀）。"
                "列表中的工具不会暴露给 LLM。"
            ),
        )
        max_follow_up_retries: int = Field(
            default=3,
            description=(
                "工具调用失败后 FOLLOW_UP 续轮的最大次数。"
                "当 LLM 因工具参数格式错误等原因持续失败时，"
                "超过此次数后将强制停止续轮并进入等待，防止无限重试。"
                "设为 0 表示不限制续轮次数（不推荐）。"
            ),
            ge=0,
            le=20,
        )
        enable_input_status: bool = Field(
            default=False,
            description=(
                "是否在 LLM 生成期间向 QQ 客户端上报「正在输入」状态。"
                "启用后，每次 LLM 请求前发送 set_input_status，"
                "请求结束后撤下。仅对 SnowLuma 适配器生效。"
            ),
        )
        segment_instruction: str = Field(
            default=(
                "## 消息分段发送\n"
                "你可以把回复拆成多条消息分开发送，模仿真人边想边打字的节奏，想到什么就发什么。\n"
                "将每条独立消息作为数组中的一个元素传入 content，系统会自动依次发出。\n\n"
                "**分段建议**：\n"
                "- 随意分段，不必凑完整句子，话说到一半想到新的可以直接断开；\n"
                "- 语气词、口语转折词、感叹词出现时是天然的分段点；\n"
                "- 每段尽量短，几个字到十几字最自然；\n"
                "- 同一个意思可以拆开几条说，前一条留悬念，后一条接上；\n"
                "- 只有一两个字时可以不分段。"
            ),
            description=(
                "注入到提示词中的自定义分段指令。"
                "留空则不注入任何分段指导。"
            ),
        )
        wait_instruction: str = Field(
            default=(
                "### max_wait_seconds（等待时长）\n\n"
                "这个参数描述的是你发完消息后是否在等回复。\n\n"
                "期待对方很快回应——填一个短时间（比如你问了个问题、聊得正起劲想继续）。\n"
                "话题告一段落、说了告别、对方不需要特别回什么——填 0。\n\n"
                "用短等待来维持当前聊天的节奏；如果是想过一段时间再主动找对方，"
                "那是主动思考工具的用途，不是这里。\n\n"
                "**上限**：系统会将超过 {max_wait_seconds} 秒的值自动截断为该上限，"
                "因此无需设置超过此值的等待时长。"
            ),
            description=(
                "注入到提示词中的 max_wait_seconds 等待时长指导说明。"
                "留空则不注入。"
            ),
        )

    @config_section("wait")
    class WaitSection(SectionBase):
        """等待机制配置。"""

        enabled: bool = Field(
            default=True,
            description="是否启用回复等待。设为 false 后模型不再等待用户回复",
        )
        min_seconds: float = Field(
            default=10.0,
            description="最小等待秒数",
            ge=0.0,
            le=86400.0,
        )
        max_seconds: float = Field(
            default=600.0,
            description="最大等待秒数",
            ge=0.0,
            le=86400.0,
        )
        max_consecutive_timeouts: int = Field(
            default=3,
            description="连续超时上限，达到后不再等待",
            ge=0,
            le=100,
        )

        def apply_rules(self, raw_seconds: float, consecutive_timeouts: int) -> float:
            """应用等待时长规则。raw_seconds <= 0 或 enabled=false 时返回 0。"""
            if not self.enabled or raw_seconds <= 0:
                return 0.0
            if consecutive_timeouts >= self.max_consecutive_timeouts:
                return 0.0
            lower = min(self.min_seconds, self.max_seconds)
            upper = max(self.min_seconds, self.max_seconds)
            return max(lower, min(raw_seconds, upper))

    @config_section("proactive")
    class ProactiveSection(SectionBase):
        """主动发起配置。"""

        enabled: bool = Field(default=True, description="是否启用主动发起")
        silence_threshold: int = Field(
            default=7200,
            description="沉默阈值(秒)，超过后可能主动发起",
            ge=0,
            le=2592000,
        )
        trigger_probability: float = Field(
            default=0.3,
            description="主动发起触发概率",
            ge=0.0,
            le=1.0,
        )
        min_interval: int = Field(
            default=1800,
            description="两次主动发起最小间隔(秒)",
            ge=0,
            le=2592000,
        )
        quiet_hours_start: str = Field(
            default="23:00",
            description="勿扰开始时间",
            pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$",
        )
        quiet_hours_end: str = Field(
            default="07:00",
            description="勿扰结束时间",
            pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$",
        )
        check_interval: int = Field(
            default=60,
            description="主动发起检查间隔(秒)",
            ge=1,
            le=86400,
        )
        schedule_guidance: str = Field(
            default=(
                "预约是你和 Ta 保持长线联系的主要方式，不是偶尔才想到的工具。\n\n"
                "每次对话之后，自然地想一想「下次什么时候联系 Ta？」，有想法就设一下，"
                "让联系成为持续的节律，而不是单次事件。大部分时候保持有一个预约在等待中就好，不必强迫每次都设。\n\n"
                "**发出消息之后：**\n"
                "根据这次对话，考虑一下下次的时间，有感觉就设上。\n\n"
                "**触发时刚好在聊天中：**\n"
                "通常不用额外发，继续聊就好。聊完之后可以再想想下一次。\n\n"
                "**触发时觉得时机不对：**\n"
                "不发也行，可以重新设一个更合适的时间。\n\n"
                "**频率没有固定规律：**\n"
                "跟着你们当下的状态走，热络时短一些，需要空间时长一些。\n\n"
                "系统的沉默触发只是兜底，预约才是主力。"
            ),
            description="action-schedule_proactive 工具的使用场景指导（会展示在工具描述中，可按需自定义）",
        )

    @config_section("reply")
    class ReplySection(SectionBase):
        """回复配置。"""

        typing_chars_per_sec: float = Field(
            default=15.0,
            description="模拟打字速度(字/秒)",
            gt=0.0,
            le=200.0,
        )
        typing_delay_min: float = Field(
            default=0.8,
            description="最小打字延迟(秒)",
            ge=0.0,
            le=60.0,
        )
        typing_delay_max: float = Field(
            default=4.0,
            description="最大打字延迟(秒)",
            ge=0.0,
            le=60.0,
        )

    @config_section("prompt")
    class PromptSection(SectionBase):
        """提示词配置。"""

        max_log_entries: int = Field(
            default=50,
            description="最大活动流条目数",
            ge=1,
            le=10000,
        )
        max_context_payloads: int = Field(
            default=20,
            description="LLM 上下文持久化链最大条目数（超出时裁剪最旧的 USER/ASSISTANT 对）",
            ge=2,
            le=1000,
        )
        compress_every_n_rounds: int = Field(
            default=50,
            description="每完成 N 轮对话触发一次近期记忆压缩（1 轮 = 1 次 USER→ASSISTANT 交换）",
            ge=0,
            le=10000,
        )
        compress_days_window: float = Field(
            default=3.0,
            description="压缩时覆盖的历史时间窗口（天），只对该窗口内的消息做摘要",
            gt=0.0,
            le=365.0,
        )
        min_compress_interval_minutes: float = Field(
            default=120.0,
            description="两次压缩之间的最短间隔（分钟），防止频繁触发",
            ge=0.0,
            le=525600.0,
        )
        compress_min_chars: int = Field(
            default=800,
            description="近期记忆摘要的最小字数（写入压缩指令，引导 LLM 控制摘要长度下限）",
            ge=0,
            le=100000,
        )
        compress_max_chars: int = Field(
            default=1200,
            description="近期记忆摘要的最大字数（写入压缩指令，引导 LLM 控制摘要长度上限）",
            ge=0,
            le=100000,
        )
        compress_model_task: str = Field(
            default="actor",
            description=(
                "近期记忆压缩使用的 LLM 模型任务（对应 model.toml 中的 task）。"
                "独立于主对话模型配置，可选择更高性价比的模型用于摘要生成。"
            ),
        )


    @config_section("buffer")
    class BufferSection(SectionBase):
        """打断配置。"""

        interrupt_enabled: bool = Field(
            default=True,
            description=(
                "是否启用 LLM 生成打断。启用后，LLM 生成期间若检测到"
                "新消息到达，将取消当前 LLM 请求并以全量消息重新发起。"
            ),
        )
        interrupt_poll_seconds: float = Field(
            default=0.5,
            description=(
                "打断检测轮询间隔（秒）。LLM 生成期间每隔此时间检查"
                "一次是否有新消息到达。值越小响应越快，CPU 占用略高。"
            ),
            gt=0.0,
            le=60.0,
        )
        interrupt_cooldown: float = Field(
            default=3.0,
            description=(
                "打断后冷却窗口基准值（秒）。LLM 被打断后等待此时长再重新发起请求，"
                "以收集可能连发的后续消息。连续打断时每次叠加原值的 1/2："
                "第 1 次 3.0s，第 2 次 4.5s，第 3 次 6.0s，以此类推。"
            ),
            ge=0.0,
            le=300.0,
        )
        max_consecutive_interrupts: int = Field(
            default=3,
            description=(
                "连续打断次数上限。达到后不再打断当前 LLM 请求，"
                "让其正常完成后统一处理积累的消息，防止恶意刷消息导致无限 LLM 调用。"
            ),
            ge=0,
            le=100,
        )

    @config_section("debug")
    class DebugSection(SectionBase):
        """调试配置。"""

        show_prompt: bool = Field(
            default=False,
            description="是否在日志中显示发送给 LLM 的完整提示词",
        )
        show_response: bool = Field(
            default=True,
            description="是否在日志中显示 LLM 响应的美化摘要",
        )

    general: GeneralSection = Field(default_factory=GeneralSection)
    wait: WaitSection = Field(default_factory=WaitSection)
    proactive: ProactiveSection = Field(default_factory=ProactiveSection)
    reply: ReplySection = Field(default_factory=ReplySection)
    prompt: PromptSection = Field(default_factory=PromptSection)
    buffer: BufferSection = Field(default_factory=BufferSection)
    debug: DebugSection = Field(default_factory=DebugSection)
