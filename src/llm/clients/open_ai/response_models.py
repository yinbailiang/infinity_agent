"""OpenAI 兼容 API 的响应模型定义"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatCompletionTokenLogprob(BaseModel):
    """单个 token 的对数概率信息"""

    token: str = Field(description='Token 字符串')
    bytes: Optional[List[int]] = Field(default=None, description='Token 的字节表示')
    logprob: float = Field(description='Token 的对数概率')
    top_logprobs: Optional[List['ChatCompletionTokenLogprob']] = Field(
        default=None, description='最可能的替代 token 列表'
    )


class LogprobsContent(BaseModel):
    """消息内容 token 的对数概率列表"""

    content: Optional[List[ChatCompletionTokenLogprob]] = Field(
        default=None, description='消息内容 token 的对数概率信息'
    )
    refusal: Optional[List[ChatCompletionTokenLogprob]] = Field(
        default=None, description='消息拒绝 token 的对数概率信息'
    )


class ToolCallFunction(BaseModel):
    """工具调用中的函数信息（流式增量）"""

    name: Optional[str] = Field(default=None, description='函数名称')
    arguments: str = Field(default='', description='JSON 参数片段（流式拼接）')


class ToolCallDelta(BaseModel):
    """单条工具调用的流式增量片段"""

    index: int = Field(default=0, description='工具调用在列表中的索引位置')
    id: Optional[str] = Field(default=None, description='工具调用唯一标识符')
    type: Literal['function'] = Field(default='function', description='调用类型')
    function: Optional[ToolCallFunction] = Field(default=None, description='函数名称与参数增量')


class Delta(BaseModel):
    """流式返回的 completion 增量"""

    content: Optional[str] = Field(default=None, description='增量的文本内容')
    role: Optional[Literal['assistant']] = Field(default=None, description='消息角色')
    function_call: Optional[Dict[Any, Any]] = Field(default=None, description='函数调用信息（已弃用）')
    refusal: Optional[str] = Field(default=None, description='拒绝消息内容')
    tool_calls: Optional[List[ToolCallDelta]] = Field(default=None, description='工具调用增量列表')


class CompletionTokensDetails(BaseModel):
    """completion 中 token 的细分统计"""

    accepted_prediction_tokens: Optional[int] = Field(
        default=None,
        description='使用预测输出时，预测中出现在 completion 中的 token 数量',
    )
    audio_tokens: Optional[int] = Field(default=None, description='模型生成的音频 token 数量')
    reasoning_tokens: Optional[int] = Field(default=None, description='模型用于推理的 token 数量')
    rejected_prediction_tokens: Optional[int] = Field(
        default=None,
        description='使用预测输出时，预测中未出现在 completion 中的 token 数量',
    )


class PromptTokensDetails(BaseModel):
    """prompt 中 token 的细分统计"""

    audio_tokens: Optional[int] = Field(default=None, description='prompt 中的音频 token 数量')
    cached_tokens: Optional[int] = Field(default=None, description='prompt 中的缓存 token 数量')


class CompletionUsage(BaseModel):
    """Token 使用统计"""

    completion_tokens: int = Field(description='生成的 completion 的 token 数量')
    prompt_tokens: int = Field(description='prompt 的 token 数量')
    total_tokens: int = Field(description='总计 token 数量')
    completion_tokens_details: Optional[CompletionTokensDetails] = Field(
        default=None, description='completion token 的细分信息'
    )
    prompt_tokens_details: Optional[PromptTokensDetails] = Field(default=None, description='prompt token 的细分信息')


class Choice(BaseModel):
    """completion 选择项"""

    delta: Delta = Field(description='流式增量内容')
    finish_reason: Optional[Literal['stop', 'length', 'content_filter', 'tool_calls', 'function_call']] = Field(
        default=None, description='模型停止生成 token 的原因'
    )
    index: int = Field(description='选择项的索引')
    logprobs: Optional[LogprobsContent] = Field(default=None, description='该 choice 的对数概率信息')


class StreamEvent(BaseModel):
    """OpenAI 兼容流式响应的单个数据块"""

    id: str = Field(description='对话的唯一标识符，每个 chunk 相同')
    choices: List[Choice] = Field(
        description='completion 选择列表，若 n>1 可包含多个元素；若设置了 include_usage，最后 chunk 可能为空'
    )
    created: int = Field(description='创建时间戳（秒）')
    model: str = Field(description='模型名称')
    object: Literal['chat.completion.chunk'] = Field(
        default='chat.completion.chunk',
        description='对象类型，固定为 chat.completion.chunk',
    )
    service_tier: Optional[Literal['auto', 'default', 'flex', 'scale', 'priority']] = Field(
        default=None, description='实际处理请求的服务层级'
    )
    system_fingerprint: Optional[str] = Field(default=None, description='后端配置指纹，可用于确定性调试')
    usage: Optional[CompletionUsage] = Field(
        default=None,
        description='仅在 stream_options.include_usage 为 true 时存在，通常最后 chunk 包含完整统计',
    )
