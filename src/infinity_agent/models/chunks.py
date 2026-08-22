"""流式响应通用模型定义"""

from typing import List, Optional

from pydantic import BaseModel, Field

from .messages import ToolCall


class UsageStats(BaseModel):
    """LLM Token 使用统计"""

    prompt_tokens: int = Field(ge=0, description='输入令牌数')
    completion_tokens: int = Field(ge=0, description='输出令牌数')
    total_tokens: int = Field(ge=0, description='总令牌数')
    reasoning_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        description='思考（推理）令牌数，未提供时为 None',
    )


class StreamChunk(BaseModel):
    """流式响应单元基类"""

    type: str


class TextChunk(StreamChunk):
    """文本增量片段"""

    type: str = Field(default='text')
    text: str = Field(description='文本内容')


class ThinkingChunk(StreamChunk):
    """思考（推理）增量片段，对应 DeepSeek 等模型的 reasoning_content"""

    type: str = Field(default='thinking')
    text: str = Field(description='思考内容增量')


class ToolCallCompleteChunk(StreamChunk):
    """聚合完成的工具调用"""

    type: str = Field(default='toolcall_complete')
    tool_calls: List[ToolCall] = Field(default_factory=lambda: [], description='聚合完成的工具调用列表')


class UsageChunk(StreamChunk):
    """Token 使用统计"""

    type: str = Field(default='usage')
    usage: UsageStats = Field(description='使用统计')


class FinishChunk(StreamChunk):
    """流结束理由"""

    type: str = Field(default='finish')
    finish_reason: str = Field(description='结束理由')


class DoneChunk(StreamChunk):
    """流完全结束标记"""

    type: str = Field(default='done')
