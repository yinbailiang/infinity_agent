"""LLM 模块公共 API"""

__version__ = '0.1.0'

from .models.chunks import (
    DoneChunk,
    FinishChunk,
    StreamChunk,
    TextChunk,
    ToolCallCompleteChunk,
    UsageChunk,
    UsageStats,
)
from .models.messages import (
    ContentType,
    FunctionCall,
    ImageUrl,
    Message,
    MessageContent,
    MessageRole,
    Messages,
    MultiModalContent,
    ToolCall,
)
from .models.tools import (
    ParameterProperty,
    ToolDefinition,
    ToolDefinitions,
    ToolFunction,
    ToolParameters,
)

__all__ = [
    '__version__',
    # -- 消息 --
    'Message',
    'MessageRole',
    'Messages',
    'MessageContent',
    'ContentType',
    'MultiModalContent',
    'ImageUrl',
    # -- 工具调用 --
    'ToolCall',
    'FunctionCall',
    # -- 工具定义 --
    'ToolDefinition',
    'ToolDefinitions',
    'ToolFunction',
    'ToolParameters',
    'ParameterProperty',
    # -- 流式 Chunk --
    'StreamChunk',
    'TextChunk',
    'ToolCallCompleteChunk',
    'UsageChunk',
    'FinishChunk',
    'DoneChunk',
    'UsageStats',
]
