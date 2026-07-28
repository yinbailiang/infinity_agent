"""LLM 客户端抽象基类"""

import types
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

from infinity_agent.clients.config import RequestConfig

from ..models import (
    Message,
    Messages,
    StreamChunk,
    TextChunk,
    ToolCall,
    ToolCallCompleteChunk,
)


class LLMClient(ABC):
    """LLM 客户端抽象基类"""

    @abstractmethod
    async def __aenter__(self) -> 'LLMClient':
        """支持 async with 资源管理"""
        ...

    @abstractmethod
    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[types.TracebackType],
    ) -> Optional[bool]: ...

    # ---------- 核心对话接口 ----------

    @abstractmethod
    def stream_chat(
        self,
        messages: Messages,
        config: Optional[RequestConfig] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        流式对话接口

        :param messages: 消息列表
        :param config: 请求配置
        :yield: StreamChunk 子类实例，调用方通过 isinstance 分支处理
        :raise: LLMError
        """
        ...

    # ---------- 工具方法 ----------

    @staticmethod
    def chunks_to_message(chunks: list[StreamChunk]) -> Message:
        """
        将流式 chunks 聚合为一条完整的 assistant Message

        遍历所有 chunk，拼接文本内容并收集工具调用，
        最终产出一条可直接追加到对话历史的 assistant 消息。

        :param chunks: 完整的流式 chunk 列表
        :return: role=assistant 的 Message
        """
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for chunk in chunks:
            if isinstance(chunk, TextChunk):
                text_parts.append(chunk.text)
            elif isinstance(chunk, ToolCallCompleteChunk):
                tool_calls.extend(chunk.tool_calls)

        content = ''.join(text_parts) if text_parts else None
        return Message.assistant(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
        )
