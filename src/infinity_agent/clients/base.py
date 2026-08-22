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
    ThinkingChunk,
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

    @classmethod
    def chunks_to_message(cls, chunks: list[StreamChunk]) -> Message:
        """
        将流式 chunks 聚合为一条完整的 assistant Message

        遍历所有 chunk，拼接文本内容、思考内容并收集工具调用，
        最终产出一条可直接追加到对话历史的 assistant 消息。

        子类可通过覆写 :meth:`_handle_chunk` 聚合自定义 chunk 类型，
        或覆写 :meth:`_assemble` 定制 assistant 消息的构造方式。

        :param chunks: 完整的流式 chunk 列表
        :return: role=assistant 的 Message
        """
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for chunk in chunks:
            cls._handle_chunk(chunk, text_parts, reasoning_parts, tool_calls)

        content = ''.join(text_parts) if text_parts else None
        reasoning_content = ''.join(reasoning_parts) if reasoning_parts else None
        return cls._assemble(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
        )

    @classmethod
    def _handle_chunk(
        cls,
        chunk: StreamChunk,
        text_parts: list[str],
        reasoning_parts: list[str],
        tool_calls: list[ToolCall],
    ) -> None:
        """单 chunk 聚合钩子（可覆写）。

        子类可在处理自定义 chunk 类型后回退到父类实现，
        例如 ``super()._handle_chunk(chunk, ...)``。

        :param chunk: 待聚合的流式 chunk
        :param text_parts: 文本增量累积列表
        :param reasoning_parts: 思考内容增量累积列表
        :param tool_calls: 聚合完成的工具调用累积列表
        """
        if isinstance(chunk, TextChunk):
            text_parts.append(chunk.text)
        elif isinstance(chunk, ThinkingChunk):
            reasoning_parts.append(chunk.text)
        elif isinstance(chunk, ToolCallCompleteChunk):
            tool_calls.extend(chunk.tool_calls)

    @classmethod
    def _assemble(
        cls,
        *,
        content: Optional[str],
        reasoning_content: Optional[str],
        tool_calls: list[ToolCall],
    ) -> Message:
        """组装 assistant 消息钩子（可覆写），子类可定制 Message 构造。"""
        return Message.assistant(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls if tool_calls else None,
        )
