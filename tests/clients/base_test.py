"""LLMClient 抽象基类测试：chunks_to_message 类方法及其子类扩展。"""

from typing import List, Optional

from infinity_agent.clients.base import LLMClient
from infinity_agent.models import (
    DoneChunk,
    FinishChunk,
    Message,
    MessageRole,
    StreamChunk,
    TextChunk,
    ThinkingChunk,
    ToolCall,
    ToolCallCompleteChunk,
    UsageChunk,
    UsageStats,
)


class _FakeChunk(StreamChunk):
    """自定义 chunk 类型，用于测试子类扩展聚合。"""

    type: str = 'fake'
    extra: str = ''


class _ExtendingClient(LLMClient):
    """演示子类扩展：覆写 _handle_chunk 聚合自定义 chunk 类型。"""

    @classmethod
    def _handle_chunk(
        cls,
        chunk: StreamChunk,
        text_parts: List[str],
        reasoning_parts: List[str],
        tool_calls: List[ToolCall],
    ) -> None:
        if isinstance(chunk, _FakeChunk):
            text_parts.append(chunk.extra)
            return
        super()._handle_chunk(chunk, text_parts, reasoning_parts, tool_calls)


class _CustomAssembleClient(LLMClient):
    """演示子类扩展：覆写 _assemble 定制 Message 构造。"""

    @classmethod
    def _assemble(
        cls,
        *,
        content: Optional[str],
        reasoning_content: Optional[str],
        tool_calls: List[ToolCall],
    ) -> Message:
        msg = super()._assemble(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
        )
        msg.name = 'extended'
        return msg


class TestChunksToMessage:
    """将流式 chunks 聚合为 assistant 消息"""

    def test_text_only_chunks(self) -> None:
        chunks = [TextChunk(text='Hello'), TextChunk(text=' '), TextChunk(text='World')]
        msg = LLMClient.chunks_to_message(chunks)
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == 'Hello World'
        assert msg.tool_calls is None

    def test_tool_call_chunks(self) -> None:
        tc = ToolCall(id='c1', function={'name': 'get_weather', 'arguments': {}})
        chunks = [ToolCallCompleteChunk(tool_calls=[tc])]
        msg = LLMClient.chunks_to_message(chunks)
        assert msg.content is None
        assert msg.tool_calls == [tc]

    def test_mixed_text_and_tool_calls(self) -> None:
        tc = ToolCall(id='c1', function={'name': 'f', 'arguments': {'x': 1}})
        chunks = [TextChunk(text='Calling... '), ToolCallCompleteChunk(tool_calls=[tc])]
        msg = LLMClient.chunks_to_message(chunks)
        assert msg.content == 'Calling... '
        assert msg.tool_calls == [tc]

    def test_ignores_non_aggregating_chunks(self) -> None:
        """usage / finish / done chunk 不参与聚合"""
        stats = UsageStats(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        chunks = [
            TextChunk(text='hi'),
            UsageChunk(usage=stats),
            FinishChunk(finish_reason='stop'),
            DoneChunk(),
        ]
        msg = LLMClient.chunks_to_message(chunks)
        assert msg.content == 'hi'
        assert msg.tool_calls is None

    def test_empty_chunks(self) -> None:
        msg = LLMClient.chunks_to_message([])
        assert msg.content is None
        assert msg.tool_calls is None

    def test_multiple_tool_call_chunks_concatenate(self) -> None:
        tc1 = ToolCall(id='c1', function={'name': 'a', 'arguments': {}})
        tc2 = ToolCall(id='c2', function={'name': 'b', 'arguments': {}})
        chunks = [ToolCallCompleteChunk(tool_calls=[tc1]), ToolCallCompleteChunk(tool_calls=[tc2])]
        msg = LLMClient.chunks_to_message(chunks)
        assert msg.tool_calls == [tc1, tc2]

    def test_thinking_chunks_aggregate_to_reasoning_content(self) -> None:
        chunks = [
            ThinkingChunk(text='考虑中 '),
            ThinkingChunk(text='再想想'),
            TextChunk(text='最终答案'),
        ]
        msg = LLMClient.chunks_to_message(chunks)
        assert msg.content == '最终答案'
        assert msg.reasoning_content == '考虑中 再想想'

    def test_thinking_only_chunks(self) -> None:
        chunks = [ThinkingChunk(text='思考')]
        msg = LLMClient.chunks_to_message(chunks)
        assert msg.content is None
        assert msg.reasoning_content == '思考'


class TestSubclassExtension:
    """子类通过覆写钩子扩展聚合行为"""

    def test_handle_chunk_extension(self) -> None:
        chunks = [_FakeChunk(extra='custom '), TextChunk(text='world')]
        msg = _ExtendingClient.chunks_to_message(chunks)
        assert msg.content == 'custom world'

    def test_handle_chunk_extension_falls_back_to_super(self) -> None:
        chunks = [ThinkingChunk(text='想'), _FakeChunk(extra='custom')]
        msg = _ExtendingClient.chunks_to_message(chunks)
        assert msg.reasoning_content == '想'
        assert msg.content == 'custom'

    def test_assemble_extension(self) -> None:
        msg = _CustomAssembleClient.chunks_to_message([TextChunk(text='hi')])
        assert msg.name == 'extended'
        assert msg.content == 'hi'
