"""LLMClient 抽象基类测试：chunks_to_message 静态方法。"""

from infinity_agent.clients.base import LLMClient
from infinity_agent.models import (
    DoneChunk,
    FinishChunk,
    MessageRole,
    TextChunk,
    ToolCall,
    ToolCallCompleteChunk,
    UsageChunk,
    UsageStats,
)


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
