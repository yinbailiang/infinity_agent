"""流式 Chunk 模型测试：StreamChunk 及子类。"""

import pytest
from pydantic import ValidationError

from infinity_agent.models import (
    DoneChunk,
    FinishChunk,
    StreamChunk,
    TextChunk,
    ToolCall,
    ToolCallCompleteChunk,
    UsageChunk,
    UsageStats,
)


class TestUsageStats:
    """Token 使用统计"""

    def test_basic(self) -> None:
        stats = UsageStats(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        assert stats.prompt_tokens == 10
        assert stats.completion_tokens == 5
        assert stats.total_tokens == 15

    def test_negative_tokens_rejected(self) -> None:
        with pytest.raises(ValidationError):
            UsageStats(prompt_tokens=-1, completion_tokens=0, total_tokens=0)


class TestStreamChunk:
    """流式响应单元基类"""

    def test_cannot_instantiate_base(self) -> None:
        """基类缺少 type 默认值，直接实例化会报错"""
        with pytest.raises(ValidationError):
            StreamChunk()


class TestTextChunk:
    """文本增量片段"""

    def test_default_type(self) -> None:
        chunk = TextChunk(text='Hello')
        assert chunk.type == 'text'
        assert chunk.text == 'Hello'

    def test_is_stream_chunk(self) -> None:
        assert isinstance(TextChunk(text='x'), StreamChunk)


class TestToolCallCompleteChunk:
    """聚合完成的工具调用"""

    def test_default_type_and_empty_tool_calls(self) -> None:
        chunk = ToolCallCompleteChunk()
        assert chunk.type == 'toolcall_complete'
        assert chunk.tool_calls == []

    def test_with_tool_calls(self) -> None:
        tc = ToolCall(id='c1', function={'name': 'f', 'arguments': {}})
        chunk = ToolCallCompleteChunk(tool_calls=[tc])
        assert chunk.tool_calls == [tc]


class TestUsageChunk:
    """Token 使用统计 chunk"""

    def test_default_type(self) -> None:
        stats = UsageStats(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        chunk = UsageChunk(usage=stats)
        assert chunk.type == 'usage'
        assert chunk.usage.total_tokens == 3


class TestFinishChunk:
    """流结束理由 chunk"""

    def test_default_type(self) -> None:
        chunk = FinishChunk(finish_reason='stop')
        assert chunk.type == 'finish'
        assert chunk.finish_reason == 'stop'


class TestDoneChunk:
    """流完全结束标记"""

    def test_default_type(self) -> None:
        chunk = DoneChunk()
        assert chunk.type == 'done'
