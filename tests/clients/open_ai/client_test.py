"""OpenAIClient 测试：SSE 解析、事件分发、流聚合与收尾。"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, List
from unittest.mock import AsyncMock

import pytest

from infinity_agent.clients.exceptions import LLMConfigError, LLMContentFilterError, LLMStreamError
from infinity_agent.clients.open_ai.client import OpenAIClient, _ToolCallAccumulator
from infinity_agent.clients.open_ai.config import OpenAIConfig, OpenAIRequestConfig
from infinity_agent.clients.open_ai.response_models import (
    Choice,
    Delta,
    StreamEvent,
    ToolCallDelta,
    ToolCallFunction,
)
from infinity_agent.models import (
    DoneChunk,
    FinishChunk,
    Message,
    TextChunk,
    ToolCallCompleteChunk,
    UsageChunk,
)


def _make_client(config: OpenAIConfig | None = None) -> OpenAIClient:
    cfg = config or OpenAIConfig(model='gpt-4o-mini', api_key='sk-test')
    return OpenAIClient(cfg)


def _event(delta: Delta, *, finish_reason: str | None = None, usage: dict | None = None) -> StreamEvent:
    return StreamEvent(
        id='chatcmpl-1',
        choices=[Choice(index=0, delta=delta, finish_reason=finish_reason)],
        created=1677652288,
        model='gpt-4o-mini',
        usage=usage,
    )


def _event_json(delta: dict, *, finish_reason: str | None = None) -> str:
    payload = {
        'id': 'chatcmpl-1',
        'object': 'chat.completion.chunk',
        'created': 1677652288,
        'model': 'gpt-4o-mini',
        'choices': [
            {'index': 0, 'delta': delta, 'finish_reason': finish_reason}
        ],
    }
    import json

    return json.dumps(payload, ensure_ascii=False)


def _fake_connection(client: OpenAIClient, lines: List[bytes]) -> None:
    """将 client._conn.request 替换为返回给定 SSE 字节行的假实现。"""

    @asynccontextmanager
    async def fake_request(endpoint: str, request_model):
        async def gen() -> AsyncGenerator[bytes, None]:
            for line in lines:
                yield line

        yield gen()

    client._conn.request = fake_request  # type: ignore[method-assign]


class TestOpenAIClientBasics:
    """客户端基础行为"""

    def test_model_property(self, openai_config: OpenAIConfig) -> None:
        client = _make_client(openai_config)
        assert client.model == 'gpt-4o-mini'

    @pytest.mark.asyncio
    async def test_context_manager(self, openai_config: OpenAIConfig) -> None:
        client = _make_client(openai_config)
        client._conn.ensure_session = AsyncMock()
        client._conn.close = AsyncMock()
        async with client:
            client._conn.ensure_session.assert_awaited_once()
        client._conn.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stream_chat_wrong_config_type(self) -> None:
        from infinity_agent.clients.config import RequestConfig

        client = _make_client()
        with pytest.raises(LLMConfigError, match='expects OpenAIRequestConfig'):
            async for _ in client.stream_chat([Message.user('Hi')], RequestConfig()):
                pass


class TestParseSseStream:
    """SSE 字节行解析"""

    def _client(self) -> OpenAIClient:
        return _make_client()

    @pytest.mark.asyncio
    async def test_parses_data_lines(self) -> None:
        client = self._client()
        lines = [
            b'data: ' + _event_json({'content': 'Hello'}).encode() + b'\n',
            b'data: ' + _event_json({'content': ' World'}).encode() + b'\n',
        ]
        events = [e async for e in client._parse_sse_stream(_agen(lines))]
        assert len(events) == 2
        assert events[0].choices[0].delta.content == 'Hello'

    @pytest.mark.asyncio
    async def test_skips_comment_and_non_data_lines(self) -> None:
        client = self._client()
        lines = [
            b': keep-alive\n',
            b'\n',
            b'data: ' + _event_json({'content': 'Hi'}).encode() + b'\n',
            b'event: message\n',
        ]
        events = [e async for e in client._parse_sse_stream(_agen(lines))]
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_done_terminates_stream(self) -> None:
        client = self._client()
        lines = [
            b'data: ' + _event_json({'content': 'Hi'}).encode() + b'\n',
            b'data: [DONE]\n',
            b'data: ' + _event_json({'content': 'ignored'}).encode() + b'\n',
        ]
        events = [e async for e in client._parse_sse_stream(_agen(lines))]
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_invalid_json_raises_stream_error(self) -> None:
        client = self._client()
        with pytest.raises(LLMStreamError, match='JSON decode error'):
            async for _ in client._parse_sse_stream(_agen([b'data: {not json}\n'])):
                pass


class TestHandleSseEvent:
    """单事件分发管线"""

    def _client(self) -> OpenAIClient:
        return _make_client()

    @pytest.mark.asyncio
    async def test_text_chunk(self) -> None:
        client = self._client()
        chunks = [
            c
            async for c in client._handle_sse_event(
                _event(Delta(content='Hello')),
                _ToolCallAccumulator(),
                include_usage=False,
            )
        ]
        assert len(chunks) == 1
        assert isinstance(chunks[0], TextChunk)
        assert chunks[0].text == 'Hello'

    @pytest.mark.asyncio
    async def test_finish_reason_flushes_tool_calls(self) -> None:
        client = self._client()
        acc = _ToolCallAccumulator()
        acc.extend(
            [
                ToolCallDelta(
                    index=0,
                    id='call_1',
                    function=ToolCallFunction(name='f', arguments='{"x":1}'),
                )
            ]
        )
        chunks = [
            c
            async for c in client._handle_sse_event(
                _event(Delta(), finish_reason='tool_calls'),
                acc,
                include_usage=False,
            )
        ]
        assert len(chunks) == 2
        assert isinstance(chunks[0], ToolCallCompleteChunk)
        assert chunks[0].tool_calls[0].function.arguments == {'x': 1}
        assert isinstance(chunks[1], FinishChunk)
        assert chunks[1].finish_reason == 'tool_calls'

    @pytest.mark.asyncio
    async def test_finish_reason_without_pending_tool_calls(self) -> None:
        client = self._client()
        chunks = [
            c
            async for c in client._handle_sse_event(
                _event(Delta(), finish_reason='stop'),
                _ToolCallAccumulator(),
                include_usage=False,
            )
        ]
        assert len(chunks) == 1
        assert isinstance(chunks[0], FinishChunk)
        assert chunks[0].finish_reason == 'stop'

    @pytest.mark.asyncio
    async def test_content_filter_raises(self) -> None:
        client = self._client()
        with pytest.raises(LLMContentFilterError):
            async for _ in client._handle_sse_event(
                _event(Delta(), finish_reason='content_filter'),
                _ToolCallAccumulator(),
                include_usage=False,
            ):
                pass

    @pytest.mark.asyncio
    async def test_usage_chunk_when_include_usage(self) -> None:
        client = self._client()
        event = _event(
            Delta(),
            usage={'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
        )
        chunks = [
            c
            async for c in client._handle_sse_event(
                event, _ToolCallAccumulator(), include_usage=True
            )
        ]
        assert len(chunks) == 1
        assert isinstance(chunks[0], UsageChunk)
        assert chunks[0].usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_usage_ignored_when_not_requested(self) -> None:
        client = self._client()
        event = _event(
            Delta(),
            usage={'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
        )
        chunks = [
            c
            async for c in client._handle_sse_event(
                event, _ToolCallAccumulator(), include_usage=False
            )
        ]
        assert chunks == []


class TestOnStreamEnd:
    """流收尾处理"""

    @pytest.mark.asyncio
    async def test_flush_and_done(self) -> None:
        client = _make_client()
        acc = _ToolCallAccumulator()
        acc.extend(
            [
                ToolCallDelta(
                    index=0,
                    id='call_1',
                    function=ToolCallFunction(name='f', arguments='{}'),
                )
            ]
        )
        chunks = [c async for c in client._on_stream_end(acc)]
        assert len(chunks) == 2
        assert isinstance(chunks[0], ToolCallCompleteChunk)
        assert isinstance(chunks[1], DoneChunk)

    @pytest.mark.asyncio
    async def test_only_done_when_nothing_pending(self) -> None:
        client = _make_client()
        chunks = [c async for c in client._on_stream_end(_ToolCallAccumulator())]
        assert len(chunks) == 1
        assert isinstance(chunks[0], DoneChunk)


class TestToolCallAccumulator:
    """跨 chunk 工具调用累积器"""

    def test_flush_returns_aggregated(self) -> None:
        acc = _ToolCallAccumulator()
        acc.extend(
            [
                ToolCallDelta(index=0, id='c1', function=ToolCallFunction(name='f', arguments='{"a":')),
                ToolCallDelta(index=0, function=ToolCallFunction(arguments='1}')),
            ]
        )
        result = acc.flush()
        assert len(result) == 1
        assert result[0].function.arguments == {'a': 1}
        # flush 后清空
        assert acc.flush() == []

    def test_flush_empty(self) -> None:
        acc = _ToolCallAccumulator()
        assert acc.flush() == []

    def test_flush_as_chunk_none_when_empty(self) -> None:
        acc = _ToolCallAccumulator()
        assert acc.flush_as_chunk() is None

    def test_flush_as_chunk(self) -> None:
        acc = _ToolCallAccumulator()
        acc.extend([ToolCallDelta(index=0, id='c1', function=ToolCallFunction(name='f', arguments='{}'))])
        chunk = acc.flush_as_chunk()
        assert isinstance(chunk, ToolCallCompleteChunk)
        assert chunk.tool_calls[0].id == 'c1'


class TestStreamChatIntegration:
    """stream_chat 端到端集成"""

    @pytest.mark.asyncio
    async def test_full_text_stream(self) -> None:
        client = _make_client()
        lines = [
            b'data: ' + _event_json({'content': 'Hello'}).encode() + b'\n',
            b'data: ' + _event_json({'content': ' World'}, finish_reason='stop').encode() + b'\n',
            b'data: [DONE]\n',
        ]
        _fake_connection(client, lines)

        chunks = [c async for c in client.stream_chat([Message.user('Hi')])]
        types = [c.type for c in chunks]
        assert types == ['text', 'text', 'finish', 'done']
        assert chunks[0].text == 'Hello'
        assert chunks[1].text == ' World'
        assert chunks[2].finish_reason == 'stop'

    @pytest.mark.asyncio
    async def test_tool_call_stream(self) -> None:
        client = _make_client()
        lines = [
            b'data: '
            + _event_json(
                {
                    'tool_calls': [
                        {
                            'index': 0,
                            'id': 'call_1',
                            'type': 'function',
                            'function': {'name': 'get_weather', 'arguments': '{"cit'},
                        }
                    ]
                }
            ).encode()
            + b'\n',
            b'data: '
            + _event_json(
                {
                    'tool_calls': [
                        {
                            'index': 0,
                            'function': {'arguments': 'y":"北京"}'},
                        }
                    ]
                }
            ).encode()
            + b'\n',
            b'data: ' + _event_json({}, finish_reason='tool_calls').encode() + b'\n',
            b'data: [DONE]\n',
        ]
        _fake_connection(client, lines)

        chunks = [c async for c in client.stream_chat([Message.user('Hi')])]
        tool_chunks = [c for c in chunks if isinstance(c, ToolCallCompleteChunk)]
        assert len(tool_chunks) == 1
        tc = tool_chunks[0].tool_calls[0]
        assert tc.id == 'call_1'
        assert tc.function.name == 'get_weather'
        assert tc.function.arguments == {'city': '北京'}
        assert isinstance(chunks[-1], DoneChunk)

    @pytest.mark.asyncio
    async def test_usage_stream(self) -> None:
        client = _make_client()
        usage_line = (
            '{"id":"x","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini",'
            '"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}'
        )
        lines = [
            b'data: ' + _event_json({'content': 'Hi'}).encode() + b'\n',
            b'data: ' + usage_line.encode() + b'\n',
            b'data: [DONE]\n',
        ]
        _fake_connection(client, lines)

        chunks = [c async for c in client.stream_chat([Message.user('Hi')])]
        usage_chunks = [c for c in chunks if isinstance(c, UsageChunk)]
        assert len(usage_chunks) == 1
        assert usage_chunks[0].usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_content_filter_raises(self) -> None:
        client = _make_client()
        lines = [
            b'data: '
            + _event_json({}, finish_reason='content_filter').encode()
            + b'\n',
        ]
        _fake_connection(client, lines)
        with pytest.raises(LLMContentFilterError):
            async for _ in client.stream_chat([Message.user('Hi')]):
                pass

    @pytest.mark.asyncio
    async def test_request_config_passed(self) -> None:
        """验证请求配置被用于构建 payload（include_usage=False → 无 usage chunk）"""
        client = _make_client()
        captured: dict = {}

        @asynccontextmanager
        async def fake_request(endpoint: str, request_model):
            captured['model'] = request_model
            async def gen() -> AsyncGenerator[bytes, None]:
                yield b'data: [DONE]\n'

            yield gen()

        client._conn.request = fake_request  # type: ignore[method-assign]
        req_config = OpenAIRequestConfig(include_usage=False)
        chunks = [
            c
            async for c in client.stream_chat([Message.user('Hi')], req_config)
        ]
        assert isinstance(chunks[0], DoneChunk)
        assert captured['model'].model == 'gpt-4o-mini'
        assert captured['model'].stream_options is None


async def _agen(lines: List[bytes]) -> AsyncGenerator[bytes, None]:
    for line in lines:
        yield line
