"""DeepSeekClient 测试：reasoning_content 解析、ThinkingChunk 输出、用量透传与全链路。"""

import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

import pytest

from infinity_agent.clients.deepseek.client import DeepSeekClient
from infinity_agent.clients.deepseek.config import DeepSeekConfig, DeepSeekRequestConfig
from infinity_agent.clients.deepseek.response_models import (
    DeepSeekChoice,
    DeepSeekDelta,
    DeepSeekStreamEvent,
)
from infinity_agent.clients.exceptions import LLMConfigError, LLMContentFilterError
from infinity_agent.clients.open_ai.client import ToolCallAccumulator
from infinity_agent.clients.open_ai.response_models import (
    ToolCallDelta,
    ToolCallFunction,
)
from infinity_agent.models import (
    DoneChunk,
    FinishChunk,
    Message,
    TextChunk,
    ThinkingChunk,
    ToolCallCompleteChunk,
    UsageChunk,
)


def _make_client(config: DeepSeekConfig | None = None) -> DeepSeekClient:
    cfg = config or DeepSeekConfig(model='deepseek-reasoner', api_key='sk-test')
    return DeepSeekClient(cfg)


def _event(
    delta: DeepSeekDelta,
    *,
    finish_reason: str | None = None,
    usage: dict | None = None,
) -> DeepSeekStreamEvent:
    return DeepSeekStreamEvent(
        id='chatcmpl-1',
        choices=[DeepSeekChoice(index=0, delta=delta, finish_reason=finish_reason)],
        created=1677652288,
        model='deepseek-reasoner',
        usage=usage,
    )


def _event_json(delta: dict, *, finish_reason: str | None = None, usage: dict | None = None) -> str:
    payload = {
        'id': 'chatcmpl-1',
        'object': 'chat.completion.chunk',
        'created': 1677652288,
        'model': 'deepseek-reasoner',
        'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}],
    }
    if usage is not None:
        payload['usage'] = usage
    return json.dumps(payload, ensure_ascii=False)


def _fake_connection(client: DeepSeekClient, lines: List[bytes]) -> None:
    """将 client._conn.request 替换为返回给定 SSE 字节行的假实现。"""

    @asynccontextmanager
    async def fake_request(endpoint: str, request_model):
        async def gen() -> AsyncGenerator[bytes, None]:
            for line in lines:
                yield line

        yield gen()

    client._conn.request = fake_request  # type: ignore[method-assign]


class TestDeepSeekResponseModels:
    """响应模型保留思考内容"""

    def test_deepseek_delta_keeps_reasoning_content(self) -> None:
        delta = DeepSeekDelta.model_validate(
            {'content': 'hello', 'reasoning_content': '让我想想'}
        )
        assert delta.content == 'hello'
        assert delta.reasoning_content == '让我想想'

    def test_stream_event_parses_reasoning_content(self) -> None:
        event = DeepSeekStreamEvent.model_validate_json(_event_json({'reasoning_content': '思考'}))
        assert event.choices[0].delta.reasoning_content == '思考'


class TestParseSseStream:
    """DeepSeek SSE 解析"""

    @pytest.mark.asyncio
    async def test_yields_deepseek_stream_event(self) -> None:
        client = _make_client()
        lines = [
            b'data: ' + _event_json({'reasoning_content': '思考中'}).encode() + b'\n',
        ]
        events = [e async for e in client._parse_sse_stream(_agen(lines))]
        assert len(events) == 1
        assert isinstance(events[0], DeepSeekStreamEvent)
        assert events[0].choices[0].delta.reasoning_content == '思考中'


class TestHandleSseEvent:
    """DeepSeek 事件分发管线"""

    def _client(self) -> DeepSeekClient:
        return _make_client()

    @pytest.mark.asyncio
    async def test_thinking_then_text(self) -> None:
        client = self._client()
        chunks = [
            c
            async for c in client._handle_sse_event(
                _event(DeepSeekDelta(reasoning_content='思考', content='答案')),
                ToolCallAccumulator(),
                include_usage=False,
            )
        ]
        assert len(chunks) == 2
        assert isinstance(chunks[0], ThinkingChunk)
        assert chunks[0].text == '思考'
        assert isinstance(chunks[1], TextChunk)
        assert chunks[1].text == '答案'

    @pytest.mark.asyncio
    async def test_reasoning_only(self) -> None:
        client = self._client()
        chunks = [
            c
            async for c in client._handle_sse_event(
                _event(DeepSeekDelta(reasoning_content='纯思考')),
                ToolCallAccumulator(),
                include_usage=False,
            )
        ]
        assert len(chunks) == 1
        assert isinstance(chunks[0], ThinkingChunk)

    @pytest.mark.asyncio
    async def test_usage_with_reasoning_tokens(self) -> None:
        client = self._client()
        event = _event(
            DeepSeekDelta(),
            usage={
                'prompt_tokens': 10,
                'completion_tokens': 20,
                'total_tokens': 30,
                'completion_tokens_details': {'reasoning_tokens': 15},
            },
        )
        chunks = [
            c
            async for c in client._handle_sse_event(
                event, ToolCallAccumulator(), include_usage=True
            )
        ]
        assert len(chunks) == 1
        assert isinstance(chunks[0], UsageChunk)
        assert chunks[0].usage.reasoning_tokens == 15
        assert chunks[0].usage.completion_tokens == 20

    @pytest.mark.asyncio
    async def test_usage_without_reasoning_details(self) -> None:
        client = self._client()
        event = _event(
            DeepSeekDelta(),
            usage={'prompt_tokens': 1, 'completion_tokens': 2, 'total_tokens': 3},
        )
        chunks = [
            c
            async for c in client._handle_sse_event(
                event, ToolCallAccumulator(), include_usage=True
            )
        ]
        assert chunks[0].usage.reasoning_tokens is None

    @pytest.mark.asyncio
    async def test_finish_reason_flushes_tool_calls(self) -> None:
        client = self._client()
        acc = ToolCallAccumulator()
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
                _event(DeepSeekDelta(), finish_reason='tool_calls'),
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
    async def test_content_filter_raises(self) -> None:
        client = self._client()
        with pytest.raises(LLMContentFilterError):
            async for _ in client._handle_sse_event(
                _event(DeepSeekDelta(), finish_reason='content_filter'),
                ToolCallAccumulator(),
                include_usage=False,
            ):
                pass


class TestDeepSeekStreamChat:
    """DeepSeekClient 全链路 stream_chat"""

    @pytest.mark.asyncio
    async def test_wrong_config_type(self) -> None:
        from infinity_agent.clients.config import RequestConfig

        client = _make_client()
        with pytest.raises(LLMConfigError, match='expects DeepSeekRequestConfig'):
            async for _ in client.stream_chat([Message.user('Hi')], RequestConfig()):
                pass

    @pytest.mark.asyncio
    async def test_full_pipeline_with_thinking_and_usage(self) -> None:
        client = _make_client()
        usage = {
            'prompt_tokens': 10,
            'completion_tokens': 5,
            'total_tokens': 15,
            'completion_tokens_details': {'reasoning_tokens': 3},
        }
        lines = [
            b'data: ' + _event_json({'reasoning_content': '思考中', 'content': '你好'}).encode() + b'\n',
            b'data: ' + _event_json({'content': '世界'}).encode() + b'\n',
            b'data: ' + _event_json({}, usage=usage).encode() + b'\n',
            b'data: [DONE]\n',
        ]
        _fake_connection(client, lines)
        chunks = [c async for c in client.stream_chat([Message.user('hi')])]
        types = [type(c).__name__ for c in chunks]
        assert types == ['ThinkingChunk', 'TextChunk', 'TextChunk', 'UsageChunk', 'DoneChunk']
        assert chunks[0].text == '思考中'
        assert chunks[1].text == '你好'
        assert chunks[2].text == '世界'
        assert chunks[3].usage.reasoning_tokens == 3
        assert isinstance(chunks[4], DoneChunk)

    @pytest.mark.asyncio
    async def test_build_payload_forwarded_to_connection(self) -> None:
        """请求体应携带 reasoning_effort 与 stream 标记"""
        client = _make_client()
        captured: dict = {}

        @asynccontextmanager
        async def fake_request(endpoint: str, request_model):
            captured['endpoint'] = endpoint
            captured['model'] = request_model.model
            captured['reasoning_effort'] = request_model.reasoning_effort
            captured['stream'] = request_model.stream

            async def gen() -> AsyncGenerator[bytes, None]:
                yield b'data: [DONE]\n'

            yield gen()

        client._conn.request = fake_request  # type: ignore[method-assign]
        chunks = [
            c
            async for c in client.stream_chat(
                [Message.user('hi')],
                DeepSeekRequestConfig(reasoning_effort='high'),
            )
        ]
        assert chunks[-1].type == 'done'
        assert captured['endpoint'] == 'chat/completions'
        assert captured['model'] == 'deepseek-reasoner'
        assert captured['reasoning_effort'] == 'high'
        assert captured['stream'] is True


async def _agen(lines: List[bytes]) -> AsyncGenerator[bytes, None]:
    for line in lines:
        yield line
