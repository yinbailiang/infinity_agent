"""OpenAI 响应模型测试：StreamEvent 及其子模型。"""

import pytest
from pydantic import ValidationError

from infinity_agent.clients.open_ai.response_models import (
    Choice,
    CompletionUsage,
    Delta,
    StreamEvent,
    ToolCallDelta,
    ToolCallFunction,
)


class TestToolCallDelta:
    """工具调用增量"""

    def test_defaults(self) -> None:
        delta = ToolCallDelta(index=0)
        assert delta.type == 'function'
        assert delta.id is None
        assert delta.function is None

    def test_full(self) -> None:
        delta = ToolCallDelta(
            index=1,
            id='call_x',
            function=ToolCallFunction(name='get_weather', arguments='{"city":'),
        )
        assert delta.index == 1
        assert delta.id == 'call_x'
        assert delta.function.name == 'get_weather'
        assert delta.function.arguments == '{"city":'

    def test_function_default_arguments(self) -> None:
        fn = ToolCallFunction(name='f')
        assert fn.arguments == ''


class TestDelta:
    """completion 增量"""

    def test_defaults(self) -> None:
        delta = Delta()
        assert delta.content is None
        assert delta.role is None
        assert delta.tool_calls is None

    def test_content(self) -> None:
        delta = Delta(content='Hello')
        assert delta.content == 'Hello'

    def test_invalid_role_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Delta(role='user')


class TestCompletionUsage:
    """Token 使用统计"""

    def test_full(self) -> None:
        usage = CompletionUsage(
            completion_tokens=5,
            prompt_tokens=10,
            total_tokens=15,
        )
        assert usage.total_tokens == 15

    def test_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            CompletionUsage(completion_tokens=1, prompt_tokens=2)


class TestStreamEvent:
    """OpenAI 流式响应单个数据块"""

    def test_parse_from_json(self) -> None:
        raw = {
            'id': 'chatcmpl-123',
            'object': 'chat.completion.chunk',
            'created': 1677652288,
            'model': 'gpt-4o-mini',
            'choices': [
                {
                    'index': 0,
                    'delta': {'content': 'Hello'},
                    'finish_reason': None,
                }
            ],
        }
        event = StreamEvent.model_validate(raw)
        assert event.id == 'chatcmpl-123'
        assert event.object == 'chat.completion.chunk'
        assert event.created == 1677652288
        assert event.model == 'gpt-4o-mini'
        assert len(event.choices) == 1
        assert event.choices[0].delta.content == 'Hello'
        assert event.choices[0].finish_reason is None

    def test_parse_with_tool_calls(self) -> None:
        raw = {
            'id': 'chatcmpl-123',
            'created': 1677652288,
            'model': 'gpt-4o-mini',
            'choices': [
                {
                    'index': 0,
                    'delta': {
                        'tool_calls': [
                            {
                                'index': 0,
                                'id': 'call_x',
                                'type': 'function',
                                'function': {'name': 'get_weather', 'arguments': '{"city":'},
                            }
                        ]
                    },
                    'finish_reason': None,
                }
            ],
        }
        event = StreamEvent.model_validate(raw)
        delta = event.choices[0].delta.tool_calls[0]
        assert delta.id == 'call_x'
        assert delta.function.name == 'get_weather'

    def test_parse_with_usage(self) -> None:
        raw = {
            'id': 'chatcmpl-123',
            'created': 1677652288,
            'model': 'gpt-4o-mini',
            'choices': [],
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 5,
                'total_tokens': 15,
            },
        }
        event = StreamEvent.model_validate(raw)
        assert event.usage is not None
        assert event.usage.total_tokens == 15

    def test_object_locked_to_chunk(self) -> None:
        raw = {
            'id': 'x',
            'created': 1,
            'model': 'm',
            'object': 'chat.completion',
            'choices': [],
        }
        with pytest.raises(ValidationError):
            StreamEvent.model_validate(raw)

    def test_choices_required(self) -> None:
        raw = {'id': 'x', 'created': 1, 'model': 'm'}
        with pytest.raises(ValidationError):
            StreamEvent.model_validate(raw)

    def test_finish_reason_valid_values(self) -> None:
        for reason in ('stop', 'length', 'content_filter', 'tool_calls', 'function_call'):
            event = StreamEvent.model_validate(
                {
                    'id': 'x',
                    'created': 1,
                    'model': 'm',
                    'choices': [
                        {'index': 0, 'delta': {}, 'finish_reason': reason}
                    ],
                }
            )
            assert event.choices[0].finish_reason == reason

    def test_invalid_finish_reason_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StreamEvent.model_validate(
                {
                    'id': 'x',
                    'created': 1,
                    'model': 'm',
                    'choices': [
                        {'index': 0, 'delta': {}, 'finish_reason': 'nope'}
                    ],
                }
            )


class TestChoice:
    """completion 选择项"""

    def test_requires_index_and_delta(self) -> None:
        with pytest.raises(ValidationError):
            Choice(delta=Delta())
        with pytest.raises(ValidationError):
            Choice(index=0)
        choice = Choice(index=0, delta=Delta())
        assert choice.index == 0
