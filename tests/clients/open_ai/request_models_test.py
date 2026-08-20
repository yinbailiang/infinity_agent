"""OpenAI 请求模型测试：ChatCompletionRequest、ResponseFormat、JsonSchema、StreamOptions。"""

import pytest
from pydantic import BaseModel, ValidationError

from infinity_agent.clients.open_ai.request_models import (
    ChatCompletionRequest,
    JsonSchema,
    ResponseFormat,
    StreamOptions,
)


class TestJsonSchema:
    """结构化输出 schema"""

    def test_serializes_model_to_json_schema(self) -> None:
        class TickOutput(BaseModel):
            symbol: str
            price: float

        schema = JsonSchema(name='tick_output', schema=TickOutput)
        dumped = schema.model_dump(mode='json', by_alias=True)
        assert dumped['name'] == 'tick_output'
        assert dumped['schema'] == {
            'title': 'TickOutput',
            'type': 'object',
            'properties': {
                'symbol': {'title': 'Symbol', 'type': 'string'},
                'price': {'title': 'Price', 'type': 'number'},
            },
            'required': ['symbol', 'price'],
        }
        assert dumped['strict'] is True


class TestResponseFormat:
    """响应格式约束"""

    def test_default_json_object(self) -> None:
        fmt = ResponseFormat()
        assert fmt.type == 'json_object'
        assert fmt.json_schema is None

    def test_json_object(self) -> None:
        fmt = ResponseFormat(type='json_object')
        assert fmt.type == 'json_object'

    def test_json_schema(self) -> None:
        class TickOutput(BaseModel):
            symbol: str

        fmt = ResponseFormat(
            type='json_schema',
            json_schema=JsonSchema(name='tick', schema=TickOutput),
        )
        assert fmt.json_schema is not None
        assert fmt.json_schema.name == 'tick'

    def test_invalid_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ResponseFormat(type='yaml')


class TestStreamOptions:
    """流式选项"""

    def test_default(self) -> None:
        opts = StreamOptions()
        assert opts.include_usage is False

    def test_custom(self) -> None:
        opts = StreamOptions(include_usage=True)
        assert opts.include_usage is True


class TestChatCompletionRequest:
    """Chat Completion 请求体"""

    def test_minimal(self) -> None:
        req = ChatCompletionRequest(
            model='gpt-4o-mini',
            messages=[{'role': 'user', 'content': 'Hi'}],
        )
        assert req.stream is False
        assert req.stream_options is None
        assert req.tools is None
        assert req.response_format is None

    def test_temperature_range(self) -> None:
        with pytest.raises(ValidationError):
            ChatCompletionRequest(model='m', messages=[], temperature=-0.1)
        with pytest.raises(ValidationError):
            ChatCompletionRequest(model='m', messages=[], temperature=2.1)
        req = ChatCompletionRequest(model='m', messages=[], temperature=1.5)
        assert req.temperature == 1.5

    def test_max_tokens_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            ChatCompletionRequest(model='m', messages=[], max_tokens=0)

    def test_top_p_range(self) -> None:
        with pytest.raises(ValidationError):
            ChatCompletionRequest(model='m', messages=[], top_p=1.5)

    def test_full_fields(self) -> None:
        req = ChatCompletionRequest(
            model='m',
            messages=[{'role': 'user', 'content': 'Hi'}],
            stream=True,
            stream_options=StreamOptions(include_usage=True),
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=-0.5,
        )
        assert req.stream is True
        assert req.temperature == 0.7
        assert req.max_tokens == 100

    def test_json_dump_excludes_none(self) -> None:
        req = ChatCompletionRequest(model='m', messages=[{'role': 'user', 'content': 'Hi'}])
        dumped = req.model_dump(mode='json', exclude_none=True)
        assert 'stream_options' not in dumped
        assert 'tools' not in dumped
        assert 'temperature' not in dumped
