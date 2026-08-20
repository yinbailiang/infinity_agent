"""OpenAI 配置模型与请求体构建测试。"""

import pytest
from pydantic import ValidationError

from infinity_agent.clients.open_ai.config import (
    OpenAIConfig,
    OpenAIConnectionConfig,
    OpenAIRequestConfig,
    build_chat_completion_request,
)
from infinity_agent.clients.open_ai.request_models import ResponseFormat
from infinity_agent.models import Message, ToolDefinition


class TestOpenAIConnectionConfig:
    """HTTP 连接配置默认值"""

    def test_defaults(self) -> None:
        cfg = OpenAIConnectionConfig()
        assert cfg.timeout == 60.0
        assert cfg.connect_timeout == 10.0
        assert cfg.max_retries == 3
        assert cfg.base_delay == 1.0
        assert cfg.max_delay == 30.0
        assert cfg.jitter is True
        assert cfg.retryable_status == frozenset({429, 500, 502, 503, 504})

    def test_custom_values(self) -> None:
        cfg = OpenAIConnectionConfig(timeout=5.0, max_retries=0, jitter=False)
        assert cfg.timeout == 5.0
        assert cfg.max_retries == 0
        assert cfg.jitter is False

    def test_invalid_timeout(self) -> None:
        with pytest.raises(ValidationError):
            OpenAIConnectionConfig(timeout=0)
        with pytest.raises(ValidationError):
            OpenAIConnectionConfig(max_retries=-1)


class TestOpenAIConfig:
    """客户端配置"""

    def test_defaults(self, openai_config: OpenAIConfig) -> None:
        assert openai_config.provider == 'openai'
        assert openai_config.model == 'gpt-4o-mini'
        assert openai_config.api_key == 'sk-test-key'
        assert openai_config.base_url == 'https://api.example.com/v1'
        assert isinstance(openai_config.connection, OpenAIConnectionConfig)

    def test_default_base_url(self) -> None:
        cfg = OpenAIConfig(model='m', api_key='k')
        assert cfg.base_url == 'https://api.openai.com/v1'

    def test_missing_model_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OpenAIConfig(api_key='k')

    def test_missing_api_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OpenAIConfig(model='m')

    def test_provider_locked_to_openai(self) -> None:
        with pytest.raises(ValidationError):
            OpenAIConfig(model='m', api_key='k', provider='azure')


class TestOpenAIRequestConfig:
    """请求级配置默认值"""

    def test_defaults(self) -> None:
        cfg = OpenAIRequestConfig()
        assert cfg.tools is None
        assert cfg.include_usage is True
        assert cfg.response_format is None

    def test_custom(self) -> None:
        tool = ToolDefinition.create('get_weather', '获取天气')
        cfg = OpenAIRequestConfig(
            tools=[tool],
            include_usage=False,
            response_format=ResponseFormat(type='json_object'),
        )
        assert cfg.tools == [tool]
        assert cfg.include_usage is False
        assert cfg.response_format is not None


class TestBuildChatCompletionRequest:
    """请求体构建"""

    def test_basic_payload(self, openai_config: OpenAIConfig) -> None:
        messages = [Message.system('You are helpful.'), Message.user('Hello')]
        payload = build_chat_completion_request(messages, openai_config)

        assert payload.model == 'gpt-4o-mini'
        assert payload.stream is True
        assert [m['role'] for m in payload.messages] == ['system', 'user']
        assert [m['content'] for m in payload.messages] == ['You are helpful.', 'Hello']

    def test_include_usage_sets_stream_options(self, openai_config: OpenAIConfig) -> None:
        messages = [Message.user('Hi')]
        payload = build_chat_completion_request(
            messages, openai_config, OpenAIRequestConfig(include_usage=True)
        )
        assert payload.stream_options is not None
        assert payload.stream_options.include_usage is True

    def test_disable_usage_no_stream_options(self, openai_config: OpenAIConfig) -> None:
        messages = [Message.user('Hi')]
        payload = build_chat_completion_request(
            messages, openai_config, OpenAIRequestConfig(include_usage=False)
        )
        assert payload.stream_options is None

    def test_tools_included(self, openai_config: OpenAIConfig) -> None:
        tool = ToolDefinition.create('get_weather', '获取天气')
        messages = [Message.user('Hi')]
        payload = build_chat_completion_request(
            messages, openai_config, OpenAIRequestConfig(tools=[tool])
        )
        assert payload.tools == [tool]

    def test_no_tools_omits_field(self, openai_config: OpenAIConfig) -> None:
        messages = [Message.user('Hi')]
        payload = build_chat_completion_request(messages, openai_config)
        assert payload.tools is None

    def test_response_format_included(self, openai_config: OpenAIConfig) -> None:
        fmt = ResponseFormat(type='json_object')
        messages = [Message.user('Hi')]
        payload = build_chat_completion_request(
            messages, openai_config, OpenAIRequestConfig(response_format=fmt)
        )
        assert payload.response_format == fmt

    def test_stream_false(self, openai_config: OpenAIConfig) -> None:
        messages = [Message.user('Hi')]
        payload = build_chat_completion_request(
            messages, openai_config, stream=False
        )
        assert payload.stream is False

    def test_default_request_config(self, openai_config: OpenAIConfig) -> None:
        messages = [Message.user('Hi')]
        payload = build_chat_completion_request(messages, openai_config)
        # 默认 include_usage=True → stream_options 存在
        assert payload.stream_options is not None

    def test_json_serializable(self, openai_config: OpenAIConfig) -> None:
        import json

        messages = [Message.user('Hi')]
        payload = build_chat_completion_request(messages, openai_config)
        dumped = payload.model_dump(mode='json', exclude_none=True)
        json.dumps(dumped)  # 不应抛异常
