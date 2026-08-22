"""DeepSeek 配置模型、请求体构建与 provider 工厂测试。"""

import pytest
from pydantic import ValidationError

from infinity_agent.clients.config import ClientConfig
from infinity_agent.clients.deepseek import (
    DeepSeekClient,
    DeepSeekConfig,
    DeepSeekRequestConfig,
    build_deepseek_request,
    create_deepseek_client,
)
from infinity_agent.clients.exceptions import LLMConfigError
from infinity_agent.clients.open_ai.config import OpenAIConnectionConfig
from infinity_agent.clients.provider import create_client
from infinity_agent.models import Message


class TestDeepSeekConfig:
    """DeepSeek 客户端配置"""

    def test_defaults(self) -> None:
        cfg = DeepSeekConfig(model='deepseek-reasoner', api_key='sk-test')
        assert cfg.provider == 'deepseek'
        assert cfg.model == 'deepseek-reasoner'
        assert cfg.base_url == 'https://api.deepseek.com'
        assert isinstance(cfg.connection, OpenAIConnectionConfig)

    def test_custom_base_url(self) -> None:
        cfg = DeepSeekConfig(model='m', api_key='k', base_url='https://api.deepseek.com/v1')
        assert cfg.base_url == 'https://api.deepseek.com/v1'

    def test_provider_locked_to_deepseek(self) -> None:
        with pytest.raises(ValidationError):
            DeepSeekConfig(model='m', api_key='k', provider='openai')

    def test_missing_model_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DeepSeekConfig(api_key='k')


class TestDeepSeekRequestConfig:
    """DeepSeek 请求级配置"""

    def test_defaults(self) -> None:
        cfg = DeepSeekRequestConfig()
        assert cfg.reasoning_effort is None
        assert cfg.include_usage is True

    def test_custom_reasoning_effort(self) -> None:
        cfg = DeepSeekRequestConfig(reasoning_effort='high')
        assert cfg.reasoning_effort == 'high'

    def test_invalid_reasoning_effort(self) -> None:
        with pytest.raises(ValidationError):
            DeepSeekRequestConfig(reasoning_effort='ultra')


class TestBuildDeepSeekRequest:
    """DeepSeek 请求体构建"""

    def test_basic_payload(self) -> None:
        cfg = DeepSeekConfig(model='deepseek-reasoner', api_key='k')
        payload = build_deepseek_request([Message.user('Hi')], cfg)

        assert payload.model == 'deepseek-reasoner'
        assert payload.stream is True
        assert payload.reasoning_effort is None

    def test_reasoning_effort_forwarded(self) -> None:
        cfg = DeepSeekConfig(model='deepseek-reasoner', api_key='k')
        payload = build_deepseek_request(
            [Message.user('Hi')], cfg, DeepSeekRequestConfig(reasoning_effort='low')
        )
        assert payload.reasoning_effort == 'low'

    def test_include_usage_sets_stream_options(self) -> None:
        cfg = DeepSeekConfig(model='m', api_key='k')
        payload = build_deepseek_request(
            [Message.user('Hi')], cfg, DeepSeekRequestConfig(include_usage=True)
        )
        assert payload.stream_options is not None
        assert payload.stream_options.include_usage is True

    def test_assistant_reasoning_content_in_payload(self) -> None:
        """多轮对话：assistant 的思考内容应序列化进请求体以便回传"""
        cfg = DeepSeekConfig(model='m', api_key='k')
        messages = [
            Message.user('1+1=?'),
            Message.assistant('2', reasoning_content='让我算一下'),
        ]
        payload = build_deepseek_request(messages, cfg)
        assert payload.messages[1]['reasoning_content'] == '让我算一下'


class TestDeepSeekProvider:
    """deepseek provider 工厂集成"""

    def test_create_deepseek_client(self) -> None:
        cfg = DeepSeekConfig(model='deepseek-reasoner', api_key='k')
        client = create_deepseek_client(cfg)
        assert isinstance(client, DeepSeekClient)
        assert client.model == 'deepseek-reasoner'

    def test_create_deepseek_client_wrong_config_type(self) -> None:
        class _Other(ClientConfig):
            provider: str = 'x'

        with pytest.raises(LLMConfigError, match='incompatible config type'):
            create_deepseek_client(_Other())

    def test_create_client_via_deepseek_provider(self) -> None:
        cfg = DeepSeekConfig(model='deepseek-reasoner', api_key='k')
        client = create_client(cfg)
        assert isinstance(client, DeepSeekClient)
