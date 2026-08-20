"""客户端工厂注册表测试：register_client / create_client。"""

import pytest

from infinity_agent.clients.base import LLMClient
from infinity_agent.clients.config import ClientConfig
from infinity_agent.clients.provider import create_client, register_client


class _DummyConfig(ClientConfig):
    provider: str = 'dummy'


class _DummyClient(LLMClient):
    """最小 LLMClient 实现，用于注册表测试。"""

    def __init__(self, config: ClientConfig) -> None:
        self.config = config

    async def __aenter__(self) -> '_DummyClient':
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    async def stream_chat(self, messages, config=None):
        if False:
            yield  # pragma: no cover - 仅为满足 async generator 语法
        return


@pytest.fixture
def registered_dummy() -> str:
    """注册 dummy provider，返回其名称以便清理。"""
    provider_name = 'dummy_test_provider'

    @register_client(provider_name)
    def _factory(config: ClientConfig) -> LLMClient:
        return _DummyClient(config)

    yield provider_name


class TestRegisterClient:
    """装饰器注册行为"""

    def test_decorator_returns_original_factory(self, registered_dummy: str) -> None:
        """装饰器应返回原函数（而非包装）"""
        @register_client('dummy_returns_original')
        def factory(config: ClientConfig) -> LLMClient:
            return _DummyClient(config)

        assert callable(factory)
        result = create_client(_DummyConfig(provider='dummy_returns_original'))
        assert isinstance(result, _DummyClient)


class TestCreateClient:
    """create_client 工厂调度"""

    def test_create_known_provider(self, registered_dummy: str) -> None:
        client = create_client(_DummyConfig(provider=registered_dummy))
        assert isinstance(client, _DummyClient)

    def test_create_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match='Unknown LLM provider'):
            create_client(_DummyConfig(provider='no_such_provider'))

    def test_unknown_provider_error_lists_available(self, registered_dummy: str) -> None:
        with pytest.raises(ValueError) as exc_info:
            create_client(_DummyConfig(provider='no_such_provider'))
        assert registered_dummy in str(exc_info.value)


class TestOpenAIProviderIntegration:
    """真实 openai provider 的工厂集成"""

    def test_create_openai_client(self, openai_config) -> None:
        from infinity_agent.clients.open_ai import OpenAIClient, create_openai_client

        client = create_openai_client(openai_config)
        assert isinstance(client, OpenAIClient)
        assert client.model == openai_config.model

    def test_create_openai_client_wrong_config_type(self) -> None:
        from infinity_agent.clients.exceptions import LLMConfigError
        from infinity_agent.clients.open_ai import create_openai_client

        with pytest.raises(LLMConfigError, match='incompatible config type'):
            create_openai_client(_DummyConfig())

    def test_create_client_via_openai_provider(self, openai_config) -> None:
        from infinity_agent.clients.open_ai import OpenAIClient

        client = create_client(openai_config)
        assert isinstance(client, OpenAIClient)
