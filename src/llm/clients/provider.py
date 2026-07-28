"""LLM 客户端工厂注册表"""

from typing import Callable, Dict

from .base import LLMClient
from .config import ClientConfig

_ClientFactory = Callable[[ClientConfig], LLMClient]
_registry: Dict[str, _ClientFactory] = {}


def register_client(provider: str) -> Callable[[_ClientFactory], _ClientFactory]:
    """装饰器：将工厂函数注册到对应的 provider。

    用法::

        @register_client("openai")
        def _create_openai(config: OpenAIConfig) -> LLMClient:
            ...
    """

    def decorator(factory: _ClientFactory) -> _ClientFactory:
        _registry[provider] = factory
        return factory

    return decorator


def create_client(config: ClientConfig) -> LLMClient:
    """根据配置创建对应的 LLM 客户端实例。

    :param config: LLMConfig 子类实例
    :raises ValueError: 未注册的 provider
    """
    factory = _registry.get(config.provider)
    if factory is None:
        raise ValueError(f'Unknown LLM provider: {config.provider!r}. Available: {list(_registry.keys())}')
    return factory(config)
