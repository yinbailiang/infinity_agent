"""DeepSeek API 特化配置模型与客户端工厂"""

from typing import TYPE_CHECKING, Literal, Optional

from pydantic import Field

from ..base import LLMClient
from ..config import ClientConfig
from ..exceptions import LLMConfigError
from ..open_ai.config import OpenAIConfig, OpenAIRequestConfig
from ..open_ai.request_models import StreamOptions
from ..provider import register_client
from .request_models import DeepSeekChatCompletionRequest

if TYPE_CHECKING:
    from ...models.messages import Messages


class DeepSeekConfig(OpenAIConfig):
    """DeepSeek API 客户端配置（连接层面，整个客户端生命周期不变）

    DeepSeek API 与 OpenAI 兼容，因此直接复用 OpenAIConfig 的全部字段，
    仅覆盖 provider 标识与默认 base_url。
    """

    provider: Literal['deepseek'] = 'deepseek'  # pyright: ignore[reportIncompatibleVariableOverride]
    base_url: str = Field(default='https://api.deepseek.com', description='DeepSeek API 基础 URL')


class DeepSeekRequestConfig(OpenAIRequestConfig):
    """DeepSeek 请求级别配置（每次调用 stream_chat 时传入）"""

    reasoning_effort: Optional[Literal['low', 'medium', 'high']] = Field(
        default=None,
        description='思考强度（deepseek-reasoner / V3.1+ 支持）；设为 low/medium 时默认隐藏思考内容',
    )


def build_deepseek_request(
    messages: 'Messages',
    /,
    model_config: DeepSeekConfig,
    request_config: Optional[OpenAIRequestConfig] = None,
    *,
    stream: bool = True,
) -> DeepSeekChatCompletionRequest:
    """从 ``DeepSeekConfig`` + ``DeepSeekRequestConfig`` + 消息列表构建请求体。

    与 ``build_chat_completion_request`` 行为一致，额外透传
    ``reasoning_effort`` 等 DeepSeek 特有参数。

    :param messages: 对话消息列表（:class:`Message` 实例）
    :param model_config: 模型与连接配置
    :param request_config: 请求级别配置（工具、用量统计、响应格式、思考强度等）
    :param stream: 是否启用流式响应，默认 ``True``
    :return: 可直接序列化为 JSON 的请求体
    """
    req_config = request_config or DeepSeekRequestConfig()

    payload = DeepSeekChatCompletionRequest(
        model=model_config.model,
        messages=[m.model_dump(mode='json') for m in messages],
        stream=stream,
    )
    if req_config.tools:
        payload.tools = req_config.tools
    if req_config.include_usage:
        payload.stream_options = StreamOptions(include_usage=True)
    if req_config.response_format:
        payload.response_format = req_config.response_format
    if isinstance(req_config, DeepSeekRequestConfig) and req_config.reasoning_effort:
        payload.reasoning_effort = req_config.reasoning_effort
    return payload


@register_client('deepseek')
def create_deepseek_client(config: ClientConfig) -> LLMClient:
    """根据 DeepSeekConfig 创建 DeepSeekClient 实例。"""
    if not isinstance(config, DeepSeekConfig):
        raise LLMConfigError(f'deepseek factory received incompatible config type: {type(config).__name__}')

    from .client import DeepSeekClient

    return DeepSeekClient(config)
