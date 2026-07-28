"""OpenAI 兼容 API 配置模型"""

from typing import TYPE_CHECKING, List, Literal, Optional

from pydantic import BaseModel, Field

from ...models.tools import ToolDefinition
from ..base import LLMClient
from ..config import ClientConfig, RequestConfig
from ..exceptions import LLMConfigError
from ..provider import register_client
from .request_models import ChatCompletionRequest, ResponseFormat, StreamOptions

if TYPE_CHECKING:
    from ...models.messages import Messages


class OpenAIConnectionConfig(BaseModel):
    """HTTP 连接配置"""

    timeout: float = Field(default=60.0, gt=0, description='请求读超时（秒）')
    connect_timeout: float = Field(default=10.0, gt=0, description='连接超时（秒）')

    max_retries: int = Field(default=3, ge=0, description='最大重试次数（0 表示不重试）')
    base_delay: float = Field(default=1.0, gt=0, description='基础退避延迟（秒）')
    max_delay: float = Field(default=30.0, gt=0, description='退避延迟上限（秒）')
    jitter: bool = Field(default=True, description='是否启用 ±25% 随机抖动')
    retryable_status: frozenset[int] = Field(
        default=frozenset({429, 500, 502, 503, 504}),
        description='可触发重试的 HTTP 状态码集合',
    )


class OpenAIConfig(ClientConfig):
    """OpenAI 兼容 API 客户端配置（连接层面，整个客户端生命周期不变）"""

    provider: Literal['openai'] = 'openai' # pyright: ignore[reportIncompatibleVariableOverride]
    model: str = Field(description='模型名称')
    api_key: str = Field(description='API 密钥')
    base_url: str = Field(default='https://api.openai.com/v1', description='API 基础 URL')
    connection: OpenAIConnectionConfig = Field(default_factory=OpenAIConnectionConfig, description='HTTP 连接与重试配置')


class OpenAIRequestConfig(RequestConfig):
    """OpenAI 兼容 API 的请求级别配置（每次调用 stream_chat 时传入）"""

    tools: Optional[List[ToolDefinition]] = Field(default=None, description='工具定义列表')
    include_usage: bool = Field(default=True, description='是否请求 token 使用统计')
    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description='响应格式约束（JSON 模式 / 结构化输出）',
    )


def build_chat_completion_request(
    messages: 'Messages',
    /,
    model_config: OpenAIConfig,
    request_config: OpenAIRequestConfig | None = None,
    *,
    stream: bool = True,
) -> ChatCompletionRequest:
    """从 ``OpenAIConfig`` + ``OpenAIRequestConfig`` + 消息列表构建请求体。

    这是 ``_build_stream_payload`` 的独立可复用版本，
    方便在客户端之外（如测试、脚本、工具调用）直接构造请求模型。

    :param messages: 对话消息列表（:class:`Message` 实例）
    :param model_config: 模型与连接配置
    :param request_config: 请求级别配置（工具、用量统计、响应格式等）
    :param stream: 是否启用流式响应，默认 ``True``
    :return: 可直接序列化为 JSON 的请求体
    """
    req_config = request_config or OpenAIRequestConfig()

    payload = ChatCompletionRequest(
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
    return payload


@register_client('openai')
def create_openai_client(config: ClientConfig) -> LLMClient:
    """根据 OpenAIConfig 创建 OpenAIClient 实例。"""
    if not isinstance(config, OpenAIConfig):
        raise LLMConfigError(f'openai factory received incompatible config type: {type(config).__name__}')

    from .client import OpenAIClient

    return OpenAIClient(config)
