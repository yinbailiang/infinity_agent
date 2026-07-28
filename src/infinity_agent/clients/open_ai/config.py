"""OpenAI 兼容 API 配置模型"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from ...models.tools import ToolDefinition
from ..base import LLMClient
from ..config import ClientConfig, RequestConfig
from ..exceptions import LLMConfigError
from ..provider import register_client
from .request_models import ResponseFormat


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


@register_client('openai')
def create_openai_client(config: ClientConfig) -> LLMClient:
    """根据 OpenAIConfig 创建 OpenAIClient 实例。"""
    if not isinstance(config, OpenAIConfig):
        raise LLMConfigError(f'openai factory received incompatible config type: {type(config).__name__}')

    from .client import OpenAIClient

    return OpenAIClient(config)
