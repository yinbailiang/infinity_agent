"""LLM 客户端配置模型与工厂"""

from pydantic import BaseModel, Field


class ClientConfig(BaseModel):
    """LLM 客户端通用配置基类"""

    provider: str = Field(description='服务提供商标识（如 openai）')

class RequestConfig(BaseModel):
    """LLM 请求配置基类"""