"""OpenAI 兼容 API 的请求模型定义"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_serializer

from ...models.tools import ToolDefinitions


class JsonSchema(BaseModel):
    """JSON Schema 模式 — 结构化输出的 schema 定义

    用法::

        schema = JsonSchema(name='tick_output', schema=TickOutput)
    """

    name: str = Field(description='schema 名称，模型可见')
    schema_: type[BaseModel] = Field(alias='schema', description='Pydantic 模型类，自动推导 JSON Schema')
    strict: bool = Field(default=True, description='是否启用严格模式')

    @field_serializer('schema_')
    def _serialize_schema(self, model: type[BaseModel], _info: Any) -> dict[str, Any]:
        return model.model_json_schema()


class ResponseFormat(BaseModel):
    """响应格式约束

    用法::

        # JSON 模式（模型自由输出 JSON）
        ResponseFormat(type='json_object')

        # 结构化输出（传入 Pydantic 模型，自动推导 schema）
        ResponseFormat(
            type='json_schema',
            json_schema=JsonSchema(name='tick_output', schema=TickOutput),
        )
    """

    type: Literal['json_object', 'json_schema'] = Field(default='json_object')
    json_schema: JsonSchema | None = Field(default=None)


class StreamOptions(BaseModel):
    """流式选项，作为 stream_options 嵌套在请求体中"""

    include_usage: bool = Field(default=False, description='是否在流式 chunk 中返回 token 使用统计')


class ChatCompletionRequest(BaseModel):
    """OpenAI 兼容的 Chat Completion 请求体"""

    model: str = Field(description='模型名称')
    messages: List[Dict[str, Any]] = Field(description='消息列表')
    stream: bool = Field(default=False, description='是否启用流式响应')
    stream_options: Optional[StreamOptions] = Field(default=None, description='流式选项')
    tools: Optional[ToolDefinitions] = Field(default=None, description='工具定义列表')
    response_format: Optional[ResponseFormat] = Field(default=None, description='响应格式约束')
    temperature: Optional[float] = Field(default=None, ge=0, le=2, description='采样温度')
    max_tokens: Optional[int] = Field(default=None, gt=0, description='最大生成 token 数')
    top_p: Optional[float] = Field(default=None, ge=0, le=1, description='核采样概率阈值')
    frequency_penalty: Optional[float] = Field(default=None, ge=-2, le=2, description='频率惩罚')
    presence_penalty: Optional[float] = Field(default=None, ge=-2, le=2, description='存在惩罚')