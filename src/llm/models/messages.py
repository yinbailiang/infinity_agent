import json
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_serializer, model_validator


class ContentType(str, Enum):
    """支持的内容类型枚举"""

    TEXT = 'text'
    IMAGE_URL = 'image_url'


class ImageUrl(BaseModel):
    url: str
    detail: Optional[Literal['auto', 'low', 'high']] = None


class MultiModalContent(BaseModel):
    """多模态消息内容单元"""

    type: ContentType = Field(description='内容类型')
    text: Optional[str] = Field(default=None, description='文本内容')
    image_url: Optional[ImageUrl] = Field(default=None, description='图片/媒体 URL')

    @model_validator(mode='after')
    def validate_mutual_exclusion(self) -> 'MultiModalContent':
        """确保 text 和 image_url 互斥且与 type 匹配"""
        if self.type == ContentType.TEXT:
            if self.text is None:
                raise ValueError("text is required when type='text'")
            if self.image_url is not None:
                raise ValueError("image_url must be None when type='text'")
        elif self.type == ContentType.IMAGE_URL:
            if self.image_url is None:
                raise ValueError("image_url is required when type='image_url'")
            if self.text is not None:
                raise ValueError("text must be None when type='image_url'")
        return self


MessageContent = Union[str, List[MultiModalContent]]


class MessageRole(str, Enum):
    """消息角色枚举"""

    SYSTEM = 'system'
    USER = 'user'
    TOOL = 'tool'
    ASSISTANT = 'assistant'


class FunctionCall(BaseModel):
    """工具/函数调用定义"""

    name: str = Field(description='函数名称')
    arguments: Dict[str, Any] = Field(description='函数参数键值对')

    @field_serializer('arguments')
    def _serialize_arguments(self, value: Dict[str, Any]) -> str:
        """序列化为 JSON 字符串，供 LLM API 调用"""
        return json.dumps(value, ensure_ascii=False)


class ToolCall(BaseModel):
    """工具调用单元"""

    id: str = Field(description='工具调用唯一标识符')
    type: Literal['function'] = Field(default='function', description='调用类型，固定为 function')
    function: FunctionCall = Field(description='函数调用详情')


class Message(BaseModel):
    """LLM 消息格式"""

    role: MessageRole = Field(description='消息角色')
    name: Optional[str] = Field(default=None, description='可选的参与者名称')
    content: Optional[MessageContent] = Field(description='消息内容，支持文本或多模态混合')
    tool_calls: Optional[List[ToolCall]] = Field(default=None, description='工具调用列表（assistant 角色发起）')
    tool_call_id: Optional[str] = Field(default=None, description='工具调用 ID（tool 角色回传结果时使用）')

    @model_validator(mode='after')
    def validate_fields(self) -> 'Message':
        """校验 tool_calls / tool_call_id / content 与 role 的一致性"""
        if self.tool_calls is not None and self.role != MessageRole.ASSISTANT:
            raise ValueError('tool_calls 仅允许在 assistant 消息中使用')
        if self.tool_call_id is not None and self.role != MessageRole.TOOL:
            raise ValueError('tool_call_id 仅允许在 tool 消息中使用')
        if self.role == MessageRole.SYSTEM and not isinstance(self.content, str):
            raise ValueError('system 消息只支持纯文本内容')
        if self.role == MessageRole.ASSISTANT and isinstance(self.content, list):
            raise ValueError('assistant 消息的 content 不能是多模态数组')
        if self.role == MessageRole.TOOL and not isinstance(self.content, str):
            raise ValueError('tool 消息的 content 必须是字符串')
        if self.role == MessageRole.TOOL and self.tool_call_id is None:
            raise ValueError('tool 消息必须提供 tool_call_id')
        if self.role == MessageRole.USER and self.content is None:
            raise ValueError('user 消息必须有内容')
        return self

    # ---- 快捷构造器 ----

    @classmethod
    def user(cls, content: Union[str, List[MultiModalContent]]) -> 'Message':
        """创建 user 消息"""
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def system(cls, content: str) -> 'Message':
        """创建 system 消息"""
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def assistant(
        cls,
        content: Optional[str] = None,
        *,
        tool_calls: Optional[List[ToolCall]] = None,
    ) -> 'Message':
        """创建 assistant 消息"""
        return cls(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, content: str, *, tool_call_id: str) -> 'Message':
        """创建 tool 消息（工具执行结果）"""
        return cls(role=MessageRole.TOOL, content=content, tool_call_id=tool_call_id)


Messages = list[Message]
