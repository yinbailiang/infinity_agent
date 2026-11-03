import json
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union, cast

from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    model_serializer,
    model_validator,
)


class ContentType(str, Enum):
    """支持的内容类型枚举"""

    TEXT = 'text'
    IMAGE_URL = 'image_url'
    FILE = 'file'


class ImageUrl(BaseModel):
    url: str
    detail: Optional[Literal['auto', 'low', 'high', 'original']] = None


class FilePart(BaseModel):
    """文件内容单元：Files API 引用或 base64 内联，二者互斥"""

    file_id: Optional[str] = Field(default=None, description='通过 Files API 上传的文件 ID，形如 file-api-...')
    file_data: Optional[str] = Field(default=None, description='base64 data URL 形式的内联文件')
    filename: Optional[str] = Field(default=None, description='文件名，仅在配合 file_data 时有效')

    @model_validator(mode='after')
    def validate_source(self) -> 'FilePart':
        """``file_id`` 与 ``file_data`` 必须二选一（两者都传或都不传都会被服务端拒收）"""
        if (self.file_id is None) == (self.file_data is None):
            raise ValueError('file_id 与 file_data 必须二选一')
        return self

    @model_serializer
    def serialize_compact(self) -> Dict[str, Any]:
        """省略空字段，保持与 ``type`` 平级的扁平结构"""
        data: Dict[str, Any] = {}
        if self.file_id is not None:
            data['file_id'] = self.file_id
        if self.file_data is not None:
            data['file_data'] = self.file_data
        if self.filename is not None:
            data['filename'] = self.filename
        return data


class MultiModalContent(BaseModel):
    """多模态消息内容单元（user 消息与 tool 结果均可使用）"""

    type: ContentType = Field(description='内容类型')
    text: Optional[str] = Field(default=None, description='文本内容')
    image_url: Optional[ImageUrl] = Field(default=None, description='图片/媒体 URL')
    file: Optional[FilePart] = Field(default=None, description='文件内容（Files API 引用或 base64 内联）')

    @model_validator(mode='before')
    @classmethod
    def pack_file_fields(cls, data: object) -> object:
        """把与 ``type`` 平级的 ``file_id`` / ``file_data`` / ``filename`` 收进 ``file``。

        服务端的 ``file`` 内容块是扁平的（``{"type": "file", "file_id": ...}``），
        这里做一次打包，保证 ``model_dump`` 的输出能被 ``model_validate`` 原样读回。
        """
        if not isinstance(data, dict):
            return data
        payload = cast(Dict[str, Any], data)
        if payload.get('type') != ContentType.FILE.value or 'file' in payload:
            return payload
        packed = {key: payload[key] for key in ('file_id', 'file_data', 'filename') if key in payload}
        if not packed:
            return payload
        rest = {key: value for key, value in payload.items() if key not in packed}
        rest['file'] = packed
        return rest

    @model_validator(mode='after')
    def validate_mutual_exclusion(self) -> 'MultiModalContent':
        """确保只有一个载荷字段，且与 type 匹配"""
        payloads: Dict[str, object] = {
            'text': self.text,
            'image_url': self.image_url,
            'file': self.file,
        }
        field = self.type.value
        if payloads[field] is None:
            raise ValueError(f'{field} is required when type={field!r}')
        extras = sorted(name for name, value in payloads.items() if name != field and value is not None)
        if extras:
            raise ValueError(f'{extras} must be None when type={field!r}')
        return self

    @classmethod
    def text_part(cls, text: str) -> 'MultiModalContent':
        """构造文本内容单元"""
        return cls(type=ContentType.TEXT, text=text)

    @classmethod
    def image_part(
        cls,
        url: str,
        *,
        detail: Optional[Literal['auto', 'low', 'high', 'original']] = None,
    ) -> 'MultiModalContent':
        """构造图片内容单元

        :param url: 图片地址，支持 http(s) 链接与 ``data:image/...;base64,`` 内联
        :param detail: 细节档位 ``auto`` / ``low`` / ``high`` / ``original``；``None`` 表示交给服务端
        """
        return cls(type=ContentType.IMAGE_URL, image_url=ImageUrl(url=url, detail=detail))

    @classmethod
    def file_part(
        cls,
        *,
        file_id: Optional[str] = None,
        file_data: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> 'MultiModalContent':
        """构造文件内容单元

        :param file_id: Files API 返回的文件 ID（形如 ``file-api-...``）
        :param file_data: base64 data URL 形式的内联文件
        :param filename: 可选文件名，仅在配合 ``file_data`` 时有效

        :raises ValueError: ``file_id`` 与 ``file_data`` 未严格二选一
        """
        return cls(
            type=ContentType.FILE,
            file=FilePart(file_id=file_id, file_data=file_data, filename=filename),
        )

    @model_serializer
    def serialize_compact(self) -> Dict[str, Any]:
        """只输出与 ``type`` 相关的字段，避免把 ``None`` 塞进请求体。

        多模态消息会原样进入模型请求体，``{"image_url": null}`` 这类空字段会被
        严格的 provider 拒收，因此这里按需省略；``file`` 的字段与 ``type`` 平级。
        """
        data: Dict[str, Any] = {'type': self.type.value}
        if self.text is not None:
            data['text'] = self.text
        if self.image_url is not None:
            data['image_url'] = self.image_url.model_dump(mode='json', exclude_none=True)
        if self.file is not None:
            data.update(self.file.model_dump(mode='json'))
        return data


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
    content: Optional[MessageContent] = Field(
        description='消息内容，支持文本或多模态混合（tool 消息亦可携带多模态数组）'
    )
    reasoning_content: Optional[str] = Field(
        default=None,
        description='思考（推理）内容，仅供 assistant 消息使用，多轮对话时回传给模型',
    )
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
        if self.role == MessageRole.TOOL and self.content is None:
            raise ValueError('tool 消息必须提供内容（文本或多模态数组）')
        if self.role == MessageRole.TOOL and self.tool_call_id is None:
            raise ValueError('tool 消息必须提供 tool_call_id')
        if self.role == MessageRole.USER and self.content is None:
            raise ValueError('user 消息必须有内容')
        if self.reasoning_content is not None and self.role != MessageRole.ASSISTANT:
            raise ValueError('reasoning_content 仅允许在 assistant 消息中使用')
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
        reasoning_content: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None,
    ) -> 'Message':
        """创建 assistant 消息（可选携带思考内容，供 DeepSeek 等多轮回传）"""
        return cls(
            role=MessageRole.ASSISTANT,
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
        )

    @classmethod
    def tool(
        cls,
        content: Union[str, List[MultiModalContent]],
        *,
        tool_call_id: str,
    ) -> 'Message':
        """创建 tool 消息（工具执行结果，可为文本或多模态数组）"""
        return cls(role=MessageRole.TOOL, content=content, tool_call_id=tool_call_id)


Messages = list[Message]
