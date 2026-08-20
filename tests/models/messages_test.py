"""消息模型测试：Message、MessageRole、MultiModalContent、ToolCall 等。"""

import pytest
from pydantic import ValidationError

from infinity_agent.models import (
    ContentType,
    FunctionCall,
    ImageUrl,
    Message,
    MessageRole,
    MultiModalContent,
    ToolCall,
)

# ============================================================================
# 枚举
# ============================================================================


class TestEnums:
    """角色与内容类型枚举"""

    def test_message_role_values(self) -> None:
        assert MessageRole.SYSTEM.value == 'system'
        assert MessageRole.USER.value == 'user'
        assert MessageRole.TOOL.value == 'tool'
        assert MessageRole.ASSISTANT.value == 'assistant'

    def test_content_type_values(self) -> None:
        assert ContentType.TEXT.value == 'text'
        assert ContentType.IMAGE_URL.value == 'image_url'


# ============================================================================
# MultiModalContent 校验
# ============================================================================


class TestMultiModalContent:
    """多模态内容互斥校验"""

    def test_text_content(self) -> None:
        mm = MultiModalContent(type=ContentType.TEXT, text='hello')
        assert mm.text == 'hello'
        assert mm.image_url is None

    def test_image_content(self) -> None:
        mm = MultiModalContent(
            type=ContentType.IMAGE_URL,
            image_url=ImageUrl(url='https://example.com/a.png'),
        )
        assert mm.image_url is not None
        assert mm.image_url.url == 'https://example.com/a.png'

    def test_text_requires_text(self) -> None:
        with pytest.raises(ValidationError):
            MultiModalContent(type=ContentType.TEXT)

    def test_text_forbids_image(self) -> None:
        with pytest.raises(ValidationError):
            MultiModalContent(
                type=ContentType.TEXT,
                text='hello',
                image_url=ImageUrl(url='https://example.com/a.png'),
            )

    def test_image_requires_image(self) -> None:
        with pytest.raises(ValidationError):
            MultiModalContent(type=ContentType.IMAGE_URL)

    def test_image_forbids_text(self) -> None:
        with pytest.raises(ValidationError):
            MultiModalContent(
                type=ContentType.IMAGE_URL,
                text='hello',
                image_url=ImageUrl(url='https://example.com/a.png'),
            )

    def test_image_url_detail(self) -> None:
        img = ImageUrl(url='https://example.com/a.png', detail='low')
        assert img.detail == 'low'


# ============================================================================
# FunctionCall / ToolCall
# ============================================================================


class TestFunctionCall:
    """函数调用序列化"""

    def test_arguments_serialized_to_json_string(self) -> None:
        fc = FunctionCall(name='get_weather', arguments={'city': '北京', 'unit': 'c'})
        assert fc.name == 'get_weather'
        assert fc.arguments == {'city': '北京', 'unit': 'c'}
        dumped = fc.model_dump(mode='json')
        assert dumped['arguments'] == '{"city": "北京", "unit": "c"}'

    def test_arguments_empty(self) -> None:
        fc = FunctionCall(name='noop', arguments={})
        assert fc.model_dump(mode='json')['arguments'] == '{}'


class TestToolCall:
    """工具调用单元"""

    def test_type_fixed_to_function(self) -> None:
        tc = ToolCall(
            id='call_1',
            function={'name': 'f', 'arguments': {}},
        )
        assert tc.type == 'function'

    def test_full_tool_call(self) -> None:
        tc = ToolCall(
            id='call_1',
            function=FunctionCall(name='f', arguments={'x': 1}),
        )
        assert tc.id == 'call_1'
        assert tc.function.name == 'f'
        assert tc.function.arguments == {'x': 1}


# ============================================================================
# Message 校验
# ============================================================================


class TestMessageValidation:
    """Message 字段与角色一致性校验"""

    def test_user_message(self) -> None:
        msg = Message.user('Hello')
        assert msg.role == MessageRole.USER
        assert msg.content == 'Hello'

    def test_user_multimodal_content(self) -> None:
        mm = MultiModalContent(type=ContentType.TEXT, text='look')
        msg = Message.user([mm])
        assert msg.content == [mm]

    def test_system_message(self) -> None:
        msg = Message.system('Be concise.')
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == 'Be concise.'

    def test_assistant_message(self) -> None:
        msg = Message.assistant('Sure!')
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == 'Sure!'

    def test_assistant_with_tool_calls(self) -> None:
        tc = ToolCall(id='c1', function={'name': 'f', 'arguments': {}})
        msg = Message.assistant(content=None, tool_calls=[tc])
        assert msg.tool_calls == [tc]

    def test_tool_message(self) -> None:
        msg = Message.tool('result', tool_call_id='c1')
        assert msg.role == MessageRole.TOOL
        assert msg.tool_call_id == 'c1'

    def test_tool_calls_only_for_assistant(self) -> None:
        tc = ToolCall(id='c1', function={'name': 'f', 'arguments': {}})
        with pytest.raises(ValidationError, match='tool_calls'):
            Message(role=MessageRole.USER, content='hi', tool_calls=[tc])

    def test_tool_call_id_only_for_tool(self) -> None:
        with pytest.raises(ValidationError, match='tool_call_id'):
            Message(role=MessageRole.USER, content='hi', tool_call_id='c1')

    def test_system_requires_text_content(self) -> None:
        with pytest.raises(ValidationError, match='纯文本'):
            Message(role=MessageRole.SYSTEM, content=None)

    def test_assistant_content_cannot_be_multimodal(self) -> None:
        mm = MultiModalContent(type=ContentType.TEXT, text='hi')
        with pytest.raises(ValidationError, match='多模态'):
            Message(role=MessageRole.ASSISTANT, content=[mm])

    def test_tool_content_must_be_string(self) -> None:
        # content 字段校验（Union[str, list]）会先于角色一致性校验失败
        with pytest.raises(ValidationError):
            Message(role=MessageRole.TOOL, content=123, tool_call_id='c1')

    def test_tool_requires_tool_call_id(self) -> None:
        with pytest.raises(ValidationError, match='tool_call_id'):
            Message(role=MessageRole.TOOL, content='result')

    def test_user_requires_content(self) -> None:
        with pytest.raises(ValidationError, match='必须有内容'):
            Message(role=MessageRole.USER, content=None)


class TestMessageSerialization:
    """Message 序列化行为"""

    def test_tool_call_arguments_as_string_in_json(self) -> None:
        msg = Message.assistant(
            content=None,
            tool_calls=[
                ToolCall(
                    id='c1',
                    function={'name': 'get_weather', 'arguments': {'city': '北京'}},
                )
            ],
        )
        dumped = msg.model_dump(mode='json')
        assert dumped['tool_calls'][0]['function']['arguments'] == '{"city": "北京"}'

    def test_role_serialized_to_value(self) -> None:
        msg = Message.user('hi')
        assert msg.model_dump(mode='json')['role'] == 'user'
