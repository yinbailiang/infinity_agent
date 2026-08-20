"""工具定义模型测试：ToolDefinition、ToolParameters、ParameterProperty、Raw。"""

import pytest
from pydantic import ValidationError

from infinity_agent.models import (
    ParameterProperty,
    Raw,
    ToolDefinition,
    ToolFunction,
    ToolParameters,
)

# ============================================================================
# Raw 标记类型
# ============================================================================


class TestRaw:
    """Raw 返回值标记"""

    def test_raw_is_str(self) -> None:
        """Raw 在类型层面等价于 str"""
        def f() -> Raw:
            return 'plain text'
        assert f() == 'plain text'


# ============================================================================
# ParameterProperty 校验
# ============================================================================


class TestParameterPropertyValidation:
    """type 与 anyOf 互斥校验"""

    def test_type_alone_ok(self) -> None:
        prop = ParameterProperty(type='string')
        assert prop.type == 'string'
        assert prop.anyOf is None

    def test_anyof_alone_ok(self) -> None:
        prop = ParameterProperty(anyOf=[ParameterProperty(type='integer')])
        assert prop.anyOf is not None

    def test_both_type_and_anyof_rejected(self) -> None:
        with pytest.raises(ValidationError, match='互斥'):
            ParameterProperty(
                type='string',
                anyOf=[ParameterProperty(type='integer')],
            )

    def test_neither_type_nor_anyof_rejected(self) -> None:
        with pytest.raises(ValidationError, match='必须提供其一'):
            ParameterProperty()

    def test_object_requires_properties_or_additional(self) -> None:
        with pytest.raises(ValidationError, match='properties'):
            ParameterProperty(type='object')

    def test_object_with_properties_ok(self) -> None:
        prop = ParameterProperty(
            type='object',
            properties={'city': ParameterProperty(type='string')},
            required=['city'],
        )
        assert prop.properties is not None

    def test_object_with_additional_properties_ok(self) -> None:
        prop = ParameterProperty(type='object', additionalProperties=True)
        assert prop.additionalProperties is True

    def test_array_requires_items(self) -> None:
        with pytest.raises(ValidationError, match='items'):
            ParameterProperty(type='array')

    def test_array_with_items_ok(self) -> None:
        prop = ParameterProperty(
            type='array',
            items=ParameterProperty(type='string'),
        )
        assert prop.items is not None

    def test_required_must_exist_in_properties(self) -> None:
        with pytest.raises(ValidationError, match='required'):
            ParameterProperty(
                type='object',
                properties={'a': ParameterProperty(type='string')},
                required=['missing'],
            )


class TestParameterPropertyConstraints:
    """enum / default 类型一致性"""

    def test_enum_string_ok(self) -> None:
        prop = ParameterProperty(type='string', enum=['zh', 'en'])
        assert prop.enum == ['zh', 'en']

    def test_enum_incompatible_type_rejected(self) -> None:
        with pytest.raises(ValidationError, match='enum'):
            ParameterProperty(type='string', enum=['zh', 42])

    def test_enum_integer_ok(self) -> None:
        prop = ParameterProperty(type='integer', enum=[1, 2, 3])
        assert prop.enum == [1, 2, 3]

    def test_default_matches_type_ok(self) -> None:
        prop = ParameterProperty(type='string', default='zh')
        assert prop.default == 'zh'

    def test_default_incompatible_type_rejected(self) -> None:
        with pytest.raises(ValidationError, match='default'):
            ParameterProperty(type='string', default=42)

    def test_default_integer_with_number_type_ok(self) -> None:
        prop = ParameterProperty(type='number', default=3)
        assert prop.default == 3

    def test_default_matches_anyof_branch_ok(self) -> None:
        prop = ParameterProperty(
            anyOf=[ParameterProperty(type='integer'), ParameterProperty(type='string')],
            default=1,
        )
        assert prop.default == 1

    def test_default_matches_no_anyof_branch_rejected(self) -> None:
        with pytest.raises(ValidationError, match='default'):
            ParameterProperty(
                anyOf=[ParameterProperty(type='integer'), ParameterProperty(type='boolean')],
                default='str',
            )


# ============================================================================
# ToolParameters / ToolFunction / ToolDefinition
# ============================================================================


class TestToolParameters:
    """参数 Schema"""

    def test_type_fixed_to_object(self) -> None:
        params = ToolParameters()
        assert params.type == 'object'
        assert params.properties == {}
        assert params.required == []


class TestToolFunction:
    """工具函数定义"""

    def test_defaults(self) -> None:
        fn = ToolFunction(name='get_weather')
        assert fn.description == ''
        assert fn.parameters.properties == {}


class TestToolDefinition:
    """工具定义与工厂方法"""

    def test_create(self) -> None:
        tool = ToolDefinition.create('get_weather', '获取天气')
        assert tool.type == 'function'
        assert tool.function.name == 'get_weather'
        assert tool.function.description == '获取天气'

    def test_add_parameters(self) -> None:
        tool = ToolDefinition.create('get_weather', '获取天气')
        params = tool.function.parameters
        params.properties['city'] = ParameterProperty(type='string', description='城市')
        params.required.append('city')

        assert params.properties['city'].type == 'string'
        assert params.required == ['city']

    def test_model_dump_tool(self) -> None:
        tool = ToolDefinition.create('get_weather', '获取天气')
        tool.function.parameters.properties['city'] = ParameterProperty(
            type='string', description='城市'
        )
        tool.function.parameters.required.append('city')

        dumped = tool.model_dump_tool()
        assert dumped['type'] == 'function'
        assert dumped['function']['name'] == 'get_weather'
        assert dumped['function']['parameters']['type'] == 'object'
        assert 'city' in dumped['function']['parameters']['properties']
        assert dumped['function']['parameters']['required'] == ['city']
