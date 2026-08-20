"""JSON Schema → ParameterProperty 转换测试。"""

from infinity_agent.tools.introspectionist.convert import (
    prop_from_schema,
    resolve_refs,
)


class TestResolveRefs:
    """$ref 引用内联解析"""

    def test_no_defs_returns_unchanged(self) -> None:
        schema = {'type': 'object', 'properties': {'a': {'type': 'string'}}}
        assert resolve_refs(schema) == schema

    def test_simple_ref(self) -> None:
        schema = {
            'type': 'object',
            '$defs': {'User': {'type': 'object', 'properties': {'name': {'type': 'string'}}}},
            'properties': {'user': {'$ref': '#/$defs/User'}},
        }
        resolved = resolve_refs(schema)
        assert resolved['properties']['user']['type'] == 'object'
        assert 'name' in resolved['properties']['user']['properties']
        assert '$ref' not in resolved['properties']['user']

    def test_nested_ref(self) -> None:
        schema = {
            'type': 'object',
            '$defs': {
                'Address': {'type': 'object', 'properties': {'city': {'type': 'string'}}},
                'User': {
                    'type': 'object',
                    'properties': {'address': {'$ref': '#/$defs/Address'}},
                },
            },
            'properties': {'user': {'$ref': '#/$defs/User'}},
        }
        resolved = resolve_refs(schema)
        address = resolved['properties']['user']['properties']['address']
        assert address['properties']['city']['type'] == 'string'

    def test_circular_ref_truncated(self) -> None:
        schema = {
            'type': 'object',
            '$defs': {
                'Node': {
                    'type': 'object',
                    'properties': {'child': {'$ref': '#/$defs/Node'}},
                }
            },
            'properties': {'root': {'$ref': '#/$defs/Node'}},
        }
        resolved = resolve_refs(schema)
        node = resolved['properties']['root']
        assert node['type'] == 'object'
        assert '循环引用截断' in node['properties']['child'].get('description', '')

    def test_deep_ref_truncated(self) -> None:
        """超过最大深度时截断（使用不同名称的引用链避免触发循环截断）"""
        schema = {
            'type': 'object',
            '$defs': {
                'A': {'type': 'object', 'properties': {'next': {'$ref': '#/$defs/B'}}},
                'B': {'type': 'object', 'properties': {'next': {'$ref': '#/$defs/C'}}},
                'C': {'type': 'object', 'properties': {'next': {'$ref': '#/$defs/D'}}},
                'D': {'type': 'object', 'properties': {'x': {'type': 'string'}}},
            },
            'properties': {'root': {'$ref': '#/$defs/A'}},
        }
        resolved = resolve_refs(schema, max_depth=2)
        node = resolved['properties']['root']
        depth1 = node['properties']['next']  # B
        depth2 = depth1['properties']['next']  # C（深度截断处）
        assert '深度截断' in depth2.get('description', '')
        assert depth2['type'] == 'object'


class TestPropFromSchema:
    """Schema 节点 → ParameterProperty"""

    def test_string(self) -> None:
        prop = prop_from_schema({'type': 'string', 'description': '名称'})
        assert prop.type == 'string'
        assert prop.description == '名称'

    def test_string_with_enum(self) -> None:
        prop = prop_from_schema({'type': 'string', 'enum': ['zh', 'en']})
        assert prop.enum == ['zh', 'en']

    def test_string_with_default(self) -> None:
        prop = prop_from_schema({'type': 'string', 'default': 'zh'})
        assert prop.default == 'zh'

    def test_optional_becomes_nullable(self) -> None:
        prop = prop_from_schema(
            {'anyOf': [{'type': 'integer'}, {'type': 'null'}]}
        )
        assert prop.type == 'integer'
        assert prop.nullable is True

    def test_union_becomes_anyof(self) -> None:
        prop = prop_from_schema(
            {'anyOf': [{'type': 'integer'}, {'type': 'string'}, {'type': 'null'}]}
        )
        assert prop.anyOf is not None
        assert [p.type for p in prop.anyOf] == ['integer', 'string']
        assert prop.nullable is True

    def test_nested_object(self) -> None:
        prop = prop_from_schema(
            {
                'type': 'object',
                'properties': {'city': {'type': 'string'}},
                'required': ['city'],
            }
        )
        assert prop.type == 'object'
        assert 'city' in prop.properties
        assert prop.required == ['city']

    def test_array_with_items(self) -> None:
        prop = prop_from_schema(
            {'type': 'array', 'items': {'type': 'integer'}}
        )
        assert prop.type == 'array'
        assert prop.items is not None
        assert prop.items.type == 'integer'

    def test_dict_str_to_str(self) -> None:
        prop = prop_from_schema(
            {
                'type': 'object',
                'additionalProperties': {'type': 'string'},
            }
        )
        assert prop.type == 'object'
        assert prop.additionalProperties is not None
        assert prop.additionalProperties.type == 'string'

    def test_bare_object_allows_any(self) -> None:
        prop = prop_from_schema({'type': 'object'})
        assert prop.type == 'object'
        assert prop.additionalProperties is True

    def test_numeric_constraints(self) -> None:
        prop = prop_from_schema(
            {'type': 'number', 'minimum': 0, 'maximum': 100}
        )
        assert prop.minimum == 0
        assert prop.maximum == 100

    def test_string_constraints(self) -> None:
        prop = prop_from_schema(
            {'type': 'string', 'minLength': 1, 'maxLength': 10, 'pattern': r'\d+'}
        )
        assert prop.minLength == 1
        assert prop.maxLength == 10
        assert prop.pattern == r'\d+'

    def test_composite_default_skipped(self) -> None:
        """dict/list 等复合默认值不写入 ParameterProperty.default"""
        prop = prop_from_schema({'type': 'array', 'items': {'type': 'string'}, 'default': ['a']})
        assert prop.default is None

    def test_empty_schema_defaults_to_string(self) -> None:
        """无 type 且无 anyOf 时回退为 string"""
        prop = prop_from_schema({})
        assert prop.type == 'string'
