"""类型签名推断与动态模型构建测试。"""

import inspect
import typing
from typing import Annotated, Dict, List, Literal, Optional, Tuple

import pytest
from pydantic import BaseModel

from infinity_agent.models.tools import Raw
from infinity_agent.tools.introspectionist.introspection import build_tool_definition
from infinity_agent.tools.introspectionist.model_builder import (
    build_param_model,
    build_return_model,
    flatten_annotated,
    is_raw_return,
    normalize_annotated,
    unwrap_annotated,
)


class TestBuildToolDefinition:
    """从函数签名构建 ToolDefinition"""

    def test_basic_params_and_required(self) -> None:
        def get_weather(city: str, unit: str = 'c') -> str:
            """获取天气"""
            return 'x'

        result = build_tool_definition(get_weather)
        assert result.definition.function.name == 'get_weather'
        assert result.definition.function.description == '获取天气'
        props = result.definition.function.parameters.properties
        assert set(props.keys()) == {'city', 'unit'}
        assert result.definition.function.parameters.required == ['city']
        assert props['unit'].default == 'c'

    def test_optional_param(self) -> None:
        def f(x: Optional[int] = None) -> str:
            return 'x'

        result = build_tool_definition(f)
        prop = result.definition.function.parameters.properties['x']
        assert prop.nullable is True
        assert prop.type == 'integer'

    def test_literal_becomes_enum(self) -> None:
        def f(mode: Literal['fast', 'slow']) -> str:
            return 'x'

        result = build_tool_definition(f)
        prop = result.definition.function.parameters.properties['mode']
        assert prop.enum == ['fast', 'slow']

    def test_annotated_description(self) -> None:
        def f(city: Annotated[str, '城市名称']) -> str:
            return 'x'

        result = build_tool_definition(f)
        prop = result.definition.function.parameters.properties['city']
        assert prop.description == '城市名称'

    def test_list_param(self) -> None:
        def f(items: List[int]) -> str:
            return 'x'

        result = build_tool_definition(f)
        prop = result.definition.function.parameters.properties['items']
        assert prop.type == 'array'
        assert prop.items is not None
        assert prop.items.type == 'integer'

    def test_dict_param(self) -> None:
        def f(env: Dict[str, str]) -> str:
            return 'x'

        result = build_tool_definition(f)
        prop = result.definition.function.parameters.properties['env']
        assert prop.type == 'object'
        assert prop.additionalProperties is not None

    def test_missing_type_annotation_raises(self) -> None:
        def f(x):  # type: ignore[no-untyped-def]
            return 'x'

        with pytest.raises(TypeError, match='类型标注'):
            build_tool_definition(f)

    def test_missing_return_annotation_raises(self) -> None:
        def f(x: int):  # type: ignore[no-untyped-def]
            return x

        with pytest.raises(TypeError, match='返回类型'):
            build_tool_definition(f)

    def test_name_override(self) -> None:
        def f(x: int) -> str:
            return 'x'

        result = build_tool_definition(f, name='renamed', description='自定义描述')
        assert result.definition.function.name == 'renamed'
        assert result.definition.function.description == '自定义描述'

    def test_no_params(self) -> None:
        def f() -> str:
            return 'x'

        result = build_tool_definition(f)
        assert result.param_model.model is None
        assert result.definition.function.parameters.properties == {}


class TestBuildParamModel:
    """参数校验模型构建"""

    def test_positional_only(self) -> None:
        def f(a: int, /, b: int = 0) -> str:
            return 'x'

        sig = inspect.signature(f)
        hints = {'a': int, 'b': int, 'return': str}
        info = build_param_model('f', sig, hints)
        assert info.model is not None

    def test_var_positional(self) -> None:
        def f(*args: int) -> str:
            return 'x'

        sig = inspect.signature(f)
        hints = {'args': Tuple[int, ...], 'return': str}
        info = build_param_model('f', sig, hints)
        assert info.model is not None
        assert 'args' in info.model.model_fields

    def test_var_keyword(self) -> None:
        def f(**kwargs: str) -> str:
            return 'x'

        sig = inspect.signature(f)
        hints = {'kwargs': str, 'return': str}
        info = build_param_model('f', sig, hints)
        assert info.model is not None
        assert 'kwargs' in info.model.model_fields

    def test_no_fields_returns_none_model(self) -> None:
        def f() -> str:
            return 'x'

        sig = inspect.signature(f)
        info = build_param_model('f', sig, {'return': str})
        assert info.model is None

    def test_incompatible_var_positional_raises(self) -> None:
        """*args 的 tuple 标注若非 (... ) 形式则拒绝"""
        def f(*args: Tuple[int, int]) -> str:
            return 'x'

        sig = inspect.signature(f)
        hints = {'args': Tuple[int, int], 'return': str}
        with pytest.raises(TypeError, match='可变位置参数'):
            build_param_model('f', sig, hints)


class TestParamModelInfoConvertArgs:
    """校验模型 → (args, kwargs) 拆解"""

    def test_positional_and_keyword(self) -> None:
        def f(a: int, b: int, *, c: int = 0) -> str:
            return 'x'

        sig = inspect.signature(f)
        hints = {'a': int, 'b': int, 'c': int, 'return': str}
        info = build_param_model('f', sig, hints)
        assert info.model is not None
        model = info.model(a=1, b=2, c=3)
        args, kwargs = info.convert_args(model)
        assert args == [1, 2]
        assert kwargs == {'c': 3}

    def test_defaults_filled(self) -> None:
        def f(a: int, b: int = 10) -> str:
            return 'x'

        sig = inspect.signature(f)
        hints = {'a': int, 'b': int, 'return': str}
        info = build_param_model('f', sig, hints)
        assert info.model is not None
        model = info.model(a=1)
        args, kwargs = info.convert_args(model)
        assert args == [1, 10]

    def test_wrong_model_type_raises(self) -> None:
        def f(a: int) -> str:
            return 'x'

        sig = inspect.signature(f)
        hints = {'a': int, 'return': str}
        info = build_param_model('f', sig, hints)

        class Other(BaseModel):
            pass

        with pytest.raises(TypeError, match='不匹配'):
            info.convert_args(Other())


class TestBuildReturnModel:
    """返回值校验模型构建"""

    def test_str_return(self) -> None:
        info = build_return_model('f', {'return': str})
        assert info.model is not None
        assert info.raw is False

    def test_none_return(self) -> None:
        info = build_return_model('f', {'return': type(None)})
        assert info.model is None
        assert info.raw is False

    def test_raw_return(self) -> None:
        def f() -> Raw:
            return 'x'

        hints = typing.get_type_hints(f, include_extras=True)
        info = build_return_model('f', hints)
        assert info.model is None
        assert info.raw is True

    def test_missing_return_raises(self) -> None:
        with pytest.raises(TypeError, match='返回类型'):
            build_return_model('f', {})


class TestTypeHelpers:
    """Annotated 相关工具函数"""

    def test_flatten_annotated(self) -> None:
        tp = Annotated[Annotated[str, 'a'], 'b']
        base, metas = flatten_annotated(tp)
        assert base is str
        # 内层元数据在前、外层在后
        assert metas == ['a', 'b']

    def test_flatten_annotated_plain(self) -> None:
        base, metas = flatten_annotated(int)
        assert base is int
        assert metas == []

    def test_unwrap_annotated(self) -> None:
        assert unwrap_annotated(Annotated[int, 'x']) is int
        assert unwrap_annotated(int) is int

    def test_normalize_annotated_merges_descriptions(self) -> None:
        tp = Annotated[str, 'hello', 'world']
        normalized = normalize_annotated(tp)
        base, metas = flatten_annotated(normalized)
        assert base is str
        from pydantic.fields import FieldInfo

        merged = [m for m in metas if isinstance(m, FieldInfo)]
        assert len(merged) == 1
        assert merged[0].description == 'hello world'

    def test_is_raw_return(self) -> None:
        def f() -> Raw:
            return 'x'

        hints = typing.get_type_hints(f, include_extras=True)
        assert is_raw_return(hints['return']) is True
        assert is_raw_return(str) is False
