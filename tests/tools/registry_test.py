"""ToolRegistry 测试：注册、定义导出、调度执行。"""

import json
from typing import Annotated, List, Literal

import pytest

from infinity_agent.models import MessageRole, ToolCall
from infinity_agent.tools import Raw, ToolRegistry


def _tool_call(name: str, arguments: dict, tool_call_id: str = 'call_1') -> ToolCall:
    return ToolCall(id=tool_call_id, function={'name': name, 'arguments': arguments})


class TestToolRegistryRegister:
    """注册行为"""

    def test_register_decorator_infers_definition(self) -> None:
        registry = ToolRegistry()

        @registry.tool(description='获取天气')
        async def get_weather(city: str) -> str:
            """获取指定城市天气"""
            return f'{city}: sunny'

        tools = registry.definitions
        assert len(tools) == 1
        tool = tools[0]
        assert tool.function.name == 'get_weather'
        # 文档字符串优先于显式 description
        assert tool.function.description == '获取指定城市天气'
        assert 'city' in tool.function.parameters.properties
        assert tool.function.parameters.required == ['city']

    def test_register_manual(self) -> None:
        registry = ToolRegistry()

        def add(a: int, b: int) -> int:
            return a + b

        registry.register(add, name='my_add', description='相加')
        tool = registry.definitions[0]
        assert tool.function.name == 'my_add'
        assert set(tool.function.parameters.properties.keys()) == {'a', 'b'}

    def test_duplicate_register_raises(self) -> None:
        registry = ToolRegistry()

        @registry.tool()
        def f(x: int) -> int:
            return x

        with pytest.raises(ValueError, match='already registered'):
            registry.register(f)

    def test_register_decorator_returns_original(self) -> None:
        registry = ToolRegistry()

        @registry.tool()
        def f(x: int) -> int:
            return x

        assert f(1) == 1

    def test_model_dump_tools(self) -> None:
        registry = ToolRegistry()

        @registry.tool()
        def f(x: int) -> int:
            return x

        dumped = registry.model_dump_tools()
        assert dumped[0]['type'] == 'function'
        assert dumped[0]['function']['name'] == 'f'
        assert 'x' in dumped[0]['function']['parameters']['properties']


class TestToolRegistryInvoke:
    """工具调度执行"""

    @pytest.mark.asyncio
    async def test_invoke_sync_function(self) -> None:
        registry = ToolRegistry()

        @registry.tool()
        def add(a: int, b: int) -> int:
            return a + b

        msg = await registry.invoke(_tool_call('add', {'a': 1, 'b': 2}))
        assert msg.role == MessageRole.TOOL
        assert msg.tool_call_id == 'call_1'
        assert json.loads(msg.content) == 3

    @pytest.mark.asyncio
    async def test_invoke_async_function(self) -> None:
        registry = ToolRegistry()

        @registry.tool()
        async def greet(name: str) -> str:
            return f'Hello, {name}'

        msg = await registry.invoke(_tool_call('greet', {'name': 'World'}))
        assert json.loads(msg.content) == 'Hello, World'

    @pytest.mark.asyncio
    async def test_invoke_unknown_tool(self) -> None:
        registry = ToolRegistry()
        msg = await registry.invoke(_tool_call('nope', {}))
        assert 'error' in json.loads(msg.content)
        assert 'Unknown tool' in json.loads(msg.content)['error']

    @pytest.mark.asyncio
    async def test_invoke_param_validation_failure(self) -> None:
        registry = ToolRegistry()

        @registry.tool()
        def add(a: int, b: int) -> int:
            return a + b

        msg = await registry.invoke(_tool_call('add', {'a': 'not int', 'b': 2}))
        assert 'error' in json.loads(msg.content)

    @pytest.mark.asyncio
    async def test_invoke_function_exception(self) -> None:
        registry = ToolRegistry()

        @registry.tool()
        def boom(x: int) -> int:
            raise RuntimeError('kaboom')

        msg = await registry.invoke(_tool_call('boom', {'x': 1}))
        assert 'error' in json.loads(msg.content)
        assert 'kaboom' in json.loads(msg.content)['error']

    @pytest.mark.asyncio
    async def test_invoke_raw_return_passthrough(self) -> None:
        registry = ToolRegistry()

        @registry.tool()
        def report(data: List[int]) -> Raw:
            return '| A |\n|---|\n'

        msg = await registry.invoke(_tool_call('report', {'data': [1, 2]}))
        assert msg.content == '| A |\n|---|\n'

    @pytest.mark.asyncio
    async def test_invoke_none_return(self) -> None:
        registry = ToolRegistry()

        @registry.tool()
        def noop(x: int) -> None:
            return None

        msg = await registry.invoke(_tool_call('noop', {'x': 1}))
        assert 'success' in msg.content

    @pytest.mark.asyncio
    async def test_invoke_all(self) -> None:
        registry = ToolRegistry()

        @registry.tool()
        def add(a: int, b: int) -> int:
            return a + b

        calls = [
            _tool_call('add', {'a': 1, 'b': 2}, tool_call_id='c1'),
            _tool_call('add', {'a': 10, 'b': 20}, tool_call_id='c2'),
        ]
        messages = await registry.invoke_all(calls)
        assert len(messages) == 2
        assert [json.loads(m.content) for m in messages] == [3, 30]

    @pytest.mark.asyncio
    async def test_invoke_all_empty(self) -> None:
        registry = ToolRegistry()
        assert await registry.invoke_all([]) == []


class TestToolRegistryWithAnnotated:
    """Annotated 描述与 Literal 枚举"""

    @pytest.mark.asyncio
    async def test_annotated_description(self) -> None:
        registry = ToolRegistry()

        @registry.tool()
        def f(city: Annotated[str, '城市名称']) -> str:
            return city

        prop = registry.definitions[0].function.parameters.properties['city']
        assert prop.description == '城市名称'

    @pytest.mark.asyncio
    async def test_literal_maps_to_enum(self) -> None:
        registry = ToolRegistry()

        @registry.tool()
        def f(lang: Literal['zh', 'en'] = 'zh') -> str:
            return lang

        prop = registry.definitions[0].function.parameters.properties['lang']
        assert prop.enum == ['zh', 'en']
        assert prop.default == 'zh'
