"""工具注册表 — 收集定义、调度 ToolCall 执行"""

import asyncio
import inspect
import json
import logging
from typing import Any, Callable, Dict, List

from ..models import Message, ToolCall, ToolDefinition
from .introspectionist.introspection import ToolFunc, build_tool_definition
from .introspectionist.model_builder import ParamModelInfo, ReturnModelInfo

logger = logging.getLogger(__name__)


class ToolEntry:
    """单个工具的运行时记录：定义 + 实现 + 参数/返回值校验模型"""

    __slots__ = ('definition', 'func', 'param_model', 'return_model', 'raw_return')

    def __init__(
        self,
        definition: ToolDefinition,
        func: ToolFunc,
        param_model: ParamModelInfo,
        return_model: ReturnModelInfo,
        *,
        raw_return: bool = False,
    ) -> None:
        self.definition = definition
        self.func = func
        self.param_model = param_model
        self.return_model = return_model
        self.raw_return = raw_return


class ToolRegistry:
    """工具注册表：收集定义、调度 ToolCall 执行。

    提供三种注册方式：

    - ``@registry.tool()`` — 自动从类型签名推断定义
    - ``@registry.register(func)`` — 手动注册

    用法::

        from .. import ToolRegistry, ToolDefinition

        registry = ToolRegistry()

        # 自动推断
        @registry.tool(description="获取天气")
        async def get_weather(city: str) -> str: ...

        # 收集定义
        tools = registry.definitions

        # 执行调用
        msg = await registry.invoke(tool_call)
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolEntry] = {}

    def tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[[ToolFunc], ToolFunc]:
        """装饰器：根据函数类型签名自动生成 ToolDefinition 并注册。

        支持同步/异步函数。从类型提示推断参数类型，从默认值判断必填/可选，
        ``Literal`` 自动映射为 enum，``Annotated`` 元数据可作为参数描述。

        用法::

            @registry.tool(description="获取指定城市的天气信息")
            async def get_weather(
                city: Annotated[str, "城市名称"],
                lang: Literal["zh", "en"] = "zh",
            ) -> str: ...

        :param name: 工具名称，默认使用函数名
        :param description: 工具描述，默认使用函数文档字符串首行
        """

        def decorator(func: ToolFunc) -> ToolFunc:
            self.register(func, name=name, description=description)
            return func

        return decorator

    def register(
        self,
        func: ToolFunc,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        tool_name = name or func.__name__
        result = build_tool_definition(func, name=tool_name, description=description)

        if tool_name in self._tools:
            raise ValueError(f'Tool {tool_name!r} is already registered. Each tool must have a unique name.')
        self._tools[tool_name] = ToolEntry(
            definition=result.definition,
            func=func,
            param_model=result.param_model,
            return_model=result.return_model,
            raw_return=result.return_model.raw,
        )
        logger.debug('Registered tool: %s', tool_name)
    

    @property
    def definitions(self) -> List[ToolDefinition]:
        """所有已注册工具的 ToolDefinition 列表，可直接传入 LLM API。"""
        return [entry.definition for entry in self._tools.values()]

    def model_dump_tools(self) -> List[Dict[str, Any]]:
        """导出为 OpenAI API 兼容的 tools 数组。"""
        return [t.model_dump_tool() for t in self.definitions]

    # ------------------------------------------------------------------
    # 调度执行
    # ------------------------------------------------------------------

    async def invoke(self, tool_call: ToolCall) -> Message:
        name = tool_call.function.name
        entry = self._tools.get(name)

        if entry is None:
            error_msg = f'Unknown tool: {name!r}'
            logger.error(error_msg)
            return Message.tool(
                content=json.dumps({'error': error_msg}, ensure_ascii=False),
                tool_call_id=tool_call.id,
            )

        try:
            # 参数校验：ToolCall.arguments → model_validate（不可跳过）
            try:
                param_info = entry.param_model
                if param_info.model is not None:
                    validated = param_info.model.model_validate(tool_call.function.arguments)
                    args, kwargs = param_info.convert_args(validated)
                else:
                    args, kwargs = [], {}
            except Exception as ve:
                logger.warning('Tool %r param validation failed: %s', name, ve)
                return Message.tool(
                    content=json.dumps({'error': str(ve)}, ensure_ascii=False),
                    tool_call_id=tool_call.id,
                )

            result = entry.func(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result

            if entry.raw_return:
                content = str(result)
            elif entry.return_model.model is not None:
                validated = entry.return_model.model.model_validate(result)
                content = validated.model_dump_json(indent=4)
            else:
                content = json.dumps(f'Tool {name!r} run success.', ensure_ascii=False)

        except Exception as e:
            logger.exception('Tool %r execution failed', name)
            content = json.dumps({'error': str(e)}, ensure_ascii=False)

        return Message.tool(content=content, tool_call_id=tool_call.id)

    async def invoke_all(self, tool_calls: List[ToolCall]) -> List[Message]:
        if not tool_calls:
            return []
        return list(await asyncio.gather(*(self.invoke(tc) for tc in tool_calls)))
