"""类型签名推断 — 从 Python 函数自动生成 ToolDefinition"""

import inspect
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    get_type_hints,
)

from llm.tools.introspectionist.model_builder import ReturnModelInfo

from ...models import ToolDefinition
from .convert import prop_from_schema, resolve_refs
from .model_builder import (
    ParamModelInfo,
    build_param_model,
    build_return_model,
)

ToolFunc = Callable[..., Any]


@dataclass(frozen=True)
class ToolDefinitionResult:
    """build_tool_definition 的返回结果。"""

    definition: ToolDefinition
    param_model: ParamModelInfo
    return_model: ReturnModelInfo


def build_tool_definition(
    func: ToolFunc,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> ToolDefinitionResult:
    tool_name = name or func.__name__
    tool_desc = func.__doc__ or description or ''
    definition = ToolDefinition.create(tool_name, tool_desc)

    try:
        hints: Dict[str, Any] = get_type_hints(func, include_extras=True)
    except Exception:
        raise TypeError(f'无法解析 {tool_name!r} 的类型标注，工具函数必须具有完整类型签名')
    sig: inspect.Signature = inspect.signature(func)

    param_info: Optional[ParamModelInfo] = build_param_model(tool_name, sig, hints)
    ret_info: ReturnModelInfo = build_return_model(tool_name, hints)

    if param_info.model is None:
        return ToolDefinitionResult(definition, param_info, ret_info)

    schema: Dict[str, Any] = param_info.model.model_json_schema()

    # ---- 4. 解析 $ref → 从 JSON Schema 反填 ToolDefinition ----
    schema = resolve_refs(schema)
    schema_props: Dict[str, Any] = schema.get('properties', {})
    required_list: List[str] = schema.get('required', [])

    for pname, prop_schema in schema_props.items():
        prop = prop_from_schema(prop_schema)
        definition.function.parameters.properties[pname] = prop
        if pname in required_list:
            definition.function.parameters.required.append(pname)

    return ToolDefinitionResult(definition, param_info, ret_info)
