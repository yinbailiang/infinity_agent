import logging
from typing import Any, cast

from ...models.tools import ParameterProperty

logger = logging.getLogger(__name__)


def resolve_refs(schema: dict[str, Any], max_depth: int = 32) -> dict[str, Any]:
    """内联解析 JSON Schema 中所有 ``$ref`` 引用。

    将 ``{"$ref": "#/$defs/ModelName"}`` 替换为 ``$defs`` 中对应的
    实际定义，递归处理嵌套引用。

    对循环引用和深度超过 ``_MAX_REF_DEPTH`` 的嵌套引用进行截断，
    截断时生成一个 ``object`` 节点并在 ``description`` 中说明原因。
    """
    defs = schema.get('$defs', {})
    if not defs:
        return schema

    resolved_refs: set[str] = set()

    def _resolve(node: Any, depth: int = 0) -> Any:
        if isinstance(node, dict):
            node = cast(dict[str, Any], node)
            if '$ref' in node:
                ref_path: str = node['$ref']
                if ref_path.startswith('#/$defs/'):
                    model_name = ref_path[len('#/$defs/') :]
                    if model_name in resolved_refs:
                        return {
                            'type': 'object',
                            'additionalProperties': True,
                            'description': f'[循环引用截断] $ref "{ref_path}" 已被解析，避免无限递归',
                        }
                    if depth >= max_depth:
                        return {
                            'type': 'object',
                            'additionalProperties': True,
                            'description': f'[深度截断] $ref "{ref_path}" 超过最大解析深度 {max_depth}',
                        }
                    resolved = defs.get(model_name)
                    if resolved is not None:
                        resolved_refs.add(model_name)
                        result = _resolve(resolved, depth + 1)
                        resolved_refs.discard(model_name)
                        return result
            return {k: _resolve(v, depth) for k, v in node.items()}
        elif isinstance(node, list):
            node = cast(list[Any], node)
            return [_resolve(item, depth) for item in node]
        return node

    return _resolve(schema)


def _resolve_non_null_branch(
    prop_schema: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    """从 anyOf/oneOf 中提取所有非 null 分支的完整 Schema。

    ``Optional[int]`` → ``{"anyOf":[{"type":"integer"},{"type":"null"}]}``
    → 返回 ``([{"type":"integer"}], True)``

    ``Union[int, str, None]`` → 返回 ``([{"type":"integer"}, {"type":"string"}], True)``

    若无非 null 分支返回 ``([], False)``。
    """
    for key in ('anyOf', 'oneOf'):
        variants = prop_schema.get(key)
        if not isinstance(variants, list):
            continue
        variants = cast(list[dict[str, Any]], variants)
        non_null = [v for v in variants if v.get('type') != 'null']
        has_null = any(v.get('type') == 'null' for v in variants)
        if non_null:
            return non_null, has_null
    return [], False


def prop_from_schema(prop_schema: dict[str, Any]) -> ParameterProperty:
    """从 JSON Schema 属性节点递归构建 ParameterProperty。"""

    raw_type = prop_schema.get('type')
    nullable: bool | None = None
    anyof_branches: list[dict[str, Any]] | None = None
    # 保存外层 envelope 字段（anyOf/oneOf 包裹层可能带有 description/default 等）
    envelope: dict[str, Any] = {k: prop_schema[k] for k in ('description', 'default', 'title') if k in prop_schema}
    if raw_type is None:
        # 尝试从 anyOf/oneOf 中提取所有非 null 分支，并检测 null 变体
        resolved_branches, has_null = _resolve_non_null_branch(prop_schema)
        if resolved_branches:
            if len(resolved_branches) > 1:
                # 多分支：全部作为 anyOf，不提取主分支
                anyof_branches = resolved_branches
            else:
                # 单分支：将 envelope 字段合并进分支，提升为主 schema
                for k, v in envelope.items():
                    resolved_branches[0].setdefault(k, v)
                prop_schema = resolved_branches[0]
                raw_type = prop_schema.get('type')
        if has_null:
            nullable = True
    if raw_type is None:
        raw_type = 'string'

    kwargs: dict[str, Any] = {
        'description': prop_schema.get('description', None),
    }
    # type 和 anyOf 二选一
    if anyof_branches:
        kwargs['anyOf'] = [prop_from_schema(branch) for branch in anyof_branches]
    else:
        kwargs['type'] = raw_type
    if nullable is not None:
        kwargs['nullable'] = nullable
    if 'enum' in prop_schema:
        kwargs['enum'] = prop_schema['enum']
    if 'default' in prop_schema:
        default_val = prop_schema['default']
        # ParameterProperty.default 仅接受 JSON 标量，跳过 dict/list 等复合默认值
        if isinstance(default_val, (str, int, float, bool, type(None))):
            kwargs['default'] = default_val

    # JSON Schema 约束字段
    for constraint in (
        'minimum',
        'maximum',
        'exclusiveMinimum',
        'exclusiveMaximum',
        'pattern',
        'minLength',
        'maxLength',
        'examples',
    ):
        if constraint in prop_schema:
            kwargs[constraint] = prop_schema[constraint]

    # 嵌套对象
    nested = prop_schema.get('properties')
    if nested and kwargs['type'] == 'object':
        kwargs['properties'] = {k: prop_from_schema(v) for k, v in nested.items()}
        nested_req = prop_schema.get('required')
        if nested_req:
            kwargs['required'] = nested_req

    # 数组元素
    items_schema = prop_schema.get('items')
    if items_schema and raw_type == 'array':
        kwargs['items'] = prop_from_schema(items_schema)

    # 自由键值对（dict[str, X] 等）
    add_props = prop_schema.get('additionalProperties')
    if add_props is not None and raw_type == 'object' and 'properties' not in kwargs:
        if isinstance(add_props, dict):
            kwargs['additionalProperties'] = prop_from_schema(cast(dict[str, Any], add_props))
        else:
            kwargs['additionalProperties'] = add_props

    # 裸 object（无 properties 且无 additionalProperties）→ 允许任意键值
    if raw_type == 'object' and 'properties' not in kwargs and 'additionalProperties' not in kwargs:
        kwargs['additionalProperties'] = True

    return ParameterProperty(**kwargs)
