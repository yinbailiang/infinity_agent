"""
工具定义模型 — OpenAI function calling 格式

提供完整的 Pydantic 模型来描述 LLM 可调用的工具 Schema，
支持流畅构建、JSON Schema 兼容输出、从描述文件加载。

用法:
    from components.llm.models.tools import ToolDefinition, ParameterProperty

    # 自动推断（推荐）
    @registry.tool(description="获取天气")
    async def get_weather(city: str, lang: str = "zh") -> str: ...

    # 手动构造（直接操作模型）
    tool = ToolDefinition.create("get_weather", "获取天气")
    params = tool.function.parameters
    params.properties["city"] = ParameterProperty(type="string", description="城市")
    params.required.append("city")
    params.properties["lang"] = ParameterProperty(type="string", default="zh", enum=["zh", "en"])
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

# ---- JSON 兼容的标量类型 ----
JsonScalar = Union[str, bool, int, float, None]
"""enum / default 字段可接受的 JSON 标量值，排除不可序列化的 Python 对象。"""


RAW_RETURN = object()
Raw = Annotated[str, RAW_RETURN]
"""返回值标记类型 — 类型检查器视为 ``str``，运行时不做校验、不转 JSON。

工具函数标注 ``-> Raw`` 时，返回 ``str`` 不会触发类型错误，
运行时跳过返回值模型校验和 JSON 序列化，直接 ``str(result)`` 原样传递。

用法::

    from ..tools import Raw

    @registry.tool()
    async def markdown_report(data: list[dict]) -> Raw:
        return "| A | B |\\n|---|---|\\n..."
"""


class ParameterProperty(BaseModel):
    """
    JSON Schema 单个参数属性。

    对应 OpenAI tool.parameters.properties 中的一项。
    支持嵌套对象：当 type="object" 时，可通过 ``properties`` 描述内部字段。
    """

    type: Optional[Literal['string', 'number', 'integer', 'boolean', 'object', 'array']] = Field(
        default=None,
        description='参数类型。与 anyOf 互斥，二选一',
    )
    anyOf: Optional[List['ParameterProperty']] = Field(
        default=None,
        description='多类型联合（Union[int, str] → anyOf=[integer, string]）',
    )
    description: Optional[str] = Field(default=None, description='参数描述')
    enum: Optional[List[JsonScalar]] = Field(default=None, description='可选枚举值列表')
    default: Optional[JsonScalar] = Field(default=None, description='默认值')
    properties: Optional[Dict[str, 'ParameterProperty']] = Field(
        default=None, description='嵌套对象属性（type=object 时使用）'
    )
    required: Optional[List[str]] = Field(default=None, description='嵌套对象的必填字段列表（type=object 时使用）')
    items: Optional['ParameterProperty'] = Field(default=None, description='数组元素 Schema（type=array 时使用）')
    additionalProperties: Optional[Union[bool, 'ParameterProperty']] = Field(
        default=None,
        description='自由键值对约束（type=object 且无固定 properties 时使用）',
    )
    nullable: Optional[bool] = Field(
        default=None,
        description='参数是否可为 null（对应 anyOf 中包含 {"type":"null"}）',
    )

    minimum: Optional[float] = Field(default=None, description='数值最小值')
    maximum: Optional[float] = Field(default=None, description='数值最大值')
    exclusiveMinimum: Optional[float] = Field(default=None, description='数值排他下限')
    exclusiveMaximum: Optional[float] = Field(default=None, description='数值排他上限')
    pattern: Optional[str] = Field(default=None, description='字符串正则模式')
    minLength: Optional[int] = Field(default=None, description='字符串最小长度')
    maxLength: Optional[int] = Field(default=None, description='字符串最大长度')
    examples: Optional[List[Any]] = Field(default=None, description='示例值列表')

    @model_validator(mode='after')
    def _validate_consistency(self) -> 'ParameterProperty':
        has_type = self.type is not None
        has_anyof = self.anyOf is not None and len(self.anyOf) > 0

        # type 和 anyOf 必须二选一，且互斥
        if has_type and has_anyof:
            raise ValueError('type 和 anyOf 互斥，只能提供其一')
        if not has_type and not has_anyof:
            raise ValueError('type 和 anyOf 必须提供其一')

        # enum 元素类型与 type 兼容
        if self.enum is not None:
            self._check_enum_consistency()

        # default 与 type 兼容
        if self.default is not None:
            self._check_default_consistency()

        # type=object 必须提供 properties 或 additionalProperties
        if self.type == 'object':
            if self.properties is None and self.additionalProperties is None:
                raise ValueError("type='object' 时必须提供 properties 或 additionalProperties")

        # type=array 必须提供 items
        if self.type == 'array' and self.items is None:
            raise ValueError("type='array' 时必须提供 items")

        # required 中的名字必须存在于 properties
        if self.required and self.properties:
            missing = [n for n in self.required if n not in self.properties]
            if missing:
                raise ValueError(f'required 中包含不在 properties 中的字段: {missing}')

        return self

    def _check_enum_consistency(self) -> None:
        """验证 enum 元素类型与声明的 type 兼容。"""
        assert self.enum is not None
        numeric_types = {'integer', 'number'}
        string_types = {'string'}
        for i, val in enumerate(self.enum):
            if self.type in numeric_types and not isinstance(val, (int, float)):
                raise ValueError(f'enum[{i}] 类型不兼容: type={self.type!r} 但 enum 值为 {type(val).__name__!r}')
            if self.type in string_types and not isinstance(val, str):
                raise ValueError(f'enum[{i}] 类型不兼容: type={self.type!r} 但 enum 值为 {type(val).__name__!r}')
            if self.type == 'boolean' and not isinstance(val, bool):
                raise ValueError(f"enum[{i}] 类型不兼容: type='boolean' 但 enum 值为 {type(val).__name__!r}")

    def _check_default_consistency(self) -> None:
        """验证 default 值与声明的 type 或 anyOf 分支兼容。"""
        type_to_python: dict[str, tuple[type, ...]] = {
            'string': (str,),
            'number': (int, float),
            'integer': (int,),
            'boolean': (bool,),
        }

        if self.type is not None:
            expected = type_to_python.get(self.type)
            if expected is not None and not isinstance(self.default, expected):
                raise ValueError(
                    f'default 类型不兼容: type={self.type!r} '
                    f'但 default={self.default!r} (类型 {type(self.default).__name__})'
                )
        elif self.anyOf:
            # anyOf 多分支：default 兼容任一分支即可
            for branch in self.anyOf:
                if branch.type is None:
                    continue
                expected = type_to_python.get(branch.type)
                if expected is not None and isinstance(self.default, expected):
                    break
            else:
                # 未匹配任何分支，仅在有明确类型分支时报错
                typed_branches = [b.type for b in self.anyOf if b.type is not None]
                if typed_branches:
                    raise ValueError(
                        f'default 类型不兼容: anyOf={typed_branches!r} '
                        f'但 default={self.default!r} (类型 {type(self.default).__name__})'
                    )


class ToolParameters(BaseModel):
    """
    工具函数的参数 Schema（JSON Schema object）。

    对应 OpenAI tool.function.parameters。
    """

    type: Literal['object'] = Field(default='object', description='固定为 object')
    properties: Dict[str, ParameterProperty] = Field(default_factory=dict, description='参数属性映射')
    required: List[str] = Field(default_factory=list, description='必填参数名列表')


class ToolFunction(BaseModel):
    """
    工具函数定义。

    对应 OpenAI tool.function。
    """

    name: str = Field(description='函数名称')
    description: str = Field(default='', description='函数描述')
    parameters: ToolParameters = Field(default_factory=ToolParameters, description='参数定义')


class ToolDefinition(BaseModel):
    """
    OpenAI 兼容的工具定义（顶层）。

    对应 tools 数组中的一项。
    """

    type: Literal['function'] = Field(default='function', description='固定为 function')
    function: ToolFunction = Field(description='函数详情')

    # ------------------------------------------------------------------
    # 工厂 / 构建器
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
    ) -> 'ToolDefinition':
        """快捷创建工具定义并开始添加参数。"""
        return cls(
            function=ToolFunction(
                name=name,
                description=description,
            )
        )

    def model_dump_tool(self, **kwargs: Any) -> Dict[str, Any]:
        """
        输出为 OpenAI API 兼容的字典。
        """
        return self.model_dump(mode='json', exclude_none=True, **kwargs)


ToolDefinitions = List[ToolDefinition]
