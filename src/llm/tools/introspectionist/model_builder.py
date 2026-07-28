"""Pydantic 模型动态构建 — 从类型签名创建参数校验模型和返回值校验模型"""

import inspect
import logging
import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Literal, Never, Optional, get_args, get_origin

from pydantic import BaseModel, Field, RootModel, create_model
from pydantic.fields import FieldInfo

from llm.models.tools import RAW_RETURN

logger = logging.getLogger(__name__)


# =============================================================================
# TypeForm — 表达"任意合法类型标注"的并集
# =============================================================================

if TYPE_CHECKING:
    TypeForm = Annotated[Any, 'TypeForm']
else:

    class _TypeFormMeta(type):
        def __instancecheck__(cls, _) -> Literal[True]:
            return True

    class TypeForm(metaclass=_TypeFormMeta):
        def __new__(cls, *args: Any, **kwargs: Any) -> Never:
            raise TypeError('TypeForm cannot be instantiated')


@dataclass(frozen=True)
class DescSep:
    sep: str = ' '


@dataclass(frozen=True)
class NoAutoDesc:
    pass


def flatten_annotated(tp: TypeForm) -> tuple[TypeForm, list[Any]]:
    """递归展平 Annotated，返回 (裸类型, 按由深到浅顺序排列的元数据列表)。"""
    if get_origin(tp) is not Annotated:
        return tp, []
    args = get_args(tp)
    inner_base, inner_metas = flatten_annotated(args[0])
    outer_metas = list(args[1:])
    return inner_base, inner_metas + outer_metas


def is_raw_return(return_type: TypeForm) -> bool:
    """检测 return_type 是否为 Raw（Annotated[str, RAW_RETURN]）。"""
    if get_origin(return_type) is typing.Annotated:
        base, metas = flatten_annotated(return_type)
        return base is str and RAW_RETURN in metas
    return False


def normalize_annotated(type_form: TypeForm) -> TypeForm:
    origin: TypeForm = get_origin(type_form)

    # Annotated[T, ...] — 一次性展平所有嵌套层
    if origin is Annotated:
        base, all_metas = flatten_annotated(type_form)
        base = normalize_annotated(base)

        fields: list[FieldInfo] = []
        descriptions: List[str] = []
        no_auto_desc: bool = False

        other: list[Any] = []

        desc_sep = DescSep()
        for meta in all_metas:
            match meta:
                case DescSep():
                    desc_sep = meta
                case NoAutoDesc():
                    no_auto_desc = True
                case str():
                    descriptions.append(meta.strip())
                    other.append(meta)
                case FieldInfo():
                    fields.append(meta)
                case _:
                    other.append(meta)

        if descriptions and not no_auto_desc:
            fields.append(Field(description=desc_sep.sep.join(descriptions)))

        return Annotated[base, *other, FieldInfo.merge_field_infos(*fields)]

    # list[T], dict[K,V], tuple...
    if origin is not None:
        args: tuple[TypeForm, ...] = get_args(type_form)
        new_args: tuple[TypeForm, ...] = tuple(normalize_annotated(x) for x in args)
        return origin[*new_args]

    return type_form


def unwrap_annotated(tp: TypeForm) -> TypeForm:
    """剥离 Annotated 包装，返回基础类型。"""
    while get_origin(tp) is Annotated:
        args: tuple[Any, ...] = get_args(tp)
        tp = args[0] if args else Any
    return tp


def build_field_def(default: Any) -> FieldInfo:
    """构建 create_model 所需的字段定义。"""
    if default is inspect.Parameter.empty:
        return Field()

    if isinstance(default, FieldInfo):
        return default

    return Field(default=default)


@dataclass(frozen=True)
class ParamModelInfo:
    model: Optional[type[BaseModel]]
    func_sig: inspect.Signature

    def convert_args(self, model: BaseModel) -> tuple[list[Any], dict[str, Any]]:
        """将校验后的模型实例拆解为 (args, kwargs)，以便透传给原始函数调用。

        根据 ``func_sig`` 中各参数的 :class:`inspect.Parameter.kind` 自动分类：

        - ``POSITIONAL_ONLY`` / ``POSITIONAL_OR_KEYWORD`` → 位置参数
        - ``VAR_POSITIONAL`` → 展开为位置参数
        - ``KEYWORD_ONLY`` → 关键字参数
        - ``VAR_KEYWORD`` → 合并至关键字参数
        """
        if self.model is not None and not isinstance(model, self.model):
            raise TypeError(f'{type(model).__name__!r} 与预期的校验模型 {self.model.__name__!r} 不匹配')

        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        consumed: set[str] = set()

        for pname, param in self.func_sig.parameters.items():
            if pname in ('self', 'cls', 'return'):
                continue

            consumed.add(pname)
            value = getattr(model, pname)
            kind = param.kind

            if kind == inspect.Parameter.VAR_POSITIONAL:
                if value:
                    args.extend(value)
            elif kind == inspect.Parameter.VAR_KEYWORD:
                if value:
                    kwargs.update(value)
            elif kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                args.append(value)
            else:
                kwargs[pname] = value

        # 兜底：模型中有但签名中没有的字段 → kwargs
        for field_name in model.model_fields:
            if field_name not in consumed:
                kwargs[field_name] = getattr(model, field_name)

        return args, kwargs


def build_param_model(
    tool_name: str,
    sig: inspect.Signature,
    hints: dict[str, TypeForm],
) -> ParamModelInfo:
    fields: dict[str, tuple[TypeForm, FieldInfo]] = {}
    for pname, param in sig.parameters.items():
        if pname in ('self', 'cls', 'return'):
            continue

        type_form: TypeForm | None = hints.get(pname)
        if type_form is None:
            raise TypeError(f'参数 {pname!r} 缺少类型标注，工具 {tool_name!r} 的所有参数必须具有完整类型签名')
        type_form = normalize_annotated(type_form=type_form)

        if param.kind == param.VAR_POSITIONAL:
            # *args: int → list[int]; *args: tuple[int, ...] → list[int]
            base, metas = flatten_annotated(type_form)
            origin = get_origin(base)
            if origin is tuple:
                t_args = get_args(base)
                if len(t_args) == 2 and t_args[1] is Ellipsis:
                    base = t_args[0]
                else:
                    raise TypeError(
                        f'无法为 {tool_name!r} 构建参数校验模型，在处理可变位置参数 {pname!r} 时遇到无效类型签名。'
                    )

            if metas:
                type_form = Annotated[List[base], *metas]
            else:
                type_form = List[base]
            field_def = Field(default_factory=list)
        elif param.kind == param.VAR_KEYWORD:
            # **kwargs: T → dict[str, T];
            base, metas = flatten_annotated(type_form)
            if metas:
                type_form = Annotated[Dict[str, base], *metas]
            else:
                type_form = Dict[str, base]
            field_def = Field(default_factory=dict)
        else:
            field_def = build_field_def(param.default)

        fields[pname] = (type_form, field_def)

    if not fields:
        return ParamModelInfo(None, sig)

    try:
        model: BaseModel = create_model(f'_Tool_{tool_name}', **fields)  # pyright: ignore[reportUnknownVariableType, reportArgumentType, reportCallIssue]
        assert isinstance(model, type)
        return ParamModelInfo(model, sig)
    except Exception:
        logger.exception('Failed to build temp model for %r', tool_name)
        raise TypeError(f'无法为 {tool_name!r} 构建参数校验模型，请检查参数类型是否均为 Pydantic 兼容类型')


@dataclass(frozen=True)
class ReturnModelInfo:
    """返回值校验模型信息。

    :attr model: 校验模型（None 表示无返回）
    :attr raw: True 表示原始字符串透传不校验，False 表示走 Pydantic 校验
    """

    model: Optional[type[BaseModel]]
    raw: bool


def build_return_model(
    tool_name: str,
    hints: dict[str, Any],
) -> ReturnModelInfo:
    return_type = hints.get('return')
    if return_type is None:
        raise TypeError(
            f'工具 {tool_name!r} 缺少返回值类型标注，必须声明完整的返回类型（如 -> str, -> dict, -> Raw, -> None）'
        )
    if return_type is type(None):  # -> None
        return ReturnModelInfo(model=None, raw=False)
    if is_raw_return(return_type):  # -> Raw
        return ReturnModelInfo(model=None, raw=True)

    try:
        return ReturnModelInfo(model=RootModel[return_type], raw=False)
    except Exception:
        logger.exception('Failed to build temp return model for %r', tool_name)
        raise TypeError(f'无法为 {tool_name!r} 构建返回校验模型，请检查返回类型是否为 Pydantic 兼容类型')
