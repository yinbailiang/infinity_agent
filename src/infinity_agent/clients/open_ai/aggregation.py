"""OpenAI 流式 tool_calls 增量片段的聚合工具"""

import json
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from ...models import FunctionCall, ToolCall
from .response_models import ToolCallDelta


class _PendingToolCall(BaseModel):
    """工具调用聚合过程中的暂存状态"""

    id: Optional[str] = Field(default=None, description='工具调用唯一标识符')
    function_name: Optional[str] = Field(default=None, description='函数名称')
    function_arguments: str = Field(default='', description='已拼接的 JSON 参数字符串')


def aggregate_tool_call_deltas(
    tool_call_deltas: List[ToolCallDelta],
) -> List[ToolCall]:
    """
    将流式 tool_calls 增量片段聚合为完整的 ToolCall 列表。

    OpenAI 流式协议中，每个 tool_call delta 包含 index 字段，
    同一 index 的多个 delta 需要拼接 arguments 字符串后解析为 dict。
    """
    pending: Dict[int, _PendingToolCall] = {}
    for tc in tool_call_deltas:
        idx: int = tc.index
        if idx not in pending:
            pending[idx] = _PendingToolCall()
        entry = pending[idx]
        if tc.id:
            entry.id = tc.id
        func = tc.function
        if func is not None:
            if func.name:
                entry.function_name = func.name
            entry.function_arguments += func.arguments

    result: List[ToolCall] = []
    for idx in sorted(pending):
        entry = pending[idx]
        tool_id: Optional[str] = entry.id
        tool_name: Optional[str] = entry.function_name
        if tool_id is None or tool_name is None:
            continue
        try:
            arguments: Dict[str, object] = json.loads(entry.function_arguments) if entry.function_arguments else {}
        except json.JSONDecodeError:
            arguments = {}
        result.append(
            ToolCall(
                id=tool_id,
                function=FunctionCall(
                    name=tool_name,
                    arguments=arguments,
                ),
            )
        )
    return result
