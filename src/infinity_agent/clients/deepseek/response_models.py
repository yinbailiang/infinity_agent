"""DeepSeek 特化的响应模型定义

DeepSeek API 与 OpenAI 完全兼容，唯一差异是流式响应中
``delta`` 额外携带 ``reasoning_content`` 字段（思考内容）。

这里通过继承 OpenAI 的响应模型并扩展该字段，复用全部既有
解析与校验逻辑，同时让思考内容得以保留。
"""

from typing import List, Optional

from pydantic import Field

from ..open_ai.response_models import Choice, Delta, StreamEvent


class DeepSeekDelta(Delta):
    """DeepSeek 流式增量：在 OpenAI Delta 基础上增加思考内容字段"""

    reasoning_content: Optional[str] = Field(
        default=None,
        description='思考（推理）内容增量，DeepSeek 特有字段',
    )


class DeepSeekChoice(Choice):
    """DeepSeek 选择项：delta 类型升级为 DeepSeekDelta"""

    delta: DeepSeekDelta = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        description='流式增量内容（含思考内容）'
    )


class DeepSeekStreamEvent(StreamEvent):
    """DeepSeek 流式响应数据块：choices 类型升级为 DeepSeekChoice"""

    choices: List[DeepSeekChoice] = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        description='completion 选择列表'
    )
