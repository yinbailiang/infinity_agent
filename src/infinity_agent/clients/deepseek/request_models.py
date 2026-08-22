"""DeepSeek 特化请求模型定义

DeepSeek 请求体与 OpenAI 兼容，额外支持 ``reasoning_effort``
（思考强度）等特有参数。
"""

from typing import Literal, Optional

from pydantic import Field

from ..open_ai.request_models import ChatCompletionRequest


class DeepSeekChatCompletionRequest(ChatCompletionRequest):
    """DeepSeek 兼容的 Chat Completion 请求体"""

    reasoning_effort: Optional[Literal['low', 'medium', 'high']] = Field(
        default=None,
        description='思考强度，DeepSeek 特有参数（deepseek-reasoner / V3.1+）',
    )
