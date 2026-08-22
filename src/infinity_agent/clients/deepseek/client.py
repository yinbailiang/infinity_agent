"""DeepSeek 特化流式客户端。

在 OpenAI 兼容客户端基础上，特化处理 DeepSeek 的思考内容：

- 流式响应中 ``delta.reasoning_content`` 被解析并输出为 :class:`ThinkingChunk`
- Token 用量中透传 ``reasoning_tokens`` 细分统计
- 请求侧支持 ``reasoning_effort`` 思考强度控制
- 其余管线（SSE 解析、tool_calls 聚合、连接重试）复用 OpenAI 基础版

数据流::

    HTTP SSE 行  →  :meth:`_parse_sse_stream`   →  DeepSeekStreamEvent
    DeepSeekStreamEvent →  :meth:`_handle_sse_event`  →  StreamChunk (yield)
    流结束后     →  :meth:`_on_stream_end`      →  补刷 + DoneChunk
"""

import logging
from typing import AsyncGenerator, ClassVar, Optional

from ...models import (
    FinishChunk,
    Messages,
    StreamChunk,
    TextChunk,
    ThinkingChunk,
    UsageChunk,
    UsageStats,
)
from ..config import RequestConfig
from ..exceptions import LLMConfigError, LLMContentFilterError
from ..open_ai.client import OpenAIClient, ToolCallAccumulator
from ..open_ai.config import OpenAIRequestConfig
from ..open_ai.response_models import StreamEvent
from .config import DeepSeekConfig, DeepSeekRequestConfig, build_deepseek_request
from .request_models import DeepSeekChatCompletionRequest
from .response_models import DeepSeekDelta, DeepSeekStreamEvent

logger = logging.getLogger(__name__)


class DeepSeekClient(OpenAIClient):
    """DeepSeek API 特化客户端：接收并输出思考内容（reasoning_content）"""

    #: 覆盖父类钩子：使用能保留 reasoning_content 字段的事件模型解析 SSE
    #: type[...] 覆写为子类型在语义上安全，pyright 对类变量做不变性检查，故局部忽略
    _event_cls: ClassVar[type[DeepSeekStreamEvent]] = DeepSeekStreamEvent  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, config: DeepSeekConfig) -> None:
        self._config: DeepSeekConfig = config
        super().__init__(config)

    async def stream_chat(
        self,
        messages: Messages,
        config: Optional[RequestConfig] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """流式对话，输出思考内容增量并自动聚合 tool_calls 分片。

        :param messages: 对话消息列表
        :param config: DeepSeek 请求级别配置（工具、用量统计、思考强度等）
        """
        request_config = config or DeepSeekRequestConfig()
        if not isinstance(request_config, DeepSeekRequestConfig):
            raise LLMConfigError(
                f'DeepSeekClient.stream_chat expects DeepSeekRequestConfig, got {type(request_config).__name__}'
            )
        request_model = self._build_stream_payload(messages, request_config)
        accumulator = ToolCallAccumulator()

        async with self._conn.request('chat/completions', request_model) as line_stream:
            async for event in self._parse_sse_stream(line_stream):
                async for chunk in self._handle_sse_event(event, accumulator, request_config.include_usage):
                    yield chunk

            # 流终止收尾
            async for chunk in self._on_stream_end(accumulator):
                yield chunk

    def _build_stream_payload(
        self,
        messages: Messages,
        request_config: OpenAIRequestConfig,
    ) -> DeepSeekChatCompletionRequest:
        """构建流式请求体（含 DeepSeek 特有参数），委托给 ``build_deepseek_request``。"""
        return build_deepseek_request(
            messages,
            self._config,
            request_config,
            stream=True,
        )

    async def _parse_sse_stream(
        self,
        line_stream: AsyncGenerator[bytes, None],
    ) -> AsyncGenerator[DeepSeekStreamEvent, None]:
        """解析 DeepSeek SSE 流，产出保留 ``reasoning_content`` 的事件对象。"""
        async for event in super()._parse_sse_stream(line_stream):
            assert isinstance(event, DeepSeekStreamEvent)
            yield event

    async def _handle_sse_event(
        self,
        event: StreamEvent,
        accumulator: ToolCallAccumulator,
        include_usage: bool,
    ) -> AsyncGenerator[StreamChunk, None]:
        """按固定管线处理单个 SSE 事件（DeepSeek 特化版）。

        与 OpenAI 版相比：
        - Step 0: 先产出思考内容增量 :class:`ThinkingChunk`
        - Step 4: Token 用量透传 ``reasoning_tokens`` 细分统计
        """
        if event.choices:
            choice = event.choices[0]
            delta = choice.delta

            # Step 0: 思考内容增量
            if isinstance(delta, DeepSeekDelta) and delta.reasoning_content:
                yield ThinkingChunk(text=delta.reasoning_content)

            # Step 1: 文本增量
            if delta.content:
                yield TextChunk(text=delta.content)

            # Step 2: 工具调用累积
            if delta.tool_calls:
                accumulator.extend(delta.tool_calls)

            # Step 3: 结束理由
            if choice.finish_reason is not None:
                if choice.finish_reason == 'content_filter':
                    raise LLMContentFilterError(
                        message='Content filtered by safety system',
                        status_code=400,
                    )
                if choice.finish_reason == 'tool_calls':
                    chunk = accumulator.flush_as_chunk()
                    if chunk is not None:
                        yield chunk
                yield FinishChunk(finish_reason=choice.finish_reason)

        # Step 4: Token 用量（透传 reasoning_tokens 细分统计）
        if event.usage and include_usage:
            details = event.usage.completion_tokens_details
            yield UsageChunk(
                usage=UsageStats(
                    prompt_tokens=event.usage.prompt_tokens,
                    completion_tokens=event.usage.completion_tokens,
                    total_tokens=event.usage.total_tokens,
                    reasoning_tokens=details.reasoning_tokens if details else None,
                ),
            )
