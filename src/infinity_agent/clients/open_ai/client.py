"""OpenAI 兼容 API 的流式客户端实现。

核心职责：
- SSE 流解析与事件分发
- tool_calls 跨 chunk 增量聚合
- 流终止时的收尾处理
"""

import json
import logging
import types
from typing import AsyncGenerator, List, Optional

from ...models import (
    DoneChunk,
    FinishChunk,
    Messages,
    StreamChunk,
    TextChunk,
    ToolCall,
    ToolCallCompleteChunk,
    UsageChunk,
    UsageStats,
)
from ..base import LLMClient
from ..exceptions import (
    LLMContentFilterError,
    LLMStreamError,
)
from .aggregation import aggregate_tool_call_deltas
from .config import OpenAIConfig, OpenAIRequestConfig, build_chat_completion_request
from .connection import ConnectionManager
from .request_models import ChatCompletionRequest
from .response_models import (
    Choice,
    StreamEvent,
    ToolCallDelta,
)

logger = logging.getLogger(__name__)


class _ToolCallAccumulator:
    """跨 chunk 累积 tool_calls 增量并在完成时产出聚合结果。"""

    __slots__ = ('_deltas',)

    def __init__(self) -> None:
        self._deltas: List[ToolCallDelta] = []

    def extend(self, deltas: List[ToolCallDelta]) -> None:
        """追加一批 tool_calls 增量片段。"""
        self._deltas.extend(deltas)

    def flush(self) -> List[ToolCall]:
        """一次性聚合并清空已累积的 tool_calls，直接产出 ToolCall 列表。"""
        if not self._deltas:
            return []
        result = aggregate_tool_call_deltas(self._deltas)
        self._deltas.clear()
        return result

    def flush_as_chunk(self) -> Optional[ToolCallCompleteChunk]:
        """将聚合结果包装为 ToolCallCompleteChunk，无待刷新时返回 None。"""
        aggregated = self.flush()
        if not aggregated:
            return None
        return ToolCallCompleteChunk(
            tool_calls=aggregated,
        )


class OpenAIClient(LLMClient):
    """OpenAI 兼容 API 的轻量异步客户端"""

    def __init__(self, config: OpenAIConfig) -> None:
        self._config = config
        self._conn = ConnectionManager(
            api_key=config.api_key,
            base_url=config.base_url,
            config=config.connection,
        )

    @property
    def model(self) -> str:
        """当前使用的模型名称。"""
        return self._config.model

    async def __aenter__(self) -> 'OpenAIClient':
        await self._conn.ensure_session()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[types.TracebackType],
    ) -> Optional[bool]:
        await self._conn.close()

    async def stream_chat(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        messages: Messages,
        config: Optional[OpenAIRequestConfig] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """流式对话，自动聚合 tool_calls 分片并产出完整 AggregatedToolCall。

        数据流::

            HTTP SSE 行  →  :meth:`_parse_sse_stream`   →  StreamEvent
            StreamEvent  →  :meth:`_handle_sse_event`   →  StreamChunk (yield)
            流结束后     →  :meth:`_on_stream_end`      →  补刷 + DoneChunk

        连接级重试（网络错误、可重试 HTTP 状态码）由
        :class:`ConnectionManager` 内部透明处理，调用方无需关心。

        异常沿三层异步生成器链向上传播，最终在调用方的
        ``async for`` 循环中抛出。

        :param messages: 对话消息列表
        :param config: 请求级别配置（工具、用量统计、响应格式等）
        """
        request_config = config or OpenAIRequestConfig()
        request_model = self._build_stream_payload(messages, request_config)
        accumulator = _ToolCallAccumulator()

        async with self._conn.request('chat/completions', request_model) as line_stream:
            async for event in self._parse_sse_stream(line_stream):
                async for chunk in self._handle_sse_event(event, accumulator, request_config.include_usage):
                    yield chunk

            # 流终止收尾
            async for chunk in self._on_stream_end(accumulator):
                yield chunk

    # ------------------------------------------------------------------
    # SSE 解析
    # ------------------------------------------------------------------

    async def _parse_sse_stream(
        self,
        line_stream: AsyncGenerator[bytes, None],
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        从字节行流中解析 SSE 并产出 StreamEvent。

        - 自动跳过注释行与非 data 行
        - 遇到 data: [DONE] 时正常结束
        - JSON 解析错误抛出 LLMStreamError
        """
        async for line_bytes in line_stream:
            line: str = line_bytes.decode('utf-8').strip()

            # 注释行
            if line.startswith(':'):
                continue

            # 非 data 行
            if not line.startswith('data: '):
                continue

            # [DONE] 终止标记
            if line == 'data: [DONE]':
                return

            # 解析 JSON
            data_str: str = line[5:].strip()
            try:
                yield StreamEvent.model_validate_json(data_str)
            except json.JSONDecodeError as e:
                raise LLMStreamError(
                    f'JSON decode error in stream: {e}',
                    original_error=e,
                    response_body=data_str,
                ) from e

    def _build_stream_payload(
        self,
        messages: Messages,
        request_config: OpenAIRequestConfig,
    ) -> ChatCompletionRequest:
        """构建流式请求模型，委托给 ``build_chat_completion_request``。"""
        return build_chat_completion_request(
            messages,
            self._config,
            request_config,
            stream=True,
        )

    async def _handle_sse_event(
        self,
        event: StreamEvent,
        accumulator: _ToolCallAccumulator,
        include_usage: bool,
    ) -> AsyncGenerator[StreamChunk, None]:
        """按固定管线处理单个 SSE 事件"""
        if event.choices:
            choice: Choice = event.choices[0]

            # Step 1: 文本增量
            if choice.delta.content:
                yield TextChunk(text=choice.delta.content)

            # Step 2: 工具调用累积
            if choice.delta.tool_calls:
                accumulator.extend(choice.delta.tool_calls)

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

        # Step 4: Token 用量
        if event.usage and include_usage:
            yield UsageChunk(
                usage=UsageStats(
                    prompt_tokens=event.usage.prompt_tokens,
                    completion_tokens=event.usage.completion_tokens,
                    total_tokens=event.usage.total_tokens,
                ),
            )

    async def _on_stream_end(
        self,
        accumulator: _ToolCallAccumulator,
    ) -> AsyncGenerator[StreamChunk, None]:
        """流终止时的收尾：补刷未完成的 tool_calls，产出 DONE 标记。"""
        chunk = accumulator.flush_as_chunk()
        if chunk is not None:
            yield chunk
        yield DoneChunk()
