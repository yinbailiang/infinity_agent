"""OpenAI 兼容 API 的 HTTP 连接管理"""

import asyncio
import json
import logging
import random
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

import aiohttp

from ..exceptions import (
    LLMContentFilterError,
    LLMContextLengthError,
    LLMHTTPError,
    LLMNetworkError,
    build_http_error,
)
from .config import OpenAIConnectionConfig
from .request_models import ChatCompletionRequest

logger = logging.getLogger(__name__)


class ConnectionManager:
    """管理 aiohttp ClientSession 的生命周期、HTTP 请求发送与连接级重试"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        config: Optional[OpenAIConnectionConfig] = None,
    ) -> None:
        self._api_key: str = api_key
        self._base_url: str = base_url.rstrip('/')
        self._config: OpenAIConnectionConfig = config or OpenAIConnectionConfig()

        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock: asyncio.Lock = asyncio.Lock()

    async def ensure_session(self) -> aiohttp.ClientSession:
        """懒初始化"""
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout_config = aiohttp.ClientTimeout(
                    total=None,
                    sock_connect=self._config.connect_timeout,
                    sock_read=self._config.timeout,
                )
                self._session = aiohttp.ClientSession(
                    headers={
                        'Authorization': f'Bearer {self._api_key}',
                        'Content-Type': 'application/json',
                    },
                    timeout=timeout_config,
                )
                logger.debug('Created new aiohttp ClientSession')
            return self._session

    async def close(self) -> None:
        """关闭底层 HTTP 会话。"""
        async with self._session_lock:
            if self._session and not self._session.closed:
                await self._session.close()
                logger.debug('Closed aiohttp ClientSession')
            self._session = None

    def _is_retryable(self, error: BaseException) -> bool:
        if isinstance(error, LLMNetworkError):
            return True
        if isinstance(error, LLMHTTPError) and error.status_code is not None:
            return error.status_code in self._config.retryable_status
        return False

    def _calc_delay(self, attempt: int) -> float:
        """计算第 attempt 次重试的等待时间（指数退避 + 可选抖动）。"""
        cfg = self._config
        delay = min(cfg.base_delay * (2 ** (attempt - 1)), cfg.max_delay)
        if cfg.jitter:
            delay = delay * (0.75 + random.random() * 0.5)  # ±25% 抖动
        return delay

    async def _do_request(
        self, endpoint: str, request_model: ChatCompletionRequest
    ) -> aiohttp.ClientResponse:
        """执行单次 POST 请求并校验 HTTP 状态码。

        Returns:
            ClientResponse

        Raises:
            LLMNetworkError: 连接/超时失败
            LLMHTTPError: 非 200 状态码
        """
        try:
            session: aiohttp.ClientSession = await self.ensure_session()
            url: str = f'{self._base_url}/{endpoint.lstrip("/")}'
            payload: Dict[str, Any] = request_model.model_dump(mode='json', exclude_none=True)
            resp: aiohttp.ClientResponse = await session.post(url, json=payload)
        except asyncio.TimeoutError as e:
            raise LLMNetworkError(f'Request timeout: {e}', original_error=e) from e
        except aiohttp.ClientError as e:
            raise LLMNetworkError(f'Connection error: {e}', original_error=e) from e

        if resp.status == 200:
            return resp

        error_text: str = await resp.text()
        resp.close()

        try:
            error_type: str = json.loads(error_text).get('error', {}).get('type', '')
        except (json.JSONDecodeError, AttributeError):
            error_type = ''

        match error_type:
            case 'content_filter':
                raise LLMContentFilterError(
                    message=f'Content filtered: {error_text}',
                    status_code=resp.status,
                    response_body=error_text,
                )
            case 'context_length_exceeded':
                raise LLMContextLengthError(
                    message=f'Context length exceeded: {error_text}',
                    status_code=resp.status,
                    response_body=error_text,
                )
            case _:
                raise build_http_error(
                    status_code=resp.status,
                    message=f'API error {resp.status}: {error_text}',
                    response_body=error_text,
                )

    async def _backoff(self, attempt: int, last_error: Optional[Exception]) -> None:
        delay: float = self._calc_delay(attempt)
        logger.warning(
            'Request failed (attempt %d/%d), retrying in %.1fs: %s',
            attempt,
            self._config.max_retries,
            delay,
            last_error,
        )
        if isinstance(last_error, LLMNetworkError):
            await self.close()
        await asyncio.sleep(delay)

    @asynccontextmanager
    async def request(
        self, endpoint: str, request_model: ChatCompletionRequest
    ) -> AsyncGenerator[AsyncGenerator[bytes, None], None]:
        last_error: Optional[Exception] = None
        total_attempts = self._config.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            try:
                resp = await self._do_request(endpoint, request_model)
            except (LLMNetworkError, LLMHTTPError) as e:
                if not self._is_retryable(e):
                    raise
                last_error = e
            else:
                try:
                    yield self._readline(resp)
                    return
                finally:
                    resp.close()

            await self._backoff(attempt, last_error)

        assert last_error is not None
        raise last_error

    async def _readline(self, resp: aiohttp.ClientResponse) -> AsyncGenerator[bytes, None]:
        while (line := await resp.content.readline()) != b'':
            yield line
