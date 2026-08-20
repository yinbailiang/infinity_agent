"""OpenAI HTTP 连接管理测试：会话、重试、退避、错误映射。"""

import asyncio
import ssl
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from infinity_agent.clients.exceptions import (
    LLMContentFilterError,
    LLMContextLengthError,
    LLMHTTPError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMServerError,
)
from infinity_agent.clients.open_ai.config import OpenAIConnectionConfig
from infinity_agent.clients.open_ai.connection import ConnectionManager
from infinity_agent.clients.open_ai.request_models import ChatCompletionRequest


def _make_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(model='m', messages=[{'role': 'user', 'content': 'Hi'}])


class TestBuildSSLContext:
    """SSL 上下文构建"""

    def test_returns_ssl_context(self) -> None:
        from infinity_agent.clients.open_ai.connection import _build_ssl_context

        ctx = _build_ssl_context()
        assert isinstance(ctx, ssl.SSLContext)


class TestIsRetryable:
    """重试判定"""

    def test_network_error_retryable(self) -> None:
        mgr = ConnectionManager(api_key='k', base_url='http://x')
        assert mgr._is_retryable(LLMNetworkError('x')) is True

    def test_retryable_status_code(self) -> None:
        mgr = ConnectionManager(api_key='k', base_url='http://x')
        assert mgr._is_retryable(LLMServerError('x', status_code=500)) is True
        assert mgr._is_retryable(LLMRateLimitError('x', status_code=429)) is True

    def test_non_retryable_status_code(self) -> None:
        mgr = ConnectionManager(api_key='k', base_url='http://x')
        assert mgr._is_retryable(LLMHTTPError('x', status_code=400)) is False

    def test_http_error_without_status_not_retryable(self) -> None:
        mgr = ConnectionManager(api_key='k', base_url='http://x')
        assert mgr._is_retryable(LLMHTTPError('x')) is False

    def test_custom_retryable_status(self) -> None:
        mgr = ConnectionManager(
            api_key='k',
            base_url='http://x',
            config=OpenAIConnectionConfig(retryable_status=frozenset({400})),
        )
        assert mgr._is_retryable(LLMHTTPError('x', status_code=400)) is True


class TestCalcDelay:
    """指数退避延迟计算"""

    def test_no_jitter(self) -> None:
        mgr = ConnectionManager(
            api_key='k',
            base_url='http://x',
            config=OpenAIConnectionConfig(jitter=False),
        )
        assert mgr._calc_delay(1) == 1.0
        assert mgr._calc_delay(2) == 2.0
        assert mgr._calc_delay(3) == 4.0

    def test_capped_by_max_delay(self) -> None:
        mgr = ConnectionManager(
            api_key='k',
            base_url='http://x',
            config=OpenAIConnectionConfig(jitter=False, max_delay=3.0),
        )
        assert mgr._calc_delay(10) == 3.0

    def test_jitter_within_bounds(self) -> None:
        mgr = ConnectionManager(api_key='k', base_url='http://x')
        for attempt in (1, 2, 3):
            delay = mgr._calc_delay(attempt)
            base = min(1.0 * (2 ** (attempt - 1)), 30.0)
            assert base * 0.75 <= delay <= base * 1.25


class TestSessionManagement:
    """会话懒初始化与关闭"""

    def test_initial_session_none(self) -> None:
        mgr = ConnectionManager(api_key='k', base_url='http://x')
        assert mgr._session is None

    @pytest.mark.asyncio
    async def test_ensure_session_creates_once(self) -> None:
        mgr = ConnectionManager(api_key='k', base_url='https://api.example.com/v1')
        mock_session = MagicMock()
        mock_session.closed = False
        with patch(
            'infinity_agent.clients.open_ai.connection.aiohttp.ClientSession',
            return_value=mock_session,
        ) as mock_cls:
            s1 = await mgr.ensure_session()
            s2 = await mgr.ensure_session()
        assert s1 is mock_session
        assert s2 is mock_session
        assert mock_cls.call_count == 1
        # 验证认证头
        headers = mock_cls.call_args.kwargs['headers']
        assert headers['Authorization'] == 'Bearer k'

    @pytest.mark.asyncio
    async def test_base_url_trailing_slash_stripped(self) -> None:
        mgr = ConnectionManager(api_key='k', base_url='https://api.example.com/v1/')
        assert mgr._base_url == 'https://api.example.com/v1'

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        mgr = ConnectionManager(api_key='k', base_url='http://x')
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        with patch(
            'infinity_agent.clients.open_ai.connection.aiohttp.ClientSession',
            return_value=mock_session,
        ):
            await mgr.ensure_session()
        await mgr.close()
        mock_session.close.assert_awaited_once()
        assert mgr._session is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self) -> None:
        mgr = ConnectionManager(api_key='k', base_url='http://x')
        await mgr.close()
        await mgr.close()


class TestDoRequest:
    """单次 HTTP 请求与状态码处理"""

    def _build_manager(self) -> ConnectionManager:
        return ConnectionManager(api_key='k', base_url='https://api.example.com/v1')

    def _mock_resp(self, status: int = 200, body: str = '') -> MagicMock:
        resp = MagicMock()
        resp.status = status
        resp.text = AsyncMock(return_value=body)
        resp.close = MagicMock()
        resp.content.readline = AsyncMock(return_value=b'')
        return resp

    @pytest.mark.asyncio
    async def test_success_returns_response(self) -> None:
        mgr = self._build_manager()
        resp = self._mock_resp(status=200)
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = AsyncMock(return_value=resp)
        with patch(
            'infinity_agent.clients.open_ai.connection.aiohttp.ClientSession',
            return_value=mock_session,
        ):
            result = await mgr._do_request('chat/completions', _make_request())
        assert result is resp
        # 校验请求 URL 与 payload
        url = mock_session.post.call_args.args[0]
        assert url == 'https://api.example.com/v1/chat/completions'

    @pytest.mark.asyncio
    async def test_timeout_raises_network_error(self) -> None:
        mgr = self._build_manager()
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = AsyncMock(side_effect=asyncio.TimeoutError)
        with patch(
            'infinity_agent.clients.open_ai.connection.aiohttp.ClientSession',
            return_value=mock_session,
        ):
            with pytest.raises(LLMNetworkError):
                await mgr._do_request('chat/completions', _make_request())

    @pytest.mark.asyncio
    async def test_client_error_raises_network_error(self) -> None:
        import aiohttp

        mgr = self._build_manager()
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = AsyncMock(side_effect=aiohttp.ClientError('conn refused'))
        with patch(
            'infinity_agent.clients.open_ai.connection.aiohttp.ClientSession',
            return_value=mock_session,
        ):
            with pytest.raises(LLMNetworkError):
                await mgr._do_request('chat/completions', _make_request())

    @pytest.mark.asyncio
    async def test_http_error_generic(self) -> None:
        mgr = self._build_manager()
        resp = self._mock_resp(status=400, body='{"error": {"type": "invalid_request_error"}}')
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = AsyncMock(return_value=resp)
        with patch(
            'infinity_agent.clients.open_ai.connection.aiohttp.ClientSession',
            return_value=mock_session,
        ):
            with pytest.raises(LLMHTTPError) as exc_info:
                await mgr._do_request('chat/completions', _make_request())
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_http_error_content_filter(self) -> None:
        mgr = self._build_manager()
        resp = self._mock_resp(status=400, body='{"error": {"type": "content_filter"}}')
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = AsyncMock(return_value=resp)
        with patch(
            'infinity_agent.clients.open_ai.connection.aiohttp.ClientSession',
            return_value=mock_session,
        ):
            with pytest.raises(LLMContentFilterError) as exc_info:
                await mgr._do_request('chat/completions', _make_request())
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_http_error_context_length(self) -> None:
        mgr = self._build_manager()
        resp = self._mock_resp(
            status=400, body='{"error": {"type": "context_length_exceeded"}}'
        )
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = AsyncMock(return_value=resp)
        with patch(
            'infinity_agent.clients.open_ai.connection.aiohttp.ClientSession',
            return_value=mock_session,
        ):
            with pytest.raises(LLMContextLengthError) as exc_info:
                await mgr._do_request('chat/completions', _make_request())
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_http_error_non_json_body(self) -> None:
        mgr = self._build_manager()
        resp = self._mock_resp(status=500, body='<html>Internal Server Error</html>')
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = AsyncMock(return_value=resp)
        with patch(
            'infinity_agent.clients.open_ai.connection.aiohttp.ClientSession',
            return_value=mock_session,
        ):
            with pytest.raises(LLMServerError):
                await mgr._do_request('chat/completions', _make_request())


class TestRequestContextManager:
    """连接级重试逻辑"""

    def _build_manager(self, **kwargs) -> ConnectionManager:
        return ConnectionManager(
            api_key='k',
            base_url='https://api.example.com/v1',
            config=OpenAIConnectionConfig(jitter=False, **kwargs),
        )

    def _mock_resp(self, status: int = 200, body: str = '') -> MagicMock:
        resp = MagicMock()
        resp.status = status
        resp.text = AsyncMock(return_value=body)
        resp.close = MagicMock()
        resp.content.readline = AsyncMock(return_value=b'')
        return resp

    @pytest.mark.asyncio
    async def test_success_first_try(self) -> None:
        mgr = self._build_manager()
        resp = self._mock_resp(status=200)
        with patch.object(mgr, '_do_request', new=AsyncMock(return_value=resp)):
            async with mgr.request('chat/completions', _make_request()) as lines:
                async for _ in lines:
                    pass
        resp.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_then_success(self) -> None:
        mgr = self._build_manager(max_retries=2)
        resp_ok = self._mock_resp(status=200)
        do_request = AsyncMock(side_effect=[LLMServerError('x', status_code=500), resp_ok])
        with patch.object(mgr, '_do_request', new=do_request), patch.object(
            mgr, '_backoff', new=AsyncMock()
        ) as backoff:
            async with mgr.request('chat/completions', _make_request()) as lines:
                async for _ in lines:
                    pass
        assert do_request.await_count == 2
        backoff.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exhaust_retries_raises_last_error(self) -> None:
        mgr = self._build_manager(max_retries=2)
        err = LLMServerError('x', status_code=500)
        with patch.object(mgr, '_do_request', new=AsyncMock(side_effect=err)), patch.object(
            mgr, '_backoff', new=AsyncMock()
        ):
            with pytest.raises(LLMServerError):
                async with mgr.request('chat/completions', _make_request()) as lines:
                    async for _ in lines:
                        pass

    @pytest.mark.asyncio
    async def test_non_retryable_error_raised_immediately(self) -> None:
        mgr = self._build_manager(max_retries=3)
        err = LLMHTTPError('x', status_code=400)
        with patch.object(mgr, '_do_request', new=AsyncMock(side_effect=err)), patch.object(
            mgr, '_backoff', new=AsyncMock()
        ) as backoff:
            with pytest.raises(LLMHTTPError):
                async with mgr.request('chat/completions', _make_request()) as lines:
                    async for _ in lines:
                        pass
        backoff.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_backoff_closes_session_on_network_error(self) -> None:
        """网络错误时退避会先关闭失效会话"""
        mgr = self._build_manager(max_retries=1)
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        mgr._session = mock_session
        with patch('asyncio.sleep', new=AsyncMock()):
            await mgr._backoff(1, LLMNetworkError('x'))
        mock_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_backoff_http_error_keeps_session(self) -> None:
        """HTTP 错误退避不关闭会话"""
        mgr = self._build_manager(max_retries=1)
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        mgr._session = mock_session
        with patch('asyncio.sleep', new=AsyncMock()):
            await mgr._backoff(1, LLMServerError('x', status_code=500))
        mock_session.close.assert_not_awaited()


class TestReadline:
    """逐行读取"""

    @pytest.mark.asyncio
    async def test_reads_lines_until_eof(self) -> None:
        mgr = ConnectionManager(api_key='k', base_url='http://x')
        resp = MagicMock()
        resp.content.readline = AsyncMock(
            side_effect=[b'data: hello\n', b'data: world\n', b'']
        )
        lines = [line async for line in mgr._readline(resp)]
        assert lines == [b'data: hello\n', b'data: world\n']
