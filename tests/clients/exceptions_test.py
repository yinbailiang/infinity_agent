"""LLM 异常层级与 build_http_error 测试。"""

import pytest

from infinity_agent.clients.exceptions import (
    LLMAuthError,
    LLMConfigError,
    LLMContentFilterError,
    LLMContextLengthError,
    LLMError,
    LLMHTTPError,
    LLMInsufficientBalanceError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMRequestError,
    LLMServerError,
    LLMStreamError,
    build_http_error,
)


class TestExceptionHierarchy:
    """异常继承关系"""

    def test_request_errors_are_llm_error(self) -> None:
        assert issubclass(LLMRequestError, LLMError)
        assert issubclass(LLMNetworkError, LLMRequestError)
        assert issubclass(LLMHTTPError, LLMRequestError)
        assert issubclass(LLMStreamError, LLMRequestError)
        assert issubclass(LLMConfigError, LLMError)

    def test_http_error_subtypes(self) -> None:
        assert issubclass(LLMAuthError, LLMHTTPError)
        assert issubclass(LLMInsufficientBalanceError, LLMHTTPError)
        assert issubclass(LLMRateLimitError, LLMHTTPError)
        assert issubclass(LLMServerError, LLMHTTPError)
        assert issubclass(LLMContentFilterError, LLMHTTPError)
        assert issubclass(LLMContextLengthError, LLMHTTPError)


class TestLLMRequestError:
    """请求错误的附加字段"""

    def test_fields_captured(self) -> None:
        original = ConnectionError('boom')
        err = LLMRequestError(
            message='request failed',
            original_error=original,
            status_code=500,
            response_body='{"error": "x"}',
        )
        assert err.original_error is original
        assert err.status_code == 500
        assert err.response_body == '{"error": "x"}'
        assert str(err) == 'request failed'

    def test_fields_optional(self) -> None:
        err = LLMRequestError('msg')
        assert err.original_error is None
        assert err.status_code is None
        assert err.response_body is None


class TestBuildHttpError:
    """按状态码构建 HTTP 异常"""

    @pytest.mark.parametrize(
        'status,expected',
        [
            (401, LLMAuthError),
            (403, LLMAuthError),
            (402, LLMInsufficientBalanceError),
            (429, LLMRateLimitError),
            (500, LLMServerError),
            (502, LLMServerError),
            (503, LLMServerError),
            (599, LLMServerError),
            (400, LLMHTTPError),
            (404, LLMHTTPError),
            (418, LLMHTTPError),
        ],
    )
    def test_status_mapping(self, status: int, expected: type) -> None:
        err = build_http_error(status_code=status)
        assert isinstance(err, expected)

    def test_default_message(self) -> None:
        err = build_http_error(status_code=500)
        assert str(err) == 'HTTP 500'

    def test_custom_message_and_fields(self) -> None:
        original = ValueError('oops')
        err = build_http_error(
            status_code=429,
            message='rate limited',
            original_error=original,
            response_body='{"error": "too many"}',
        )
        assert str(err) == 'rate limited'
        assert err.original_error is original
        assert err.status_code == 429
        assert err.response_body == '{"error": "too many"}'


class TestStreamError:
    """流式解析错误"""

    def test_message_and_original(self) -> None:
        original = ValueError('bad json')
        err = LLMStreamError('stream broken', original_error=original)
        assert err.original_error is original
