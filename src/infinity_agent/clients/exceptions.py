"""LLM 通用异常层级"""

from typing import Optional


class LLMError(Exception):
    """LLM 基础异常"""

    pass


class LLMConfigError(LLMError):
    """配置/注册错误"""

    pass


class LLMRequestError(LLMError):
    """LLM 请求失败的基类"""

    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.original_error = original_error
        self.status_code = status_code
        self.response_body = response_body


class LLMNetworkError(LLMRequestError):
    """网络层错误：连接失败、超时、DNS 解析等"""

    pass


class LLMHTTPError(LLMRequestError):
    """HTTP 响应状态码非 200 的错误（基类）"""

    pass


class LLMAuthError(LLMHTTPError):
    """认证失败：401 或 403"""

    pass


class LLMInsufficientBalanceError(LLMHTTPError):
    """余额不足：402"""

    pass


class LLMRateLimitError(LLMHTTPError):
    """速率限制：429"""

    pass


class LLMServerError(LLMHTTPError):
    """服务端错误：5xx"""

    pass


class LLMContentFilterError(LLMHTTPError):
    """内容安全过滤"""

    pass


class LLMContextLengthError(LLMHTTPError):
    """上下文长度超出模型限制"""

    pass


class LLMStreamError(LLMRequestError):
    """流式响应解析错误"""

    pass


def build_http_error(
    status_code: int,
    message: Optional[str] = None,
    original_error: Optional[Exception] = None,
    response_body: Optional[str] = None,
) -> LLMHTTPError:
    """根据 HTTP 状态码构建对应的异常实例。"""
    if message is None:
        message = f'HTTP {status_code}'

    if status_code in (401, 403):
        return LLMAuthError(
            message=message,
            status_code=status_code,
            original_error=original_error,
            response_body=response_body,
        )
    elif status_code == 402:
        return LLMInsufficientBalanceError(
            message=message, status_code=status_code, original_error=original_error, response_body=response_body
        )
    elif status_code == 429:
        return LLMRateLimitError(
            message=message,
            status_code=status_code,
            original_error=original_error,
            response_body=response_body,
        )
    elif 500 <= status_code < 600:
        return LLMServerError(
            message=message,
            status_code=status_code,
            original_error=original_error,
            response_body=response_body,
        )
    else:
        return LLMHTTPError(
            message=message,
            status_code=status_code,
            original_error=original_error,
            response_body=response_body,
        )
