"""共享的 fixtures、测试用 Payload 与工具函数。"""

import asyncio
import logging
from typing import Any, Callable, List, Optional

import pytest

from infinity_agent.clients.open_ai.config import OpenAIConfig
from infinity_agent.models import Message, MessageRole, ToolCall

# 抑制日志噪音
logging.getLogger('infinity_agent').setLevel(logging.WARNING)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def openai_config() -> OpenAIConfig:
    """一个基础的 OpenAIConfig 实例。"""
    return OpenAIConfig(
        model='gpt-4o-mini',
        api_key='sk-test-key',
        base_url='https://api.example.com/v1',
    )


# ============================================================================
# 工具函数
# ============================================================================


def make_tool_call(
    name: str,
    arguments: Optional[dict[str, Any]] = None,
    tool_call_id: str = 'call_test_1',
) -> ToolCall:
    """构造一个 ToolCall 实例。"""
    return ToolCall(
        id=tool_call_id,
        function={
            'name': name,
            'arguments': arguments or {},
        },
    )


def make_messages(*roles: MessageRole) -> List[Message]:
    """根据角色列表构造一组最小消息。"""
    messages: List[Message] = []
    for role in roles:
        if role == MessageRole.SYSTEM:
            messages.append(Message.system('You are a helpful assistant.'))
        elif role == MessageRole.USER:
            messages.append(Message.user('Hello'))
        elif role == MessageRole.ASSISTANT:
            messages.append(Message.assistant('Hi'))
        elif role == MessageRole.TOOL:
            messages.append(Message.tool('ok', tool_call_id='call_1'))
    return messages


async def wait_for_condition(condition: Callable[[], bool], timeout: float = 1.0) -> None:
    """轮询等待条件成立，超时则抛出 AssertionError。"""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError('condition not met within timeout')
