import ast
import asyncio
import base64
import logging
import math
import os
import subprocess
from typing import Annotated, Optional

from pydantic import Field

from infinity_agent import (
    Message,
    Messages,
    StreamChunk,
    TextChunk,
    ToolCall,
    ToolCallCompleteChunk,
)
from infinity_agent.clients import OpenAIConfig, OpenAIRequestConfig, create_client
from infinity_agent.clients.exceptions import LLMNetworkError
from infinity_agent.models.tools import Raw
from infinity_agent.tools import ToolRegistry

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='app.log',
    filemode='a',
)

logger = logging.getLogger(__name__)


tools = ToolRegistry()


@tools.tool()
def powershell(command: str, timeout: Annotated[float, Field(ge=1, le=300)] = 30) -> dict[str, str | int]:
    """执行 PowerShell 命令并返回 stdout、stderr 和返回码。"""
    preamble = "$ProgressPreference = 'SilentlyContinue'; "
    encoded = base64.b64encode((preamble + command).encode('utf-16-le')).decode()
    try:
        result = subprocess.run(
            ['powershell.exe', '-NoProfile', '-EncodedCommand', encoded],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
        )
        return {
            'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip() if result.returncode != 0 else '',
            'returncode': result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {'stdout': '', 'stderr': '命令执行超时', 'returncode': -1}


# ------------------------------------------------------------------
# AST 白名单 — 仅允许安全节点与函数
# ------------------------------------------------------------------

_SAFE_NODES: frozenset[type[ast.AST]] = frozenset(
    {
        ast.Expression,
        ast.Constant,
        ast.UnaryOp,
        ast.UAdd,
        ast.USub,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.Call,
        ast.Name,
        ast.Load,
    }
)

_SAFE_BUILTINS: dict[str, object] = {name: getattr(math, name) for name in dir(math) if not name.startswith('_')} | {
    'abs': abs,
    'round': round,
    'min': min,
    'max': max,
    'pow': pow,
}


def _validate_ast(node: ast.AST) -> None:
    """递归校验 AST 节点，拒绝白名单外的任何操作。"""
    if type(node) not in _SAFE_NODES:
        raise ValueError(f'不允许的操作: {type(node).__name__}')

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError('不允许的调用形式')
        if node.func.id not in _SAFE_BUILTINS:
            raise ValueError(f'不允许的函数: {node.func.id}')
    elif isinstance(node, ast.Name):
        if node.id not in _SAFE_BUILTINS:
            raise ValueError(f'不允许的变量: {node.id}')

    for child in ast.iter_child_nodes(node):
        _validate_ast(child)


@tools.tool()
def calc(formula: str = Field(max_length=256), max_out: int = Field(default=32, ge=32, le=256)) -> Raw:
    """python eval 计算器"""
    formula = formula.strip()

    try:
        tree = ast.parse(formula, mode='eval')
    except SyntaxError as e:
        raise ValueError(f'公式语法错误: {e}')

    _validate_ast(tree)

    try:
        result: object = eval(
            compile(tree, '<calc>', 'eval'),
            {'__builtins__': {}},
            _SAFE_BUILTINS,
        )
    except Exception as e:
        raise ValueError(f'计算错误: {e}')

    result = str(result)

    if len(result) > max_out:
        result = result[:max_out] + '\n输出过长，已截断'
    return result


async def main() -> None:
    key = os.getenv('DEEPSEEK_API_KEY')
    if not key:
        raise RuntimeError('未设置环境变量 DEEPSEEK_API_KEY')
    config = OpenAIConfig(model='deepseek-v4-pro', api_key=key, base_url='https://api.deepseek.com/v1')
    for definition in tools.definitions:
        print(definition.model_dump_json(exclude_none=True))
    messages: Messages = [Message.system('你是一个助手。')]
    async with create_client(config) as client:
        while True:
            user_input: str = await asyncio.to_thread(lambda: input('>'))
            match user_input:
                case '/quit':
                    break
                case _:
                    pass
            messages.append(Message.user(user_input))

            while True:
                chunks: list[StreamChunk] = []
                tool_calls_list: Optional[list[ToolCall]] = None

                request_config = OpenAIRequestConfig(tools=tools.definitions)
                for attempt in range(1, 4):
                    try:
                        async for chunk in client.stream_chat(messages, request_config):
                            chunks.append(chunk)
                            match chunk:
                                case TextChunk():
                                    print(chunk.text, end='')
                                case ToolCallCompleteChunk():
                                    tool_calls_list = chunk.tool_calls
                                case _:
                                    pass
                        break
                    except LLMNetworkError as e:
                        if attempt == 3:
                            raise
                        wait = 2 ** (attempt - 1)
                        logger.warning('网络错误，%ds 后重试 (%d/3): %s', wait, attempt, e)
                        print(f'\n[网络波动，{wait}s 后重试...]')
                        chunks.clear()
                        await asyncio.sleep(wait)

                print()

                messages.append(client.chunks_to_message(chunks))

                if tool_calls_list is not None:
                    for call in tool_calls_list:
                        print(call.function.model_dump_json())
                    tool_results: Messages = await tools.invoke_all(tool_calls_list)
                    for result in tool_results:
                        print(result.content)
                    messages.extend(tool_results)
                else:
                    break


if __name__ == '__main__':
    asyncio.run(main())
