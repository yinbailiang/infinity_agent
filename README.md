# InfinityAgent — Async LLM Client

[![Test](https://github.com/yinbailiang/infinity_agent/actions/workflows/test.yml/badge.svg)](https://github.com/yinbailiang/infinity_agent/actions/workflows/test.yml)
[![Pyright](https://img.shields.io/badge/pyright-strict-blue)](ENGINEERING.md)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE.md)
[![Supported Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://pypi.org/project/infinity_agent/)

**强类型、可扩展的 LLM 异步客户端库 — 支持 OpenAI 兼容 API、流式对话、工具调用。**

## ✨ Features

| 类别 | 能力 |
| - | - |
| 类型安全 | Pydantic 负载校验 · pyright **strict** |
| 多 Provider | 工厂注册模式，轻松扩展新 LLM 服务商 |
| OpenAI 兼容 | 完整实现 OpenAI Chat Completions 流式 API |
| 工具调用 | ToolRegistry 自动从 Python 函数推断 Tool Schema |
| 流式对话 | `stream_chat()` 异步生成器，支持 Text / ToolCall / Usage chunk |
| 多模态消息 | 文本 + 图片混合内容：user 消息与 **tool 结果**均可携带，支持 http(s) 与 `data:` URL |
| 连接管理 | 指数退避重试 + 抖动 + 连接池复用 |

## 📦 Installation

```bash
pip install infinity_agent
```

## 🚀 Quick Start

```python
from infinity_agent import (
    Message,
    Messages,
    OpenAIConfig,
    create_client,
    ToolRegistry,
)

# 创建客户端
config = OpenAIConfig(
    model='gpt-4o-mini',
    api_key='sk-...',
)
client = create_client(config)

# 流式对话
messages = Messages(
    [
        Message.system('你是一个有用的助手'),
        Message.user('你好'),
    ]
)

async with client:
    async for chunk in client.stream_chat(messages):
        print(chunk)
```

### 🖼 多模态与图片工具结果

`MultiModalContent` 可挂在 user 消息上，也可以作为 **tool 消息的结果**（DeepSeek 等
provider 支持工具返回图片）。`text_part` / `image_part` / `file_part` 是便捷构造器：
`image_part` 的 `url` 支持 http(s) 链接与 `data:image/...;base64,` 内联，`detail` 可选
`auto` / `low` / `high` / `original`；`file_part` 走 Files API 的 `file_id` 或内联
`file_data`（二者互斥）：

```python
from infinity_agent import Message, MultiModalContent

# 用户发图
messages = [
    Message.user(
        [
            MultiModalContent.text_part('这张图里有什么？'),
            MultiModalContent.image_part('https://example.com/a.png', detail='low'),
        ]
    ),
]

# 复用 Files API 上传过的文件（file_id 与 file_data 二选一）
messages[0].content.append(MultiModalContent.file_part(file_id='file-api-xxxxxxxx'))

# 工具返回图片（紧跟在 assistant 的 tool_calls 之后）
messages.append(
    Message.tool(
        [
            MultiModalContent.text_part('截图如下'),
            MultiModalContent.image_part('data:image/png;base64,AAAA'),
        ],
        tool_call_id='call_1',
    )
)
```

约束（由模型验证器保证）：`system` 与 `assistant` 消息只能是纯文本，`tool` 消息必须
同时给出 `content` 与 `tool_call_id`。序列化会省略空字段，避免把
`{"image_url": null}` 送进请求体。

## 🧱 Architecture

```
infinity_agent/
├── clients/          # 客户端抽象 + provider 实现
│   ├── base.py       #   LLMClient 抽象基类
│   ├── config.py     #   ConnectionConfig / LLMConfig
│   ├── exceptions.py #   异常层级
│   ├── provider.py   #   工厂注册表
│   └── open_ai/      #   OpenAI 兼容实现
├── models/           # 消息 / Chunk / 工具定义
│   ├── messages.py   #   Message, Messages, ToolCall
│   ├── chunks.py     #   StreamChunk 子类
│   └── tools.py      #   ToolDefinition, ParameterProperty
└── tools/            # 工具注册与类型推断
    ├── registry.py   #   ToolRegistry
    └── introspection.py  #   函数 → ToolDefinition 自动推断
```

## 📄 License

[MIT](LICENSE.md)

## Part of InfinitySystem

![icon](docs/res/infinity_icon/256x256.png)
