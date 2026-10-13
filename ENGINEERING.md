# 工程质量

InfinityAgent 是一个工程纪律驱动的项目。以下是所有质量保障措施的完整说明。

---

## 类型安全

### Pyright Strict

生产代码全量启用 [Pyright](https://github.com/microsoft/pyright) **strict** 模式——Python 生态中最严格的类型检查级别；pre-commit 以 `--level error` 对 `src/` 强制执行，任何类型错误都会阻止提交。

```toml
# pyproject.toml
[tool.pyright]
typeCheckingMode = "strict"
reportMissingTypeStubs = false
reportMissingImports = "error"
venvPath = "."
venv = ".venv"
```

**零遮蔽是项目标准**（见 `.agents/FOR_AGENT.md` 铁律）。当前 `src/` 存在 9 处存量豁免，全部限定具体错误码，集中于 Pydantic 继承收窄、类变量覆写与动态建模三类框架性摩擦，待专项清理。

| 指标 | 数值 |
| - | - |
| 类型检查模式 | `strict`（全项目） |
| pre-commit 检查范围 | `src/`（`--level error`） |
| 生产代码遮蔽注释 | 9 处存量（目标 **0**） |

### Ruff Lint

仅忽略两条规则，均有明确理由：

```toml
[tool.ruff.lint]
ignore = [
    "E501",     # 行长度由 formatter 处理
    "ASYNC109", # async 函数的 timeout 参数是有意的 API 设计
]
```

启用规则组：`E`（pycodestyle）、`F`（pyflakes）、`I`（isort）、`ASYNC`（异步最佳实践）、`PLE`（pylint 错误）。检查范围为 `src/`。

---

## 文档覆盖

使用 [interrogate](https://github.com/econchick/interrogate) 强制公开 API 的 docstring 覆盖。

```toml
[tool.interrogate]
fail-under = 60
ignore-init-method = true
ignore-init-module = true
ignore-magic = true
ignore-private = true
ignore-semiprivate = true
ignore-nested-functions = true
exclude = ["tests", "benchmarks", "experimental", "docs"]
```

| 指标 | 数值 |
| - | - |
| 公开 API docstring 覆盖 | **89.6%** |
| 最低阈值 | 60% |
| 工程文档 | `README.md` · `AGENT.md` · `.agents/` 引导体系（12 篇 Markdown） |

---

## 测试

### 覆盖率

```text
300 passed · 0 failed · 0.52s
总覆盖率: 95%（1165 statements / 53 missed）
```

27 个模块中 18 个达到 **100%** 覆盖，最低模块 80%（`model_builder.py`）。pre-commit 门禁要求不低于 80%，CI 要求不低于 90%。

### 架构

```text
tests/
├── conftest.py             共享 fixtures 与测试工具函数
├── clients/                镜像 src/clients/
│   ├── base_test.py
│   ├── exceptions_test.py
│   ├── provider_test.py
│   ├── open_ai/
│   │   ├── aggregation_test.py
│   │   ├── client_test.py
│   │   ├── config_test.py
│   │   ├── connection_test.py
│   │   ├── request_models_test.py
│   │   └── response_models_test.py
│   └── deepseek/
│       ├── client_test.py
│       └── config_test.py
├── models/
│   ├── chunks_test.py
│   ├── messages_test.py
│   └── tools_test.py
└── tools/
    ├── registry_test.py
    └── introspectionist/
        ├── convert_test.py
        └── introspection_test.py
```

测试结构完全镜像源码结构。`pytest-asyncio` + `asyncio_mode = "auto"`，异步测试零样板代码。

### 运行

```bash
uv run pytest --cov=infinity_agent -v    # 全量 + 覆盖率
uv run pytest tests/clients/ -v          # 客户端层
uv run pytest tests/tools/ -v            # 工具层
```

---

## Pre-commit 门禁

> **需要 [uv](https://docs.astral.sh/uv/)**。钩子通过 `uv run` 调用工具链，
> 确保在所有开发环境中使用一致的 venv 和依赖版本。

每次提交自动执行，全部通过才允许 commit：

| 钩子 | 范围 | 作用 |
| - | - | - |
| `ruff-check` | `^src/` | Lint + 自动修复 |
| `ruff-format` | `^src/` | 格式化 |
| `pyright` | `^src/` | `--level error` 类型检查 |
| `interrogate` | `^src/` | docstring 覆盖 >= 60% |
| `pytest-cov` | 全量 | 测试 + 覆盖率 >= 80% |
| `interrogate-badge` | `src/` | docstring 覆盖率徽章生成 |
| `agent-kit-check` | 全量 | `AGENT.md` 与 `.agents/` 防漂移校验 |
| `check-ast` | 全项目 | Python 语法校验 |
| `check-toml/yaml/json` | 全项目 | 配置文件校验 |
| `check-merge-conflict` | 全项目 | 合并冲突残留 |
| `check-case-conflict` | 全项目 | 文件名大小写冲突 |
| `end-of-file-fixer` | 非 `.py` | 文件末尾换行 |
| `trailing-whitespace` | 非 `.py` | 行尾空白（`.py` 由 ruff 处理） |

```bash
uv run pre-commit run --all-files   # 手动全量运行
```

---

## 模块化

### KISS 原则

| 指标 | 数值 |
| - | - |
| 最大单文件 | ~309 行 (`model_builder.py`) |
| 模块平均 | ~104 行/文件 |
| 生产文件数 | 27 个 `.py` |

每个文件一个职责，打开瞬间看完。新增 Provider 通过继承既有客户端实现，不改动核心管线。

### 目录结构

```text
src/infinity_agent/
├── __init__.py                 公开 API 入口
├── clients/                    客户端抽象 + Provider 实现
│   ├── base.py                 LLMClient 抽象基类
│   ├── config.py               配置基类（ClientConfig / RequestConfig）
│   ├── exceptions.py           异常层级
│   ├── provider.py             工厂注册表（register_client / create_client）
│   ├── open_ai/
│   │   ├── aggregation.py      tool_calls 流式片段聚合
│   │   ├── client.py           OpenAI 兼容流式客户端
│   │   ├── config.py           OpenAIConfig 与请求构建
│   │   ├── connection.py       HTTP 连接管理（重试 / 退避 / SSE 解析）
│   │   ├── request_models.py   请求体模型
│   │   └── response_models.py  响应模型（StreamEvent / Delta / Choice）
│   └── deepseek/
│       ├── client.py           DeepSeek 特化客户端（思考内容）
│       ├── config.py           DeepSeekConfig 与请求构建
│       ├── request_models.py   DeepSeek 请求模型
│       └── response_models.py  响应模型（reasoning_content 扩展）
├── models/                     消息 / Chunk / 工具定义
│   ├── chunks.py               StreamChunk 体系（Text / Thinking / ToolCall / Usage）
│   ├── messages.py             Message / Messages / 多模态内容
│   └── tools.py                ToolDefinition 参数体系
└── tools/                      工具注册与类型推断
    ├── registry.py             ToolRegistry（收集定义、调度执行）
    └── introspectionist/
        ├── convert.py          JSON Schema 转换（$ref 内联解析等）
        ├── introspection.py    函数签名 → ToolDefinition 推断
        └── model_builder.py    动态 Pydantic 参数/返回值校验模型构建
```

---

## 版本策略

- **SemVer**：`MAJOR.MINOR.PATCH`
- 推送 `v*` 标签触发 [`.github/workflows/publish.yml`](.github/workflows/publish.yml)：构建产物 → GitHub Release → PyPI 发布

---

## 提交规范

遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/)：

```text
<type>(<scope>): <subject>
```

| type | 场景 |
| --- | --- |
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `docs` | 文档变更 |
| `test` | 测试补全 |
| `refactor` | 重构（不改行为） |
| `chore` | 版本号、依赖、构建工具 |

`scope` 对应模块名：`open_ai`、`deepseek`、`models`、`tools`、`convert`、`connection`、`model_builder` 等。

示例：

```text
feat(deepseek): add reasoning content support
fix(convert): keep empty array items schema
fix(connection): use certifi certificate bundle
test(models): cover multimodal message serialization
chore(version): v0.2.1
```

---

## CI

测试矩阵覆盖 Python 3.12 / 3.13 × 三大平台，每次 push / PR 自动运行 pre-commit 全量门禁与全量测试；推送 `v*` 标签触发发布流程。

```yaml
# .github/workflows/test.yml
uv run pytest --cov=infinity_agent --cov-report=term-missing --cov-fail-under=90 -v -m ""
```
