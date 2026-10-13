# SKILL: 代码审查前检查

**触发**：提交代码或用户要求 pre-commit 检查。

## 步骤

1. `uv run ruff check .` — lint 检查
2. `uv run ruff format .` — 格式化（仅检查不写入时，为命令追加 `--check`）
3. **零遮蔽检查** — `src/infinity_agent/` 下不得出现任何 `# type:` 或 `# pyright` 注释
   - Windows: `Select-String -Path src\infinity_agent\**\*.py -Pattern "# type:|# pyright"` 应无匹配
   - Unix: `grep -r "# type:\|# pyright" src/infinity_agent/` 应无输出
4. `uv run pyright` — 类型检查（必须零错误）
5. `uv run pytest tests/ --cov=infinity_agent --cov-fail-under=80 -q` — 全量测试
6. `uv run interrogate src/infinity_agent/ --generate-badge docs/res/interrogate_badge.svg` — 公开 API docstring 覆盖率检查与徽章生成

> 如有问题，修复后从第 1 步重新全量运行。
