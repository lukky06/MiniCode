# MiniCode — Lightweight Local Coding Agent Harness

[![CI](https://github.com/lukky06/MiniCode/actions/workflows/ci.yml/badge.svg)](https://github.com/lukky06/MiniCode/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.1.0-555555)](./pyproject.toml)

MiniCode 是一个运行在本地代码仓库中的轻量级 Coding Agent Harness。

项目基于模型原生 Tool Calling 构建有界 Agent Loop，由模型决定下一步操作，Harness 负责工具执行、上下文治理、权限控制、失败恢复和运行记录。它可以在受控工作区内完成代码检索、分析、修改和聚焦验证，并保留可审计的执行链路。

## 核心能力

- **Agent Loop**：支持多轮 Tool Calling，并通过 Step、Tool Call 和超时预算限制无效循环。
- **代码工具**：提供文件读取、搜索、修改、补丁和命令执行等基础能力。
- **上下文治理**：维护 Canonical Message History，对大结果和长会话进行有界压缩。
- **安全执行**：文件操作受 Workspace 边界保护，命令经过 Policy、Permission 和 Approval 检查。
- **恢复机制**：支持 Session、Checkpoint、Recovery、Execution Journal 和 stale-write 防护。
- **交互能力**：提供 TypeScript TUI、Plan Mode、运行中 Steering、Session Fork 和只读 Review。
- **扩展能力**：支持 Repository Memory、Skills、只读 Subagent、Worktree Worker 和 MCP。
- **可观测与评测**：记录模型调用、Tool Result、验证结果和 Trace，并提供 Benchmark / SWE-bench 适配。

## 快速开始

环境要求：

- Python 3.11+
- Git
- Node.js 22.19+（交互 TUI）
- 一个受支持模型 Provider 的 API Key，或本地 Ollama

安装：

```bash
git clone https://github.com/lukky06/MiniCode.git
cd MiniCode
python -m pip install .
```

以 DeepSeek 为例：

```powershell
$env:DEEPSEEK_API_KEY = "..."
minicode --provider deepseek
```

Linux / macOS：

```bash
export DEEPSEEK_API_KEY="..."
minicode --provider deepseek
```

进入任意 Git 仓库后即可使用：

```bash
minicode
minicode "解释这个项目的核心执行流程"
minicode "定位这个失败测试的原因并修复"
```

一次性非交互执行：

```bash
minicode exec "解释这个仓库"
```

## 架构

```text
CLI / TypeScript TUI
        ↓
   RunExecutor
        ↓
    AgentLoop
      ├─ ModelStepRunner
      ├─ Context
      ├─ ToolBatchExecutor
      │    └─ ToolRuntime
      │         ├─ ToolRegistry
      │         ├─ Policy / Permission / Approval
      │         ├─ ToolReuseTracker
      │         └─ Execution Journal
      └─ RunLifecycle
           ├─ Session / Checkpoint
           ├─ Recovery
           └─ Trace
```

核心流程保持为：

```text
用户任务
→ 模型决策
→ Tool Calling
→ 工具执行与结果回填
→ 模型继续决策
→ 聚焦验证
→ 最终回答
```

## 模型后端

当前支持 Qwen、DeepSeek、Kimi、OpenAI、Anthropic 和 Ollama，通过统一 `ModelClient` 与 Provider Adapter 接入 Agent Loop。

常用默认配置可以放在 `~/.minicode/config.toml`：

```toml
provider = "deepseek"
model = "deepseek-chat"
permission_mode = "workspace-write"
approval_policy = "on-request"
sandbox = "local"
```

## 开发

安装开发依赖：

```bash
python -m pip install -e ".[dev]"
```

运行聚焦测试：

```bash
python -m pytest tests/test_policy.py tests/test_agent_loop.py -q
python -m compileall -q minicode_harness tests
```

更多说明见：

- [Architecture](./docs/architecture.md)
- [Usage](./docs/usage.md)

## License

本项目采用 [Apache License 2.0](LICENSE) 开源。
