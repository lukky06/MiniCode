# MiniCode Usage

MiniCode 是一个运行在本地代码仓库中的 Coding Agent。它可以读取和搜索代码、修改文件、执行受控命令、运行聚焦验证，并保存 Session、Checkpoint 和 Trace。

## 1. 环境要求

- Python 3.11+
- Git
- Node.js 22.19+（交互 TUI）
- 一个受支持模型 Provider 的 API Key，或本地 Ollama

## 2. 安装

```bash
git clone https://github.com/lukky06/MiniCode.git
cd MiniCode
python -m pip install .
```

开发安装：

```bash
python -m pip install -e ".[dev]"
```

## 3. 配置模型

以 DeepSeek 为例。

Windows PowerShell：

```powershell
$env:DEEPSEEK_API_KEY = "..."
minicode --provider deepseek
```

Linux / macOS：

```bash
export DEEPSEEK_API_KEY="..."
minicode --provider deepseek
```

MiniCode 当前支持 Qwen、DeepSeek、Kimi、OpenAI、Anthropic 和 Ollama。

常用默认配置可以写入：

```text
~/.minicode/config.toml
```

示例：

```toml
provider = "deepseek"
model = "deepseek-chat"
permission_mode = "workspace-write"
approval_policy = "on-request"
sandbox = "local"
```

命令行显式参数优先于用户配置。

## 4. 启动交互 Session

进入任意 Git 仓库：

```bash
cd /path/to/repository
minicode
```

也可以直接带一个初始任务：

```bash
minicode "解释这个项目的核心执行流程"
minicode "定位当前失败测试并修复"
```

交互模式会保留当前 Session，可以继续追加任务。

常见任务示例：

```text
解释主要模块和调用关系
定位这个异常的根因
给这个边界条件补一个测试
修改这段代码并运行聚焦验证
检查当前 Git Diff 是否存在明显问题
```

## 5. 一次性执行

脚本或非交互环境使用 `exec`：

```bash
minicode exec "解释这个仓库"
minicode exec "修复失败测试并运行最小验证"
```

需要只读执行时：

```bash
minicode exec "分析这个模块" --no-write
```

## 6. Session 与 Run

MiniCode 区分 Session 和 Run：

- Session 表示一段持续的多轮对话；
- Run 表示一次具体任务执行。

继续当前工作区最近 Session：

```bash
minicode --continue
```

或者：

```bash
minicode resume
```

打开指定 Session：

```bash
minicode resume <session_id>
```

查看 Session 和 Run：

```text
/sessions
/runs
/status
```

Session 支持命名与分叉：

```text
/rename parser-refactor
/fork
```

## 7. Checkpoint 与 Recovery

当一个 Run 因中断、模型错误或其他可恢复原因停止时，可以从 Checkpoint 恢复：

```bash
minicode recover <run_id>
```

恢复前会重新检查工作区状态，避免直接在已经发生外部修改的 Workspace 上继续旧执行状态。

查看 Trace 或 Report：

```bash
minicode trace [run_id]
minicode report [run_id]
```

交互模式下也可以使用：

```text
/trace
/recover
/report
```

## 8. Permission

MiniCode 提供三种 Permission Mode：

```text
read-only
workspace-write
full-access
```

`read-only` 适合仓库分析和代码解释；`workspace-write` 允许在当前 Workspace 内执行受控文件修改；`full-access` 放宽部分已经通过确定性安全策略的非命令副作用。

Approval Policy 控制需要人工确认的操作：

```text
on-request
never
```

查看当前权限：

```text
/permissions
```

Workspace 边界、敏感路径和危险命令策略始终生效。

## 9. Plan Mode

Plan Mode 用于先分析仓库并形成执行计划：

```text
/plan on
```

输入需要规划的任务后，模型只使用低风险只读能力进行探索。

查看状态：

```text
/plan status
```

退出：

```text
/plan off
```

交互 TUI 在规划完成后可以继续规划、结束规划，或者基于当前 Session 中已经形成的计划进入执行阶段。

## 10. Runtime Steering

Agent 正在运行时，可以继续输入补充要求。

新输入会在当前完整 Tool Batch 结束后进入同一 Canonical History，供下一次模型调用使用。

这适合在长任务中补充约束，例如：

```text
不要修改 public API
只运行这个模块的测试
先检查现有实现再决定是否重构
```

取消当前运行可以使用 Esc 或 Ctrl+C。

## 11. Context

查看最近 Run 的上下文占用：

```text
/context
```

手动压缩当前 Session 的模型可见历史：

```text
/compact
```

也可以指定关注点：

```text
/compact focus on parser migration
```

压缩只调整模型可见投影，完整 Session 历史仍会保留用于持久化和审计。

## 12. Memory

MiniCode 支持仓库级长期记忆，用于保存需要跨 Session 复用的约束和项目背景。

查看 Memory：

```text
/memory
```

Memory 使用精简索引和按需 Topic 读取，避免把全部长期信息默认塞进每一次模型请求。

## 13. Review

可以对当前 Git Diff 发起只读 Review：

```text
/review
```

也可以指定关注点：

```text
/review focus on recovery correctness
```

Review 使用只读能力，不直接修改 Workspace。

## 14. 常用 Slash Commands

```text
/status
/sessions
/runs
/context
/compact [focus]
/memory
/trace
/recover
/report
/diff
/model
/permissions
/plan [on|off|status]
/rename <name>
/fork [run]
/review [focus]
/help
/exit
```

## 15. 开发验证

运行常用聚焦测试：

```bash
python -m pytest tests/test_policy.py tests/test_agent_loop.py -q
```

检查 Python 源码是否可以正常编译：

```bash
python -m compileall -q minicode_harness tests
```

更完整的实现结构见 [Architecture](./architecture.md)。