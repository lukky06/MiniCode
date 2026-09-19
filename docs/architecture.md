# MiniCode Architecture

MiniCode 是一个面向本地代码仓库的轻量级 Coding Agent Harness。它负责把模型推理、工具调用、代码修改、验证和运行状态组织成一条可控、可恢复、可审计的执行链路。

## 1. 整体结构

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
      │         └─ WorkspaceGuard
      └─ RunLifecycle
           ├─ Session / Checkpoint
           ├─ Recovery
           └─ Trace
```

一次用户请求对应一个 User Turn，一个 User Turn 内可以发生多次模型调用。模型负责决定下一步动作，Harness 负责提供上下文、执行工具、约束副作用并保存运行状态。

核心链路保持为：

```text
用户任务
→ 构造模型请求
→ 模型生成回答或 Tool Call
→ 工具校验与执行
→ Tool Result 回填
→ 模型继续决策
→ 聚焦验证
→ 最终回答
```

## 2. 有界 Agent Loop

`AgentLoop` 是一次任务执行的核心协调器。

它持续执行“模型调用 → 工具执行 → 结果回填”，直到模型给出最终回答，或者触发步数、工具次数、取消、审批拒绝、模型错误等停止条件。

运行时通过 Step、Tool Call 和超时预算限制长任务，避免 Agent 在失败命令、重复读取或无效探索中持续循环。

Agent Loop 本身只负责主流程控制，模型调用、工具批次、生命周期和恢复逻辑分别由独立模块处理，减少核心循环中的状态分支。

## 3. Canonical Messages

MiniCode 使用一条 Canonical Message History 作为模型可见的会话历史：

```text
User Message
Assistant Tool Call
Tool Result
Assistant Tool Call
Tool Result
Assistant Final Answer
```

工具证据直接通过 Tool Result 回填，不额外维护第二套 Repo Transcript、Evidence Pack 或模型可见执行状态。

这样可以保持 Provider 原生 Tool Calling 协议，同时降低重复上下文带来的 Token 消耗和状态同步成本。

## 4. Context Management

上下文由两部分负责：

- `ContextBuilder` 构造动态 System Prompt；
- `ContextPreparer` 根据模型上下文预算治理历史消息。

当上下文增长时，MiniCode 优先压缩可以重新获取的信息，例如较早的大型文件读取结果和冗长工具输出，并保留：

- 当前用户请求；
- 最近的关键工具证据；
- Assistant Tool Call 与 Tool Result 的协议配对；
- 已经发生且不能安全重放的副作用事实。

大型工具结果可以外置为 Artifact，模型上下文只保留有界摘要和定位信息。

在普通压缩不足时，运行时还可以执行受约束的语义摘要和上下文溢出恢复。

## 5. Repository Memory V3

Repository Memory 保存跨 Run 仍有价值的仓库知识。每个新的 top-level Run 先冻结 `memory_summary.md`、`MEMORY.md` 和当时可见的 immutable rollout summaries，当前 Run 后续始终读取这份 Snapshot。

模型默认只接收小型 `memory_summary.md`。需要详细信息时继续复用统一工具：

```text
search(source="memory", query="windows git")
read(source="memory", target="MEMORY.md")
read(source="memory", target="rollout_summaries/<run>--<slug>.md")
```

Snapshot 完成后，Runtime 会异步启动 repository 级 Memory Pipeline。`pipeline.lock` 使用跨进程非阻塞互斥，保证同一仓库同一时间只有一个 Pipeline 工作；抢锁失败直接跳过，不阻塞 Coding Run。另一个短临界区 `durable.lock` 只保护 durable Summary/Handbook/rollout summaries 的提交与 Run-start Snapshot 读取，避免并发进程冻结“旧 Summary + 新 Handbook”的混合视图；它不覆盖 Phase 1/Phase 2 模型调用。

```text
Earlier terminal Run
    ↓
Phase 1 extraction
    ↓
stage1/<run_id>.json
    ↓
Phase 2 consolidation
    ├─ raw_memories.md
    ├─ rollout_summaries/
    ├─ MEMORY.md
    └─ memory_summary.md
```

Phase 1 和 Phase 2 都是无 Tool 的独立模型请求。Phase 1 失败时该 Run 保持未处理，Phase 2 失败时不推进 consolidation cursor。后台写入不会改变正在执行的 Run；只有之后的新 top-level Run 才会冻结新的 durable memory。Resume 继续复用原 Run Snapshot。

V3 不维护固定 Topic、Candidate、ReviewRecord 或 approve/reject 工作流。显式管理入口只保留 `memory status`、`memory show` 和 `memory consolidate`。

## 6. Tool Runtime

工具统一进入 `ToolRuntime` 执行。

主要职责包括：

```text
Tool Call
  ↓
参数校验
  ↓
路径 / 命令 / 风险策略
  ↓
Permission / Approval
  ↓
实际执行
  ↓
Trace / Journal
  ↓
Tool Result
```

当前核心工具覆盖代码读取、搜索、编辑、写入、补丁、命令执行和任务辅助能力。

文件工具受 Workspace 边界保护。覆盖已有文件时会结合已读取版本进行 stale-write 校验，降低外部修改被静默覆盖的风险。

命令执行经过确定性的 Command Policy，再结合 Permission Mode 和 Approval Policy 判断是否允许执行。

## 7. Session、Checkpoint 与 Recovery

MiniCode 将长期会话和单次运行分开管理：

- **Session** 保存多轮对话和 Canonical History；
- **Run** 表示一次具体 Agent 执行；
- **Checkpoint** 保存未完成 Run 的可恢复状态。

每个 Run 将完整 Canonical History 作为追加日志保存在 `history.jsonl`。Checkpoint 只记录已接受的 History 长度与摘要，以及工具预算、任务状态和工作区摘要。正常保存只追加新消息；如果崩溃留下尚未被 Checkpoint 接受的尾部记录，恢复后的下一次保存会先回到已确认前缀再继续追加。恢复时先校验 History 前缀，再重新检查 Workspace 状态，避免继续执行超出恢复点的消息或已经被外部修改的代码状态。

模型层的瞬时错误，例如限流、连接失败、超时和服务过载，会经过有界恢复策略处理。上下文超限会交回 Context 层压缩，而已经成功执行的工具不会因为模型重试被自动重放。

## 8. Trace 与 Execution Journal

MiniCode 对运行过程进行显式记录，包括：

- 模型调用与耗时；
- Tool Call 与 Tool Result；
- 审批结果；
- 文件修改；
- 命令执行；
- 错误和恢复过程；
- Checkpoint 与运行结束状态。

Trace 主要用于调试和复盘，Execution Journal 用于记录具有副作用的关键执行事实。

## 9. Permission 与 Plan Mode

Permission Mode 控制当前 Run 可执行的副作用范围，Approval Policy 决定需要人工确认的操作。

`run_command` 使用 Sandbox-first 的语言无关策略。命令先经过少量 Hard Safety 不变量，再匹配用户声明的 argv token prefix `command_rules`；没有规则命中时，Docker 沙箱中的普通命令自动执行，本地宿主命令进入审批。`command_rules` 使用 `deny > ask > allow` 的固定优先级，不包含 Python、npm、Maven 等工具级解析逻辑。`read-only` Permission Mode 下 Docker 将 `/workspace` 只读挂载，写能力模式才使用读写挂载。

确定性的 WorkspaceGuard、敏感路径和 Hard Safety 始终优先执行，权限模式、显式 `allow` 和人工审批都不会绕过这些边界。Session 级命令授权仅复用需要审批的非沙箱命令范围，不改变规则或沙箱能力。

Plan Mode 继续复用同一套 Agent Loop，只调整模型可见工具集合和运行规则。规划阶段只开放低风险只读能力，使模型可以检索仓库并形成计划，同时保持原有 Session、Context 和 Trace 链路。

## 10. Subagent 与 Worktree Worker

MiniCode 支持受限的任务委派能力。

只读 Subagent 使用独立上下文执行聚焦的检索和分析任务，并只把最终摘要返回父 Agent。它具有固定预算，不能递归派生，也不直接修改工作区。

需要隔离修改时，可以使用独立 Git Worktree Worker。Worker 在独立工作树中运行，父 Agent 保留最终集成控制权，避免多个执行单元直接并发修改同一个 Workspace。

## 11. TUI 与 Runtime 边界

交互层使用 TypeScript TUI，Python Runtime 负责 Agent 语义和执行状态。

两者通过 JSONL stdio 通信：

```text
TypeScript TUI
   ⇅ JSONL
Python Backend
   ↓
RunExecutor
   ↓
AgentLoop
```

TUI 负责输入、Markdown、工具活动、审批面板和状态展示；Session、权限、工具策略、Checkpoint 和 Agent 决策仍由 Python Runtime 负责。

这种边界使终端交互层可以独立演进，同时保持核心 Harness 行为一致。

## 12. Model Provider

Agent Loop 只依赖统一模型接口。

不同 Provider Adapter 负责把统一请求转换成厂商协议，再将模型响应归一化为内部 `ModelResponse`。当前支持 Qwen、DeepSeek、Kimi、OpenAI、Anthropic 和 Ollama。

Provider 差异被限制在模型适配层，Context、ToolRuntime、Checkpoint 和 Agent Loop 不需要绑定具体模型厂商。

## 13. Design Scope

MiniCode 的设计重点是本地仓库级任务的可靠执行，因此持续控制运行时复杂度：

- 保持一条主 Agent Loop；
- 保持一条 Canonical Message History；
- Tool Runtime 作为统一副作用入口；
- 状态恢复使用显式 Checkpoint；
- 扩展能力尽量复用现有主链。

项目不会为了功能数量引入无界反思循环、复杂工作流编排或默认并行写入。