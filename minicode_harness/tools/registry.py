"""Tool registry for provider function calling."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import re
from threading import Lock
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, StrictStr, field_validator, model_validator

from minicode_harness.mcp import MCPManager
from minicode_harness.runtime.cancellation import CancellationToken
from minicode_harness.runtime.runtime_tasks import (
    BackgroundCommandManager,
    RuntimeTaskRegistry,
)
from minicode_harness.policy import (
    CommandPolicyResult,
    RiskLevel,
    check_command_allowed,
    render_argv,
    render_command_policy_for_prompt,
    risk_level_for_tool,
)
from minicode_harness.skills import Skill, SkillLoader
from minicode_harness.workspace import WorkspaceAccessError, WorkspaceGuard

from .command_executor import CommandExecutor, LocalCommandExecutor
from .read_tools import find_files, inspect_git_diff, read_artifact, read_file, search_text
from .write_tools import (
    apply_patch,
    edit_file,
    extract_patch_paths,
    preview_edit_file,
    preview_write_file,
    StaleWriteError,
    write_file,
)


TaskStatusValue = Literal["pending", "in_progress", "completed", "cancelled"]
MAX_TASK_TOOL_ITEMS = 12


MemoryTopicName = Literal[
    "instructions",
    "build-and-test",
    "debugging",
    "decisions",
    "environment",
]


class ReadArgs(BaseModel):
    """Read one explicitly typed resource."""

    source: Literal["workspace", "artifact", "memory", "skill", "diff"]
    target: str | None = Field(None, min_length=1, description="Exact resource name or path.")
    start_line: int | None = Field(None, ge=1, description="Optional 1-based start line.")
    end_line: int | None = Field(None, ge=1, description="Optional 1-based end line.")

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_source_contract(self) -> "ReadArgs":
        if self.source == "diff":
            if self.target is not None or self.start_line is not None or self.end_line is not None:
                raise ValueError("diff reads do not accept target or line ranges.")
            return self
        if self.target is None:
            raise ValueError(f"{self.source} reads require target.")
        if self.source in {"memory", "skill"} and (
            self.start_line is not None or self.end_line is not None
        ):
            raise ValueError(f"{self.source} reads do not accept line ranges.")
        if self.source == "memory" and self.target not in {
            "instructions",
            "build-and-test",
            "debugging",
            "decisions",
            "environment",
        }:
            raise ValueError("memory reads require one registered Topic name.")
        if (
            self.start_line is not None
            and self.end_line is not None
            and self.end_line < self.start_line
        ):
            raise ValueError("end_line must be greater than or equal to start_line.")
        return self


class SearchArgs(BaseModel):
    """Search workspace or Artifact files and text through one bounded protocol."""

    source: Literal["workspace", "artifact"] = Field(
        "workspace",
        description="Search root type.",
    )
    kind: Literal["files", "text"]
    query: str = Field(..., min_length=1, description="Glob pattern or text/regex query.")
    path: str = Field(
        ".",
        description="Source-relative file or directory to search. '.' means the selected source root.",
    )
    limit: int = Field(50, ge=1, le=2000, description="Maximum results to return.")
    max_depth: int = Field(12, ge=0, le=64, description="Maximum depth for file search.")
    file_glob: str | None = Field(None, description="Optional text-search file glob.")
    use_regex: bool = Field(False, description="Treat text query as regex.")
    case_sensitive: bool = Field(True, description="Case-sensitive text matching.")

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_kind_contract(self) -> "SearchArgs":
        if self.kind == "files" and (
            self.file_glob is not None or self.use_regex or not self.case_sensitive
        ):
            raise ValueError("file search does not accept text-search options.")
        if self.kind == "text" and self.limit > 500:
            raise ValueError("text search limit must not exceed 500.")
        return self


class ApplyPatchArgs(BaseModel):
    """Arguments for apply_patch."""

    patch: str = Field(..., min_length=1, description="Unified diff patch to apply.")


class EditArgs(BaseModel):
    """Arguments for edit."""

    path: str = Field(..., description="File path inside the workspace.")
    old_text: str = Field(..., min_length=1, description="Exact uniquely matching text block.")
    new_text: str = Field(..., description="Replacement text block.")


class WriteArgs(BaseModel):
    """Arguments for write."""

    path: str = Field(..., description="File path inside the workspace.")
    content: str = Field(..., description="Complete UTF-8 file content.")
    overwrite: bool = Field(
        False,
        description="Set true only when intentionally replacing an existing complete file.",
    )


class DelegateTaskArgs(BaseModel):
    """Arguments for the bounded read-only subagent."""

    task: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="One focused question with one scope/result; no independent objectives.",
    )


class RequestUserInputOptionArgs(BaseModel):
    """One bounded choice for structured Plan-mode user input."""

    label: StrictStr = Field(min_length=1, max_length=80)
    description: StrictStr | None = Field(default=None, max_length=240)
    model_config = {"extra": "forbid"}


class RequestUserInputArgs(BaseModel):
    """Ask one product/design decision that repository exploration cannot resolve."""

    question: StrictStr = Field(min_length=1, max_length=500)
    options: list[RequestUserInputOptionArgs] = Field(min_length=2, max_length=4)
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_unique_options(self) -> "RequestUserInputArgs":
        labels = [option.label.strip().casefold() for option in self.options]
        if any(not label for label in labels):
            raise ValueError("User input option labels must not be empty.")
        if len(labels) != len(set(labels)):
            raise ValueError("User input option labels must be unique.")
        return self


class TaskArgs(BaseModel):
    """Create, update, or list bounded run-local tasks."""

    action: Literal["create", "update", "list"]
    tasks: list[StrictStr] | None = Field(None, max_length=MAX_TASK_TOOL_ITEMS)
    updates: dict[str, TaskStatusValue] | None = Field(None, max_length=MAX_TASK_TOOL_ITEMS)

    model_config = {"extra": "forbid"}

    @field_validator("tasks")
    @classmethod
    def normalize_tasks(cls, tasks: list[str] | None) -> list[str] | None:
        if tasks is None:
            return None
        normalized = [" ".join(task.split()) for task in tasks]
        for index, task in enumerate(normalized):
            if not task:
                raise ValueError(f"tasks[{index}] must not be empty.")
            if len(task) > 120:
                raise ValueError(f"tasks[{index}] must not exceed 120 characters.")
        return normalized

    @field_validator("updates")
    @classmethod
    def validate_task_ids(
        cls,
        updates: dict[str, TaskStatusValue] | None,
    ) -> dict[str, TaskStatusValue] | None:
        if updates is None:
            return None
        for task_id in updates:
            if re.fullmatch(r"\d+", task_id) is None:
                raise ValueError(f"Invalid task ID: {task_id}")
        return updates

    @model_validator(mode="after")
    def validate_action_contract(self) -> "TaskArgs":
        if self.action == "create":
            if not self.tasks or self.updates is not None:
                raise ValueError("create requires non-empty tasks and does not accept updates.")
        elif self.action == "update":
            if not self.updates or self.tasks is not None:
                raise ValueError("update requires non-empty updates and does not accept tasks.")
        elif self.tasks is not None or self.updates is not None:
            raise ValueError("list does not accept tasks or updates.")
        return self


class RuntimeTaskStatusArgs(BaseModel):
    """Query one runtime task or list all tasks in this Run."""

    task_id: str | None = Field(None, pattern=r"^(?:cmd|worker)_\d{4}$")
    model_config = {"extra": "forbid"}


class RuntimeTaskStopArgs(BaseModel):
    """Request cooperative cancellation for one runtime task."""

    task_id: str = Field(..., pattern=r"^(?:cmd|worker)_\d{4}$")
    model_config = {"extra": "forbid"}


class RunCommandArgs(BaseModel):
    """Validated argv protocol for foreground or explicit background execution."""

    argv: list[StrictStr] = Field(
        ...,
        min_length=1,
        max_length=64,
        description=(
            "Process argv. The first item is the executable and remaining items are exact "
            "arguments. Do not include shell wrappers, pipes, redirects, or command chaining."
        ),
    )
    timeout_seconds: int = Field(
        120,
        ge=1,
        le=600,
        description="Command timeout in seconds.",
    )
    background: bool = False

    model_config = {"extra": "forbid"}

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, argv: list[str]) -> list[str]:
        total_chars = 0
        for index, argument in enumerate(argv):
            if not argument.strip():
                raise ValueError(f"argv[{index}] must not be empty.")
            if "\x00" in argument or "\n" in argument or "\r" in argument:
                raise ValueError(f"argv[{index}] must be NUL-free and single-line.")
            if len(argument) > 4096:
                raise ValueError(f"argv[{index}] must not exceed 4096 characters.")
            total_chars += len(argument)
        if total_chars > 16_384:
            raise ValueError("argv must not exceed 16384 total characters.")
        return argv


def compact_tool_schema_for_provider(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a compact provider-facing copy without changing validation Schema."""

    def compact(value: Any, *, root: bool = False) -> Any:
        if isinstance(value, list):
            return [compact(item) for item in value]
        if not isinstance(value, dict):
            return value

        result: dict[str, Any] = {}
        for key, item in value.items():
            if key == "title":
                continue
            if key == "default" and item is None:
                continue
            if root and key == "description":
                continue
            result[key] = compact(item)
        return result

    return compact(schema, root=True)


@dataclass(frozen=True, slots=True)
class ToolAdmission:
    """One validated tool call plus its deterministic admission decision."""

    name: str
    arguments: dict[str, Any]
    parsed_arguments: BaseModel | dict[str, Any]
    risk_level: RiskLevel
    requires_approval: bool
    command_policy: CommandPolicyResult | None = None

    @property
    def allowed(self) -> bool:
        return self.command_policy.allowed if self.command_policy is not None else True


class ToolDefinition(BaseModel):
    """A registered tool with schema, risk, and execution handler."""

    name: str
    description: str
    args_model: type[BaseModel] | None = None
    parameters: dict[str, Any] | None = None
    handler: Callable[[Any], Any]
    risk_level: RiskLevel = RiskLevel.LOW
    read_only: bool = True
    destructive: bool = False
    result_reconstructible: bool = False
    counts_against_tool_budget: bool = True

    model_config = {"arbitrary_types_allowed": True}

    @property
    def side_effecting(self) -> bool:
        return self.destructive or not self.read_only

    def schema_for_model(self) -> dict[str, Any]:
        """Return an OpenAI-compatible function tool schema."""

        parameters = self.parameters
        if parameters is None:
            if self.args_model is None:
                parameters = {"type": "object", "properties": {}}
            else:
                parameters = self.args_model.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": compact_tool_schema_for_provider(parameters),
            },
        }

    def validate(
        self,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], BaseModel | dict[str, Any]]:
        """Validate once and return normalized plus handler-ready arguments."""

        if self.args_model is None:
            normalized = dict(arguments)
            return normalized, normalized
        parsed_args = self.args_model.model_validate(arguments)
        return parsed_args.model_dump(mode="python"), parsed_args

    def execute_validated(self, parsed_arguments: BaseModel | dict[str, Any]) -> Any:
        """Run a handler with arguments already validated by ``validate``."""

        return self.handler(parsed_arguments)


class ToolRegistry:
    """Registry of workspace tools."""

    def __init__(
        self,
        workspace: str,
        *,
        enable_write: bool = False,
        enable_command: bool = True,
        artifact_dir: str | None = None,
        skill_loader: SkillLoader | None = None,
        skill_names: Iterable[str] | None = None,
        mcp_manager: MCPManager | None = None,
        subagent_handler: Callable[[str], Any] | None = None,
        memory_topic_reader: Callable[[MemoryTopicName], Any] | None = None,
        task_create_handler: Callable[[list[str]], Any] | None = None,
        task_update_handler: Callable[[dict[str, TaskStatusValue]], Any] | None = None,
        task_list_handler: Callable[[], Any] | None = None,
        request_user_input_handler: Callable[[str, list[dict[str, str | None]]], Any] | None = None,
        cancellation_token: CancellationToken | None = None,
        command_executor: CommandExecutor | None = None,
        runtime_task_registry: RuntimeTaskRegistry | None = None,
        background_command_manager: BackgroundCommandManager | None = None,
        worktree_worker_handler: Callable[[str], Any] | None = None,
    ) -> None:
        self.workspace = workspace
        self.enable_write = enable_write
        self._workspace_guard = WorkspaceGuard(workspace)
        self._mutation_locks_guard = Lock()
        self._mutation_locks: dict[str, Any] = {}
        self.artifact_dir = artifact_dir
        self.skill_loader = skill_loader
        self.mcp_manager = mcp_manager
        self.subagent_handler = subagent_handler
        self.memory_topic_reader = memory_topic_reader
        self.task_create_handler = task_create_handler
        self.task_update_handler = task_update_handler
        self.task_list_handler = task_list_handler
        self.request_user_input_handler = request_user_input_handler
        self.cancellation_token = cancellation_token
        self.command_executor = command_executor or LocalCommandExecutor()
        self.runtime_task_registry = runtime_task_registry
        self.background_command_manager = background_command_manager
        self.worktree_worker_handler = worktree_worker_handler
        self.skill_names = tuple(
            skill_loader.resolve_names(skill_names) if skill_loader is not None else []
        )
        self._tools = {
            "read": ToolDefinition(
                name="read",
                description=(
                    "Read one explicitly typed resource. Workspace and Artifact targets must identify a file; "
                    "optional line ranges select an inclusive local range. Memory and Skill targets use exact "
                    "registered names, while source=diff returns the current Git diff without a target. Large "
                    "results may be externalized to an Artifact. For Memory, read at most two exact indexed Topics "
                    "per user turn."
                ),
                args_model=ReadArgs,
                handler=self._execute_read,
                risk_level=RiskLevel.LOW,
                result_reconstructible=True,
            ),
            "search": ToolDefinition(
                name="search",
                description=(
                    "Search bounded Workspace/Artifact resources. kind=files locates candidate paths by glob; narrow "
                    "path/glob before widening depth/limit, not whole-repository inventory. kind=text searches matching "
                    "lines with optional file_glob/regex/case controls. path may be file/dir; '.' is source root. Results "
                    "may truncate by bounds."
                ),
                args_model=SearchArgs,
                handler=self._execute_search,
                risk_level=RiskLevel.LOW,
                result_reconstructible=True,
            ),
        }
        task_handlers = (
            self.task_create_handler,
            self.task_update_handler,
            self.task_list_handler,
        )
        if any(handler is not None for handler in task_handlers) and not all(
            handler is not None for handler in task_handlers
        ):
            raise ValueError("Task tools require create, update, and list handlers together.")
        if self.request_user_input_handler is not None:
            self._tools["request_user_input"] = ToolDefinition(
                name="request_user_input",
                description=(
                    "Ask one bounded user decision only when repository exploration cannot determine a material "
                    "product or design preference. Offer 2-4 concise options; do not use for facts available in the repository."
                ),
                args_model=RequestUserInputArgs,
                handler=self._execute_request_user_input,
                risk_level=RiskLevel.LOW,
                read_only=True,
                result_reconstructible=False,
                counts_against_tool_budget=False,
            )

        self._tools["task"] = ToolDefinition(
            name="task",
            description=(
                "Create, update, or list bounded run-local tasks. Use only for multi-step work; "
                "at most one task may be in_progress."
            ),
            args_model=TaskArgs,
            handler=self._execute_task,
            risk_level=RiskLevel.LOW,
            read_only=True,
            result_reconstructible=False,
            counts_against_tool_budget=False,
        )
        self._tools.update(
            {
                "delegate_task": ToolDefinition(
                    name="delegate_task",
                    description=(
                        "Delegate one focused read-only question to a bounded isolated subagent. Do not bundle modules/"
                        "workflows or whole-repository coverage. Parent gets one concise completed/partial summary; reuse "
                        "confirmed findings."
                    ),
                    args_model=DelegateTaskArgs,
                    handler=lambda args: (
                        self.subagent_handler(args.task)
                        if self.subagent_handler is not None
                        else _unavailable_tool("subagents_disabled")
                    ),
                    risk_level=RiskLevel.LOW,
                ),
                "delegate_worktree": ToolDefinition(
                    name="delegate_worktree",
                    description=(
                        "Delegate one independent code change to a bounded background worker in an "
                        "isolated Git worktree. The worker cannot recurse, merge, or commit automatically."
                    ),
                    args_model=DelegateTaskArgs,
                    handler=self._execute_delegate_worktree,
                    risk_level=RiskLevel.MEDIUM,
                    read_only=False,
                ),
                "runtime_task_status": ToolDefinition(
                    name="runtime_task_status",
                    description="Query one runtime task or list all runtime tasks in this Run.",
                    args_model=RuntimeTaskStatusArgs,
                    handler=self._execute_runtime_task_status,
                    counts_against_tool_budget=False,
                    result_reconstructible=True,
                ),
                "runtime_task_stop": ToolDefinition(
                    name="runtime_task_stop",
                    description="Request cooperative cancellation of one running runtime task.",
                    args_model=RuntimeTaskStopArgs,
                    handler=self._execute_runtime_task_stop,
                    risk_level=RiskLevel.MEDIUM,
                    read_only=False,
                    counts_against_tool_budget=False,
                ),
            }
        )
        if enable_write:
            self._tools.update(
                {
                    "edit": ToolDefinition(
                        name="edit",
                        description=(
                            "Replace one exact uniquely matching block in an existing UTF-8 "
                            "workspace file. Use this as the default for small and medium focused "
                            "edits; old_text must match exactly once."
                        ),
                        args_model=EditArgs,
                        handler=lambda args: self._execute_path_mutation(
                            args.path,
                            lambda: edit_file(
                                self.workspace,
                                args.path,
                                args.old_text,
                                args.new_text,
                            ),
                        ),
                        risk_level=RiskLevel.MEDIUM,
                        read_only=False,
                        result_reconstructible=True,
                    ),
                    "apply_patch": ToolDefinition(
                        name="apply_patch",
                        description=(
                            "Apply a unified diff for multi-file, non-contiguous, deletion, rename, "
                            "or structurally complex changes. Prefer edit for one focused "
                            "replacement."
                        ),
                        args_model=ApplyPatchArgs,
                        handler=lambda args: apply_patch(self.workspace, args.patch),
                        risk_level=RiskLevel.MEDIUM,
                        read_only=False,
                        result_reconstructible=True,
                    ),
                    "write": ToolDefinition(
                        name="write",
                        description=(
                            "Create a new UTF-8 workspace file. Replacing an existing file requires "
                            "overwrite=true and should be used only when a complete rewrite is "
                            "intentional."
                        ),
                        args_model=WriteArgs,
                        handler=lambda args: self._execute_path_mutation(
                            args.path,
                            lambda: write_file(
                                self.workspace,
                                args.path,
                                args.content,
                                overwrite=args.overwrite,
                            ),
                        ),
                        risk_level=RiskLevel.MEDIUM,
                        read_only=False,
                        result_reconstructible=True,
                    ),
                }
            )
        if enable_write and enable_command:
            command_description = render_command_policy_for_prompt()
            if self.command_executor.sandboxed:
                command_description += (
                    " Commands run inside an isolated Linux Docker container with "
                    "the repository mounted at /workspace; use container-compatible executables."
                )
            self._tools["run_command"] = ToolDefinition(
                name="run_command",
                description=command_description,
                args_model=RunCommandArgs,
                handler=lambda args: (
                    self.background_command_manager.start(
                        args.argv,
                        args.timeout_seconds,
                    )
                    if args.background and self.background_command_manager is not None
                    else self.command_executor.execute(
                        self.workspace,
                        args.argv,
                        args.timeout_seconds,
                        self.cancellation_token,
                    )
                ),
                risk_level=RiskLevel.HIGH,
                read_only=False,
            )
        self._register_mcp_tools()

    def _execute_path_mutation(
        self,
        path: str,
        operation: Callable[[], Any],
    ) -> Any:
        """Serialize mutations to one resolved file while other paths remain parallel."""

        resolved_path = self._workspace_guard.resolve(path)
        lock_key = str(resolved_path).casefold()
        with self._mutation_locks_guard:
            path_lock = self._mutation_locks.setdefault(lock_key, Lock())
        with path_lock:
            return operation()

    def _execute_read(self, args: ReadArgs) -> Any:
        target = args.target or ""
        if args.source == "workspace":
            return read_file(
                self.workspace,
                target,
                start_line=args.start_line,
                end_line=args.end_line,
            )
        if args.source == "artifact":
            if self.artifact_dir is None:
                return _unavailable_tool("artifacts_disabled")
            return read_artifact(
                self.artifact_dir,
                target,
                start_line=args.start_line,
                end_line=args.end_line,
            )
        if args.source == "memory":
            if self.memory_topic_reader is None:
                return _unavailable_tool("memory_disabled")
            return self.memory_topic_reader(target)
        if args.source == "skill":
            if self.skill_loader is None or not self.skill_names:
                return _unavailable_tool("skills_disabled")
            return self._load_skill(target)
        return inspect_git_diff(self.workspace)

    def _execute_search(self, args: SearchArgs) -> Any:
        search_root = self.workspace
        if args.source == "artifact":
            if self.artifact_dir is None:
                return _unavailable_tool("artifacts_disabled")
            search_root = self.artifact_dir
        if args.kind == "files":
            return find_files(
                search_root,
                args.path,
                args.query,
                max_results=args.limit,
                max_depth=args.max_depth,
                cancellation_token=self.cancellation_token,
            )
        return search_text(
            search_root,
            args.query,
            args.path,
            max_matches=args.limit,
            max_depth=args.max_depth,
            use_regex=args.use_regex,
            case_sensitive=args.case_sensitive,
            file_glob=args.file_glob,
            cancellation_token=self.cancellation_token,
        )

    def _execute_request_user_input(self, args: RequestUserInputArgs) -> Any:
        if self.request_user_input_handler is None:
            return _unavailable_tool("user_input_unavailable")
        return self.request_user_input_handler(
            " ".join(args.question.split()),
            [
                {
                    "label": " ".join(option.label.split()),
                    "description": (
                        " ".join(option.description.split())
                        if option.description is not None
                        else None
                    ),
                }
                for option in args.options
            ],
        )

    def _execute_task(self, args: TaskArgs) -> Any:
        if args.action == "create":
            if self.task_create_handler is None:
                return _unavailable_tool("task_store_disabled")
            return self.task_create_handler(args.tasks or [])
        if args.action == "update":
            if self.task_update_handler is None:
                return _unavailable_tool("task_store_disabled")
            return self.task_update_handler(args.updates or {})
        if self.task_list_handler is None:
            return _unavailable_tool("task_store_disabled")
        return self.task_list_handler()

    def _execute_delegate_worktree(self, args: DelegateTaskArgs) -> Any:
        if not self.enable_write:
            return _unavailable_tool("write_disabled")
        if self.worktree_worker_handler is None:
            return _unavailable_tool("worktree_workers_disabled")
        return self.worktree_worker_handler(args.task)

    def _execute_runtime_task_status(self, args: RuntimeTaskStatusArgs) -> Any:
        if self.runtime_task_registry is None:
            if args.task_id is None:
                return {"tasks": []}
            return {"status": "not_running", "task_id": args.task_id}
        return self.runtime_task_registry.status(args.task_id)

    def _execute_runtime_task_stop(self, args: RuntimeTaskStopArgs) -> Any:
        if self.runtime_task_registry is None:
            return {"status": "not_running", "task_id": args.task_id}
        return self.runtime_task_registry.stop(args.task_id)

    def _register_mcp_tools(self) -> None:
        if self.mcp_manager is None:
            return
        for spec in self.mcp_manager.list_tools():
            risk = (
                RiskLevel.HIGH
                if spec.annotations.destructive
                else RiskLevel.LOW
                if spec.annotations.read_only
                else RiskLevel.MEDIUM
            )
            self._tools[spec.registry_name] = ToolDefinition(
                name=spec.registry_name,
                description=(
                    f"MCP tool from server {spec.server_name}: {spec.description}".strip()
                ),
                parameters=spec.input_schema,
                handler=lambda arguments, name=spec.registry_name: self.mcp_manager.call_tool(
                    name, arguments
                ),
                risk_level=risk,
                read_only=spec.annotations.read_only,
                destructive=spec.annotations.destructive,
            )

    def _load_skill(self, name: str) -> Skill:
        if self.skill_loader is None or name not in self.skill_names:
            available = ", ".join(self.skill_names) if self.skill_names else "none"
            raise FileNotFoundError(
                f"Skill is not available in this run: {name}. Available skills: {available}."
            )
        return self.skill_loader.load(name)

    def schemas(self, tool_names: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """Return tool schemas for model function calling."""

        ordered_tools = sorted(self._tools.items())
        if tool_names is None:
            return [tool.schema_for_model() for _, tool in ordered_tools]
        allowed_names = set(tool_names)
        return [
            tool.schema_for_model()
            for name, tool in ordered_tools
            if name in allowed_names
        ]

    def readonly_tool_names(self) -> tuple[str, ...]:
        """Return tools safe to expose while workspace side effects are disabled."""

        return tuple(
            name
            for name, tool in sorted(self._tools.items())
            if tool.read_only
            and not tool.destructive
            and tool.risk_level == RiskLevel.LOW
        )

    def counts_against_tool_budget(self, name: str) -> bool:
        """Return whether one tool consumes the workspace-action budget."""

        tool = self._tools.get(name)
        return True if tool is None else tool.counts_against_tool_budget

    def history_effects(self) -> dict[str, dict[str, bool]]:
        """Return internal execution semantics without changing model schemas."""

        effects = {
            name: {
                "read_only": tool.read_only,
                "destructive": tool.destructive,
                "result_reconstructible": tool.result_reconstructible,
            }
            for name, tool in self._tools.items()
        }
        return effects

    def admit(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolAdmission:
        """Validate once and compute the reusable admission decision."""

        tool = self._tools.get(name)
        if tool is None:
            raise UnknownToolError(f"Unknown tool: {name}")
        normalized, parsed_arguments = tool.validate(arguments)
        command_policy = None
        risk_level = tool.risk_level
        requires_approval = risk_level in {RiskLevel.MEDIUM, RiskLevel.HIGH}
        if name == "delegate_worktree" and (
            not self.enable_write or self.worktree_worker_handler is None
        ):
            requires_approval = False
        if name == "runtime_task_stop":
            if self.runtime_task_registry is None:
                requires_approval = False
            else:
                status = self.runtime_task_registry.status(str(normalized["task_id"]))
                tasks = status.get("tasks") if isinstance(status, dict) else None
                requires_approval = bool(
                    isinstance(tasks, list)
                    and tasks
                    and tasks[0].get("status") == "running"
                )
        if name == "run_command":
            argv = list(normalized["argv"])
            classify = getattr(self.command_executor, "classify", None)
            command_policy = (
                classify(argv)
                if callable(classify)
                else check_command_allowed(
                    argv,
                    sandboxed=bool(getattr(self.command_executor, "sandboxed", False)),
                    rules=tuple(getattr(self.command_executor, "command_rules", ())),
                )
            )
            risk_level = command_policy.risk_level
            requires_approval = command_policy.requires_approval
        return ToolAdmission(
            name=name,
            arguments=normalized,
            parsed_arguments=parsed_arguments,
            risk_level=risk_level,
            requires_approval=requires_approval,
            command_policy=command_policy,
        )

    def risk_level(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> RiskLevel:
        if arguments is not None:
            return self.admit(name, arguments).risk_level
        tool = self._tools.get(name)
        if tool is None:
            return risk_level_for_tool(name)
        return tool.risk_level

    def requires_approval(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> bool:
        if arguments is not None:
            return self.admit(name, arguments).requires_approval
        return self.risk_level(name) in {RiskLevel.MEDIUM, RiskLevel.HIGH}

    def execute_admitted(
        self,
        admission: ToolAdmission,
        *,
        approval_granted: bool = False,
        expected_file_sha256: str | None = None,
    ) -> Any:
        """Execute one call already validated by ``admit``."""

        tool = self._tools.get(admission.name)
        if tool is None:
            raise UnknownToolError(f"Unknown tool: {admission.name}")
        if not admission.allowed:
            reason = (
                admission.command_policy.reason
                if admission.command_policy is not None
                else "Tool admission denied."
            )
            raise PermissionError(reason or "Tool admission denied.")
        if admission.name == "write":
            parsed = admission.parsed_arguments
            if not isinstance(parsed, WriteArgs):
                raise TypeError("write admission did not contain WriteArgs")
            if parsed.overwrite and expected_file_sha256 is None:
                target = self._workspace_guard.resolve(parsed.path)
                if target.exists():
                    raise StaleWriteError(
                        "Existing file overwrite requires a successful workspace read first."
                    )
            return self._execute_path_mutation(
                parsed.path,
                lambda: write_file(
                    self.workspace,
                    parsed.path,
                    parsed.content,
                    overwrite=parsed.overwrite,
                    expected_sha256=expected_file_sha256,
                ),
            )
        if admission.name == "run_command":
            if bool(admission.arguments.get("background", False)):
                if self.background_command_manager is None:
                    raise RuntimeError("Background command execution is not available.")
                return self.background_command_manager.start(
                    list(admission.arguments["argv"]),
                    int(admission.arguments["timeout_seconds"]),
                    approval_granted=approval_granted,
                )
            return self.command_executor.execute(
                self.workspace,
                list(admission.arguments["argv"]),
                int(admission.arguments["timeout_seconds"]),
                self.cancellation_token,
                approval_granted=approval_granted,
            )
        return tool.execute_validated(admission.parsed_arguments)

    def preview_admitted(self, admission: ToolAdmission) -> dict[str, Any]:
        """Return an approval preview from one reusable admission decision."""

        name = admission.name
        arguments = admission.arguments
        if name not in self._tools:
            raise UnknownToolError(f"Unknown tool: {name}")
        risk_level = admission.risk_level
        if name == "apply_patch":
            patch = str(arguments.get("patch", ""))
            files = extract_patch_paths(patch)
            return {
                "tool": name,
                "risk_level": risk_level.value,
                "files": files,
                "diff_summary": _diff_summary(patch),
                "summary": {
                    "tool": name,
                    "risk_level": risk_level.value,
                    "files": files,
                    "changed_lines": _changed_line_count(patch),
                },
            }
        if name == "edit":
            preview = preview_edit_file(
                self.workspace,
                str(arguments.get("path", "")),
                str(arguments.get("old_text", "")),
                str(arguments.get("new_text", "")),
            )
            return {
                "tool": name,
                "risk_level": risk_level.value,
                **preview.model_dump(mode="json"),
                "summary": {
                    "tool": name,
                    "risk_level": risk_level.value,
                    "path": preview.path,
                    "matches": preview.matches,
                    "replacement": f"{preview.old_chars} -> {preview.new_chars} chars",
                },
            }
        if name == "write":
            preview = preview_write_file(
                self.workspace,
                str(arguments.get("path", "")),
                str(arguments.get("content", "")),
                overwrite=bool(arguments.get("overwrite", False)),
            )
            diff_summary = _diff_summary(preview.diff)
            return {
                "tool": name,
                "risk_level": risk_level.value,
                **preview.model_dump(mode="json", exclude={"diff"}),
                "diff_summary": diff_summary,
                "summary": {
                    "tool": name,
                    "risk_level": risk_level.value,
                    "path": preview.path,
                    "created": preview.created,
                    "overwrite": preview.overwrite,
                    "old_bytes": preview.old_bytes,
                    "new_bytes": preview.new_bytes,
                    "changed_lines": preview.changed_lines,
                },
            }
        if name == "run_command":
            policy_result = admission.command_policy
            if policy_result is None:
                raise ValueError("run_command admission is missing command policy")
            argv = list(arguments["argv"])
            command = render_argv(argv)
            timeout_seconds = int(arguments["timeout_seconds"])
            return {
                "tool": name,
                "risk_level": policy_result.risk_level.value,
                "argv": argv,
                "command": command,
                "workspace": self.workspace,
                "timeout_seconds": timeout_seconds,
                "allowlist_rule": policy_result.rule,
                "policy_action": policy_result.action.value,
                "policy_category": policy_result.category.value,
                "effects": list(policy_result.effects),
                "requires_approval": policy_result.requires_approval,
                "allowed": policy_result.allowed,
                "reason": policy_result.reason,
                "summary": {
                    "tool": name,
                    "risk_level": policy_result.risk_level.value,
                    "argv": argv,
                    "command": command,
                    "policy_action": policy_result.action.value,
                    "policy_category": policy_result.category.value,
                    "requires_approval": policy_result.requires_approval,
                    "allowed": policy_result.allowed,
                    "effects": list(policy_result.effects),
                },
            }
        return {
            "tool": name,
            "risk_level": risk_level.value,
            "arguments": dict(arguments),
            "summary": {
                "tool": name,
                "risk_level": risk_level.value,
                "arguments": dict(arguments),
            },
        }


def _unavailable_tool(reason: str) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


class UnknownToolError(WorkspaceAccessError):
    """Raised when a model requests an unknown tool."""


def _changed_line_count(patch: str) -> int:
    return sum(
        1
        for line in patch.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )


def _diff_summary(patch: str) -> str:
    lines = patch.splitlines()
    if len(lines) <= 40:
        return patch
    return "\n".join(lines[:20] + ["...<diff preview truncated>..."] + lines[-20:])
