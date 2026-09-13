"""Risk-based, language-neutral command policy."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil

from pydantic import BaseModel, Field

from .risk import RiskLevel


SAFE_ARGUMENT = re.compile(r"^[A-Za-z0-9_./\\:#,@+=*?\-]+$")
MAVEN_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
MAVEN_TEST_SELECTOR = re.compile(
    rf"^{MAVEN_IDENTIFIER}(?:\.{MAVEN_IDENTIFIER})*(?:#{MAVEN_IDENTIFIER})?"
    rf"(?:,{MAVEN_IDENTIFIER}(?:\.{MAVEN_IDENTIFIER})*(?:#{MAVEN_IDENTIFIER})?)*$"
)
URL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")

SAFE_COMMAND_EXAMPLES = (
    "python -m pytest -q [test_path]",
    "python -m unittest [test_selector]",
    "python -c <read_only_diagnostic>",
    "python -m compileall -q [path]",
    "python -m ruff check [path]",
    "python -m mypy [path]",
    "node --check <file>",
    "npm|pnpm|yarn test [-- test_path]",
    "npm|pnpm|yarn run lint|typecheck|build",
    "go test ./...",
    "go vet ./...",
    "cargo test|check|clippy [filter]",
    "dotnet test|build [project]",
    "mvn -q test",
    "mvn -q -Dtest=<TestClass> test",
    "mvn -q -DskipTests compile",
    "gradle|./gradlew test|check|build",
    "make test|check|lint",
    "ctest --test-dir build",
    "cmake --build build",
    "<tool> --version",
)
APPROVAL_COMMAND_EXAMPLES = (
    "python -c <code_not_proven_read_only>",
    "python <workspace_relative_script>",
    "node -e <diagnostic_code>",
    "package installation or dependency changes",
    "network clients and external service commands",
    "Git history or index mutations",
    "unknown workspace-local commands",
)
DENIED_COMMAND_HINTS = (
    "shell interpreters and unquoted shell chaining",
    "privilege escalation and system administration",
    "workspace-external or sensitive paths",
    "destructive recursive deletion",
    "destructive Git reset/clean/restore operations",
    "forced remote history rewrites",
    "environment or credential dumping",
)

class CommandPolicyAction(StrEnum):
    """Deterministic action selected for one command."""

    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class CommandCategory(StrEnum):
    """High-level command purpose used for approval and audit."""

    VERIFICATION = "verification"
    BUILD = "build"
    DIAGNOSTIC = "diagnostic"
    INFORMATION = "information"
    REPOSITORY_SCRIPT = "repository_script"
    FILESYSTEM = "filesystem"
    DEPENDENCY = "dependency"
    NETWORK = "network"
    GIT_HISTORY = "git_history"
    GIT_MUTATION = "git_mutation"
    EXTERNAL_SYSTEM = "external_system"
    UNKNOWN = "unknown"
    DANGEROUS = "dangerous"


class CommandPolicyResult(BaseModel):
    """Risk classification for one command invocation."""

    action: CommandPolicyAction
    risk_level: RiskLevel
    category: CommandCategory
    rule: str | None = None
    argv: list[str] = Field(default_factory=list)
    effects: list[str] = Field(default_factory=list)
    reason: str | None = None

    @property
    def allowed(self) -> bool:
        """Return whether the command may reach the execution pipeline."""

        return self.action != CommandPolicyAction.DENY

    @property
    def requires_approval(self) -> bool:
        """Return whether execution needs an approval decision."""

        return self.action == CommandPolicyAction.REQUIRE_APPROVAL


def render_command_policy_for_prompt() -> str:
    """Return the minimal model-facing command contract."""

    return (
        "Run argv in the workspace with shell=False. "
        "Use focused build, test, lint, or diagnostic commands. "
        "Policy may allow, require approval, or deny execution."
    )


def render_argv(argv: Sequence[str]) -> str:
    """Render argv for display, Trace, and reports; never use this text for execution."""

    return shlex.join([str(argument) for argument in argv])


def resolve_command_executable_identity(
    workspace: Path | str,
    executable: str,
) -> str | None:
    """Resolve one command executable to a stable exact identity for Session grants."""

    raw = executable.strip()
    if not raw:
        return None
    try:
        if Path(raw).is_absolute() or "/" in raw or "\\" in raw:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = Path(workspace) / candidate
            resolved = candidate.resolve(strict=True)
        else:
            found = shutil.which(raw)
            if found is None:
                return None
            resolved = Path(found).resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_file():
        return None
    return os.path.normcase(str(resolved))


def resolve_command_session_grant(
    workspace: Path | str,
    argv: Sequence[str],
) -> str | None:
    """Return a versioned, narrowly scoped Session grant identity."""

    policy = classify_argv(argv)
    if not policy.requires_approval or not policy.argv:
        return None
    executable = resolve_command_executable_identity(workspace, policy.argv[0])
    if executable is None:
        return None

    if policy.category == CommandCategory.GIT_HISTORY and len(policy.argv) >= 2:
        scope = ["git_history", policy.argv[1]]
    else:
        scope = ["exact", *policy.argv[1:]]
    payload = json.dumps(
        [executable, policy.category.value, *scope],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"v2:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def classify_argv(argv: Sequence[str]) -> CommandPolicyResult:
    """Classify validated argv as allowed, approval-required, or denied."""

    normalized: list[str] = []
    for index, argument in enumerate(argv):
        if not isinstance(argument, str):
            return _deny(f"argv[{index}] must be a string.")
        if not argument.strip():
            return _deny(f"argv[{index}] must not be empty.")
        if "\n" in argument or "\r" in argument or "\x00" in argument:
            return _deny(f"argv[{index}] must be NUL-free and single-line.")
        normalized.append(argument)
    if not normalized:
        return _deny("Command argv must not be empty.")
    if _contains_argv_shell_operator(normalized):
        return _deny(
            "Shell chaining, pipes, redirects, substitution, and separator arguments are denied.",
            normalized,
        )

    boundary_reason = _workspace_boundary_violation(normalized)
    if boundary_reason:
        return _deny(boundary_reason, normalized)

    dangerous_reason = _dangerous_command_reason(normalized)
    if dangerous_reason:
        return _deny(dangerous_reason, normalized)

    validators: tuple[
        Callable[[list[str]], tuple[str, CommandCategory] | None], ...
    ] = (
        _python_safe_rule,
        _node_safe_rule,
        _go_safe_rule,
        _cargo_safe_rule,
        _dotnet_safe_rule,
        _maven_safe_rule,
        _gradle_safe_rule,
        _ruby_safe_rule,
        _php_safe_rule,
        _elixir_safe_rule,
        _swift_safe_rule,
        _native_safe_rule,
        _information_rule,
        _git_read_rule,
    )
    for validator in validators:
        match = validator(normalized)
        if match is not None:
            rule, category = match
            effects = (
                ["may create workspace-local build or test artifacts"]
                if category in {CommandCategory.VERIFICATION, CommandCategory.BUILD}
                else []
            )
            return _allow(rule, normalized, category=category, effects=effects)

    approval = _approval_category(normalized)
    if approval is not None:
        category, rule, effects, reason = approval
        return _require_approval(
            rule,
            normalized,
            category=category,
            effects=effects,
            reason=reason,
        )

    return _require_approval(
        "unknown workspace-local command",
        normalized,
        category=CommandCategory.UNKNOWN,
        effects=["executes an unrecognized process in the workspace"],
        reason="The command is not known to be read-only or a bounded verification command.",
        risk_level=RiskLevel.HIGH,
    )


def check_command_allowed(argv: Sequence[str]) -> CommandPolicyResult:
    """Return the risk classification for one argv command invocation."""

    return classify_argv(argv)


def _python_safe_rule(argv: list[str]) -> tuple[str, CommandCategory] | None:
    if (
        len(argv) >= 3
        and _executable_name(argv[0]) in {"python", "python3"}
        and argv[1] == "-c"
        and _safe_python_diagnostic(argv[2])
    ):
        return "python -c read-only diagnostic", CommandCategory.DIAGNOSTIC
    if argv[:3] in (["python", "-m", "pytest"], ["python3", "-m", "pytest"]):
        if _pytest_args_allowed(argv[3:]):
            return "python -m pytest [focused args]", CommandCategory.VERIFICATION
        return None
    if argv and argv[0] == "pytest" and _pytest_args_allowed(argv[1:]):
        return "pytest [focused args]", CommandCategory.VERIFICATION
    if argv[:3] in (["python", "-m", "unittest"], ["python3", "-m", "unittest"]):
        if all(_relative_selector(arg) or arg in {"-q", "-v", "discover"} for arg in argv[3:]):
            return "python -m unittest [focused args]", CommandCategory.VERIFICATION
    if argv and argv[0] in {"tox", "nox"}:
        if all(_safe_argument(arg) for arg in argv[1:]):
            return f"{argv[0]} [focused args]", CommandCategory.VERIFICATION
    if argv[:3] in (["python", "-m", "compileall"], ["python3", "-m", "compileall"]):
        tail = argv[3:]
        if tail and all(arg == "-q" or _relative_selector(arg) for arg in tail):
            return "python -m compileall [paths]", CommandCategory.BUILD
    if argv[:4] in (["python", "-m", "ruff", "check"], ["python3", "-m", "ruff", "check"]):
        if all(_relative_selector(arg) for arg in argv[4:]):
            return "python -m ruff check [paths]", CommandCategory.VERIFICATION
    if argv[:3] in (["python", "-m", "mypy"], ["python3", "-m", "mypy"]):
        if all(_relative_selector(arg) for arg in argv[3:]):
            return "python -m mypy [paths]", CommandCategory.VERIFICATION
    return None


def _pytest_args_allowed(args: list[str]) -> bool:
    allowed_flags = {
        "-q",
        "-x",
        "-s",
        "-v",
        "--lf",
        "--ff",
        "--disable-warnings",
        "--collect-only",
        "--setup-show",
    }
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in allowed_flags or re.fullmatch(r"--maxfail=\d+", arg) or re.fullmatch(
            r"--tb=(auto|long|short|line|native|no)", arg
        ):
            index += 1
            continue
        if arg == "-k":
            if index + 1 >= len(args) or not args[index + 1].strip():
                return False
            index += 2
            continue
        if arg.startswith("-") or not _relative_selector(arg):
            return False
        index += 1
    return True


def _safe_python_diagnostic(source: str) -> bool:
    """Return whether a short Python snippet is structurally read-only."""

    if not source.strip() or len(source) > 2000:
        return False
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        return False

    allowed_statements = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.Expr)
    if any(not isinstance(statement, allowed_statements) for statement in tree.body):
        return False

    banned_modules = {
        "asyncio",
        "builtins",
        "ctypes",
        "ftplib",
        "http",
        "importlib",
        "multiprocessing",
        "os",
        "pathlib",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "sys",
        "tempfile",
        "threading",
        "urllib",
    }
    banned_calls = {
        "__import__",
        "chdir",
        "chmod",
        "chown",
        "compile",
        "connect",
        "delattr",
        "delete",
        "eval",
        "exec",
        "exit",
        "input",
        "kill",
        "mkdir",
        "open",
        "popen",
        "post",
        "put",
        "quit",
        "remove",
        "rename",
        "replace",
        "rmdir",
        "run",
        "send",
        "setattr",
        "system",
        "touch",
        "unlink",
        "write",
        "write_bytes",
        "write_text",
    }

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if any(module.split(".", 1)[0] in banned_modules for module in modules):
                return False
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(not isinstance(target, ast.Name) for target in targets):
                return False
        if isinstance(node, ast.Call):
            call_name = _python_call_name(node.func)
            if any(part in banned_calls for part in call_name.split(".") if part):
                return False
    return True


def _python_call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _python_call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _node_safe_rule(argv: list[str]) -> tuple[str, CommandCategory] | None:
    if argv[:2] == ["node", "--check"] and len(argv) == 3 and _relative_selector(argv[2]):
        return "node --check <file>", CommandCategory.VERIFICATION
    if len(argv) < 2 or argv[0] not in {"npm", "pnpm", "yarn"}:
        return None
    if argv[1] == "test":
        tail = argv[2:]
        if tail and tail[0] == "--":
            tail = tail[1:]
        if all(_relative_selector(arg) for arg in tail):
            return f"{argv[0]} test [focused args]", CommandCategory.VERIFICATION
        return None
    if argv[1] == "run" and len(argv) >= 3 and argv[2] in {
        "test",
        "lint",
        "typecheck",
        "build",
        "check",
    }:
        tail = argv[3:]
        if tail and tail[0] == "--":
            tail = tail[1:]
        if all(_relative_selector(arg) for arg in tail):
            category = CommandCategory.BUILD if argv[2] == "build" else CommandCategory.VERIFICATION
            return f"{argv[0]} run {argv[2]} [focused args]", category
    if argv[0] == "yarn" and argv[1] in {"lint", "typecheck", "build", "check"}:
        if all(_relative_selector(arg) for arg in argv[2:]):
            category = CommandCategory.BUILD if argv[1] == "build" else CommandCategory.VERIFICATION
            return f"yarn {argv[1]} [focused args]", category
    return None


def _go_safe_rule(argv: list[str]) -> tuple[str, CommandCategory] | None:
    if len(argv) < 2 or argv[0] != "go" or argv[1] not in {"test", "vet"}:
        return None
    allowed_flags = {"-v", "-race", "-count=1"}
    if all(arg in allowed_flags or _relative_selector(arg) for arg in argv[2:]):
        return f"go {argv[1]} [packages]", CommandCategory.VERIFICATION
    return None


def _cargo_safe_rule(argv: list[str]) -> tuple[str, CommandCategory] | None:
    if len(argv) < 2 or argv[0] != "cargo" or argv[1] not in {"test", "check", "clippy"}:
        return None
    allowed_flags = {"--all", "--all-targets", "--lib", "--bins", "--tests", "--quiet"}
    if all(arg in allowed_flags or _simple_filter(arg) for arg in argv[2:]):
        category = CommandCategory.BUILD if argv[1] == "check" else CommandCategory.VERIFICATION
        return f"cargo {argv[1]} [filter]", category
    return None


def _dotnet_safe_rule(argv: list[str]) -> tuple[str, CommandCategory] | None:
    if len(argv) < 2 or argv[0] != "dotnet" or argv[1] not in {"test", "build"}:
        return None
    allowed_flags = {"--no-restore", "--no-build", "--nologo"}
    if all(arg in allowed_flags or _relative_selector(arg) for arg in argv[2:]):
        category = CommandCategory.BUILD if argv[1] == "build" else CommandCategory.VERIFICATION
        return f"dotnet {argv[1]} [project]", category
    return None


def _maven_safe_rule(argv: list[str]) -> tuple[str, CommandCategory] | None:
    if not argv or argv[0] != "mvn":
        return None
    if argv in (["mvn", "test"], ["mvn", "-q", "test"]):
        return "mvn test", CommandCategory.VERIFICATION
    if argv == ["mvn", "-q", "-DskipTests", "compile"]:
        return "mvn -q -DskipTests compile", CommandCategory.BUILD
    if len(argv) == 4 and argv[1] == "-q":
        if argv[2] == "test" and argv[3].startswith("-Dtest="):
            selector_argument = argv[3]
        elif argv[2].startswith("-Dtest=") and argv[3] == "test":
            selector_argument = argv[2]
        else:
            selector_argument = ""
        selector = selector_argument[len("-Dtest=") :]
        if selector and MAVEN_TEST_SELECTOR.fullmatch(selector):
            return "mvn -q -Dtest=<selector> test", CommandCategory.VERIFICATION
    return None


def _gradle_safe_rule(argv: list[str]) -> tuple[str, CommandCategory] | None:
    executables = {"gradle", "gradlew", "gradlew.bat", "./gradlew"}
    if len(argv) < 2 or argv[0] not in executables or argv[1] not in {"test", "check", "build"}:
        return None
    if not argv[2:]:
        category = CommandCategory.BUILD if argv[1] == "build" else CommandCategory.VERIFICATION
        return f"gradle {argv[1]}", category
    if argv[1] == "test" and len(argv) == 4 and argv[2] == "--tests" and _simple_filter(argv[3]):
        return "gradle test --tests <selector>", CommandCategory.VERIFICATION
    return None


def _ruby_safe_rule(argv: list[str]) -> tuple[str, CommandCategory] | None:
    if argv[:3] == ["bundle", "exec", "rspec"] and all(
        _relative_selector(arg) for arg in argv[3:]
    ):
        return "bundle exec rspec [path]", CommandCategory.VERIFICATION
    return None


def _php_safe_rule(argv: list[str]) -> tuple[str, CommandCategory] | None:
    if argv == ["composer", "test"]:
        return "composer test", CommandCategory.VERIFICATION
    return None


def _elixir_safe_rule(argv: list[str]) -> tuple[str, CommandCategory] | None:
    if argv[:2] == ["mix", "test"] and all(_relative_selector(arg) for arg in argv[2:]):
        return "mix test [path]", CommandCategory.VERIFICATION
    return None


def _swift_safe_rule(argv: list[str]) -> tuple[str, CommandCategory] | None:
    if argv == ["swift", "test"]:
        return "swift test", CommandCategory.VERIFICATION
    return None


def _native_safe_rule(argv: list[str]) -> tuple[str, CommandCategory] | None:
    if len(argv) == 2 and argv[0] == "make" and argv[1] in {"test", "check", "lint"}:
        return f"make {argv[1]}", CommandCategory.VERIFICATION
    if argv[:2] == ["ctest", "--test-dir"] and len(argv) == 3 and _relative_selector(argv[2]):
        return "ctest --test-dir <path>", CommandCategory.VERIFICATION
    if argv[:2] == ["cmake", "--build"] and len(argv) == 3 and _relative_selector(argv[2]):
        return "cmake --build <path>", CommandCategory.BUILD
    return None


def _information_rule(argv: list[str]) -> tuple[str, CommandCategory] | None:
    executable = _executable_name(argv[0])
    version_tools = {
        "python",
        "python3",
        "node",
        "npm",
        "pnpm",
        "yarn",
        "go",
        "cargo",
        "rustc",
        "java",
        "javac",
        "mvn",
        "gradle",
        "dotnet",
        "ruby",
        "php",
        "mix",
        "swift",
        "cmake",
        "git",
    }
    if executable in version_tools and len(argv) == 2 and argv[1] in {
        "--version",
        "-V",
        "-version",
        "version",
    }:
        return f"{executable} version", CommandCategory.INFORMATION
    return None


def _git_read_rule(argv: list[str]) -> tuple[str, CommandCategory] | None:
    if not argv or _executable_name(argv[0]) != "git" or len(argv) < 2:
        return None
    subcommand = argv[1]
    if subcommand == "status" and all(arg in {"--short", "--porcelain"} for arg in argv[2:]):
        return "git status [read-only]", CommandCategory.INFORMATION
    if subcommand == "diff" and all(
        arg in {"--stat", "--name-only", "--cached", "--staged"}
        for arg in argv[2:]
    ):
        return "git diff [current worktree]", CommandCategory.INFORMATION
    if subcommand == "ls-files":
        return "git ls-files [read-only]", CommandCategory.INFORMATION
    if subcommand == "rev-parse":
        return "git rev-parse [read-only]", CommandCategory.INFORMATION
    return None


def _approval_category(
    argv: list[str],
) -> tuple[CommandCategory, str, list[str], str] | None:
    executable = _executable_name(argv[0])

    if executable in {"python", "python3"}:
        if len(argv) >= 3 and argv[1] == "-c":
            return (
                CommandCategory.DIAGNOSTIC,
                "python -c diagnostic",
                ["executes model-provided Python code in the workspace"],
                "Interpreter snippets can perform side effects and require approval.",
            )
        if len(argv) >= 2 and _relative_selector(argv[1]):
            return (
                CommandCategory.REPOSITORY_SCRIPT,
                "python workspace-relative script",
                ["executes repository-controlled Python code", "may modify workspace files"],
                "Repository scripts are not statically known to be side-effect free.",
            )

    if executable == "node":
        if len(argv) >= 3 and argv[1] in {"-e", "--eval"}:
            return (
                CommandCategory.DIAGNOSTIC,
                "node diagnostic expression",
                ["executes model-provided JavaScript in the workspace"],
                "Interpreter expressions can perform side effects and require approval.",
            )
        if len(argv) >= 2 and _relative_selector(argv[1]):
            return (
                CommandCategory.REPOSITORY_SCRIPT,
                "node workspace-relative script",
                ["executes repository-controlled JavaScript", "may modify workspace files"],
                "Repository scripts are not statically known to be side-effect free.",
            )

    if executable in {"pip", "pip3", "uv", "poetry", "npm", "pnpm", "yarn", "composer", "gem"}:
        if any(arg in {"install", "add", "remove", "uninstall", "update", "upgrade", "sync"} for arg in argv[1:]):
            return (
                CommandCategory.DEPENDENCY,
                "dependency environment mutation",
                ["may modify dependency files", "may modify the active toolchain environment", "may access the network"],
                "Dependency changes have persistent environment or network effects.",
            )

    if executable in {"curl", "wget", "http", "httpie", "ssh", "scp", "sftp", "ftp", "nc", "ncat"}:
        return (
            CommandCategory.NETWORK,
            "network command",
            ["accesses an external network resource", "may transmit workspace or environment data"],
            "Network access is an external side effect and requires approval.",
        )

    if executable == "git" and len(argv) >= 2:
        if argv[1] in {
            "log",
            "show",
            "blame",
            "reflog",
            "cat-file",
            "ls-tree",
        }:
            return (
                CommandCategory.GIT_HISTORY,
                f"git {argv[1]}",
                ["reads repository history or object data"],
                "Repository-history inspection can expose future fixes or benchmark answers and requires approval.",
            )
        if argv[1] in {
            "add",
            "commit",
            "checkout",
            "restore",
            "reset",
            "clean",
            "merge",
            "rebase",
            "cherry-pick",
            "tag",
            "branch",
            "push",
            "pull",
            "fetch",
            "clone",
            "switch",
            "stash",
        }:
            effects = ["modifies repository index, worktree, history, or remote state"]
            if argv[1] in {"push", "pull", "fetch", "clone"}:
                effects.append("accesses a remote Git service")
            return (
                CommandCategory.GIT_MUTATION,
                f"git {argv[1]}",
                effects,
                "Git state mutations and remote operations require approval.",
            )

    if executable in {"rm", "del", "erase", "rmdir", "mv", "move", "cp", "copy", "mkdir", "touch"}:
        return (
            CommandCategory.FILESYSTEM,
            "filesystem mutation",
            ["modifies or removes workspace files"],
            "Filesystem mutations should normally use structured workspace tools and require approval.",
        )

    if executable in {"docker", "podman", "kubectl", "helm", "terraform", "ansible", "psql", "mysql", "redis-cli"}:
        return (
            CommandCategory.EXTERNAL_SYSTEM,
            "external system command",
            ["may modify containers, infrastructure, databases, or external services"],
            "External system commands have effects outside the workspace and require approval.",
        )

    return None


def _dangerous_command_reason(argv: list[str]) -> str | None:
    executable = _executable_name(argv[0])
    denied_executables = {
        "sh",
        "bash",
        "zsh",
        "fish",
        "cmd",
        "cmd.exe",
        "powershell",
        "powershell.exe",
        "pwsh",
        "wsl",
        "sudo",
        "su",
        "doas",
        "runas",
        "systemctl",
        "service",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "mkfs",
        "diskpart",
        "format",
        "mount",
        "umount",
        "reg",
        "reg.exe",
        "sc",
        "sc.exe",
        "env",
        "printenv",
        "set",
    }
    if executable in denied_executables:
        return f"Executable {executable!r} is denied by the system and shell safety policy."

    lowered = [arg.lower() for arg in argv[1:]]
    if executable == "git":
        if len(lowered) >= 2 and lowered[0] == "reset" and "--hard" in lowered[1:]:
            return "Destructive git reset --hard is denied."
        if lowered and lowered[0] == "clean" and any("f" in arg.lstrip("-") for arg in lowered[1:] if arg.startswith("-")):
            return "Destructive git clean with force is denied."
        if lowered and lowered[0] in {"checkout", "restore"} and any(arg in {".", "*", "--worktree"} for arg in lowered[1:]):
            return "Broad destructive Git worktree restoration is denied."
        if lowered and lowered[0] == "push" and any(arg in {"--force", "-f", "--force-with-lease"} for arg in lowered[1:]):
            return "Forced remote history rewrites are denied."

    if executable == "rm":
        flags = "".join(arg.lstrip("-") for arg in lowered if arg.startswith("-"))
        targets = [arg for arg in lowered if not arg.startswith("-")]
        if "r" in flags and "f" in flags and any(target in {".", "*", "./*"} for target in targets):
            return "Broad recursive forced deletion is denied."
    if executable in {"del", "erase", "rmdir"}:
        if any(arg in {"/s", "/q"} for arg in lowered) and any("*" in arg or arg == "." for arg in lowered):
            return "Broad recursive deletion is denied."
    if executable in {"docker", "podman"} and len(lowered) >= 2:
        if lowered[0] in {"system", "volume", "image", "container"} and "prune" in lowered[1:]:
            return "Container runtime prune operations are denied."
    if executable in {"kill", "taskkill"} and any(arg in {"-9", "/f"} for arg in lowered):
        if any(arg in {"-1", "*", "/im"} for arg in lowered):
            return "Broad forced process termination is denied."
    return None


def _workspace_boundary_violation(argv: list[str]) -> str | None:
    code_payload_indexes: set[int] = set()
    executable = _executable_name(argv[0])
    if executable in {"python", "python3"} and len(argv) >= 3 and argv[1] == "-c":
        code_payload_indexes.add(2)
    if executable == "node" and len(argv) >= 3 and argv[1] in {"-e", "--eval"}:
        code_payload_indexes.add(2)

    for index, raw in enumerate(argv[1:], start=1):
        if index in code_payload_indexes:
            continue
        candidate = raw
        if candidate.startswith("-") and "=" not in candidate:
            continue
        if candidate.startswith("-") and "=" in candidate:
            candidate = candidate.split("=", 1)[1]
        if not candidate or URL_PATTERN.match(candidate):
            continue
        normalized = candidate.replace("\\", "/")
        if normalized.startswith("~") or WINDOWS_ABSOLUTE_PATH.match(candidate):
            return f"Workspace-external path arguments are denied: {candidate}"
        if normalized.startswith("/") and not re.fullmatch(r"/[A-Za-z]", normalized):
            return f"Absolute path arguments are denied: {candidate}"
        parts = [part for part in PurePosixPath(normalized).parts if part not in {"", "."}]
        if ".." in parts:
            return f"Parent-directory traversal is denied: {candidate}"
        lowered_parts = [part.lower() for part in parts]
        if any(part in {".git", ".mini-code", "runs"} for part in lowered_parts[:-1]):
            return f"Sensitive workspace path is denied: {candidate}"
        if lowered_parts:
            name = lowered_parts[-1]
            if name in {".git", ".mini-code", "runs", ".env", "id_rsa"} or name.endswith((".pem", ".key", ".p12", ".jks")):
                return f"Sensitive credential path is denied: {candidate}"
    return None


def _contains_argv_shell_operator(argv: Sequence[str]) -> bool:
    """Reject explicit shell-control tokens even though execution uses shell=False."""

    operators = {
        "|",
        "||",
        "&&",
        ";",
        ">",
        ">>",
        "<",
        "<<",
        "2>",
        "2>>",
        "2>&1",
        "1>&2",
        "&",
    }
    return any(argument in operators for argument in argv)


def _safe_argument(value: str) -> bool:
    if not value or not SAFE_ARGUMENT.fullmatch(value):
        return False
    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or WINDOWS_ABSOLUTE_PATH.match(value):
        return False
    return ".." not in normalized.split("/")


def _relative_selector(value: str) -> bool:
    return _safe_argument(value) and not value.startswith("-")


def _simple_filter(value: str) -> bool:
    return _safe_argument(value) and not value.startswith("-")


def _executable_name(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1].lower()


def _allow(
    rule: str,
    argv: list[str],
    *,
    category: CommandCategory,
    effects: list[str] | None = None,
) -> CommandPolicyResult:
    return CommandPolicyResult(
        action=CommandPolicyAction.ALLOW,
        risk_level=RiskLevel.LOW,
        category=category,
        rule=rule,
        argv=argv,
        effects=effects or [],
    )


def _require_approval(
    rule: str,
    argv: list[str],
    *,
    category: CommandCategory,
    effects: list[str],
    reason: str,
    risk_level: RiskLevel = RiskLevel.MEDIUM,
) -> CommandPolicyResult:
    return CommandPolicyResult(
        action=CommandPolicyAction.REQUIRE_APPROVAL,
        risk_level=risk_level,
        category=category,
        rule=rule,
        argv=argv,
        effects=effects,
        reason=reason,
    )


def _deny(reason: str, argv: list[str] | None = None) -> CommandPolicyResult:
    return CommandPolicyResult(
        action=CommandPolicyAction.DENY,
        risk_level=RiskLevel.HIGH,
        category=CommandCategory.DANGEROUS,
        argv=argv or [],
        reason=reason,
    )
