"""Sandbox-first, language-neutral command policy."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil

from pydantic import BaseModel, Field, field_validator

from minicode_harness.workspace.guard import (
    SENSITIVE_DIRECTORY_NAMES,
    SENSITIVE_FILE_NAMES,
    SENSITIVE_SUFFIXES,
)

from .risk import RiskLevel


URL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")

SAFE_COMMAND_EXAMPLES = (
    "commands contained by the configured sandbox",
    "commands matched by an explicit allow prefix rule",
)
APPROVAL_COMMAND_EXAMPLES = (
    "unsandboxed commands without an explicit allow rule",
    "commands matched by an explicit ask prefix rule",
)
DENIED_COMMAND_HINTS = (
    "shell interpreters and privilege escalation",
    "workspace-external or sensitive paths",
    "broad destructive deletion and destructive Git operations",
    "forced remote history rewrites and environment dumping",
)


class CommandPolicyAction(StrEnum):
    """Deterministic action selected for one command."""

    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class CommandCategory(StrEnum):
    """Stable audit category for command decisions."""

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


class CommandRuleDecision(StrEnum):
    """User-configurable decision for a token-prefix command rule."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class CommandRule(BaseModel):
    """Language-neutral command rule matched against argv token prefixes."""

    decision: CommandRuleDecision
    prefix: tuple[str, ...]

    @field_validator("prefix")
    @classmethod
    def validate_prefix(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("command rule prefix must not be empty")
        normalized: list[str] = []
        for index, token in enumerate(value):
            if not isinstance(token, str) or not token.strip():
                raise ValueError(f"command rule prefix[{index}] must be a non-empty string")
            if "\n" in token or "\r" in token or "\x00" in token:
                raise ValueError(f"command rule prefix[{index}] must be single-line and NUL-free")
            normalized.append(token.strip())
        return tuple(normalized)


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
        return self.action != CommandPolicyAction.DENY

    @property
    def requires_approval(self) -> bool:
        return self.action == CommandPolicyAction.REQUIRE_APPROVAL


def render_command_policy_for_prompt() -> str:
    """Return the minimal model-facing command contract."""

    return (
        "Run argv in the workspace with shell=False. Commands inside the configured "
        "sandbox normally run directly; explicit command rules may allow, ask, or deny, "
        "and hard safety boundaries are never approval-bypassable."
    )


def render_argv(argv: Sequence[str]) -> str:
    """Render argv for display, Trace, and reports; never execute this text."""

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
    *,
    sandboxed: bool = False,
    rules: Sequence[CommandRule] = (),
) -> str | None:
    """Return a narrow Session grant identity for an approval-requiring command."""

    if sandboxed:
        return None
    policy = classify_argv(argv, sandboxed=False, rules=rules)
    if not policy.requires_approval or not policy.argv:
        return None
    executable = resolve_command_executable_identity(workspace, policy.argv[0])
    if executable is None:
        return None

    scope_kind, scope = _session_scope(policy.argv)
    payload = json.dumps(
        [executable, scope_kind, *scope],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"v3:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def classify_argv(
    argv: Sequence[str],
    *,
    sandboxed: bool = False,
    rules: Sequence[CommandRule] = (),
) -> CommandPolicyResult:
    """Classify argv using hard safety, declarative rules, and sandbox context."""

    normalized, error = _normalize_argv(argv)
    if error:
        return _deny(error, normalized)

    hard_reason = _hard_safety_violation(normalized, sandboxed=sandboxed)
    if hard_reason:
        return _deny(hard_reason, normalized)

    matched = _matching_rule(normalized, rules)
    if matched is not None:
        rendered_prefix = render_argv(matched.prefix)
        rule_name = f"command rule {matched.decision.value}: {rendered_prefix}"
        if matched.decision == CommandRuleDecision.DENY:
            return _deny(
                f"Command matched explicit deny rule: {rendered_prefix}",
                normalized,
                rule=rule_name,
            )
        if matched.decision == CommandRuleDecision.ASK:
            return _require_approval(
                rule_name,
                normalized,
                reason=f"Command matched explicit ask rule: {rendered_prefix}",
            )
        return _allow(rule_name, normalized)

    if sandboxed:
        return _allow(
            "sandbox default",
            normalized,
            effects=["execution remains inside the configured command sandbox"],
        )

    return _require_approval(
        "local execution default",
        normalized,
        reason="Unsandboxed host command execution requires approval unless an allow rule matches.",
        risk_level=RiskLevel.MEDIUM,
    )


def check_command_allowed(
    argv: Sequence[str],
    *,
    sandboxed: bool = False,
    rules: Sequence[CommandRule] = (),
) -> CommandPolicyResult:
    """Return the command decision for the current execution boundary."""

    return classify_argv(argv, sandboxed=sandboxed, rules=rules)


def _normalize_argv(argv: Sequence[str]) -> tuple[list[str], str | None]:
    normalized: list[str] = []
    for index, argument in enumerate(argv):
        if not isinstance(argument, str):
            return normalized, f"argv[{index}] must be a string."
        if not argument.strip():
            return normalized, f"argv[{index}] must not be empty."
        if "\n" in argument or "\r" in argument or "\x00" in argument:
            return normalized, f"argv[{index}] must be NUL-free and single-line."
        normalized.append(argument)
    if not normalized:
        return [], "Command argv must not be empty."
    return normalized, None


def _matching_rule(
    argv: list[str],
    rules: Sequence[CommandRule],
) -> CommandRule | None:
    matches = [rule for rule in rules if _rule_matches(argv, rule.prefix)]
    if not matches:
        return None
    priority = {
        CommandRuleDecision.DENY: 3,
        CommandRuleDecision.ASK: 2,
        CommandRuleDecision.ALLOW: 1,
    }
    return max(matches, key=lambda rule: (priority[rule.decision], len(rule.prefix)))


def _rule_matches(argv: Sequence[str], prefix: Sequence[str]) -> bool:
    if len(argv) < len(prefix):
        return False
    for index, expected in enumerate(prefix):
        actual = argv[index]
        if index == 0:
            if _executable_name(actual) != _executable_name(expected):
                return False
        elif actual != expected:
            return False
    return True


def _hard_safety_violation(argv: list[str], *, sandboxed: bool) -> str | None:
    boundary_reason = _workspace_boundary_violation(argv, sandboxed=sandboxed)
    if boundary_reason:
        return boundary_reason

    executable = _executable_name(argv[0])
    if executable in {
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
    }:
        return f"Shell interpreter {executable!r} is denied; use structured argv directly."
    if executable in {
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
    }:
        return f"Executable {executable!r} crosses the command privilege/system boundary."
    if executable in {"env", "printenv", "set"}:
        return "Environment dumping is denied by the credential-safety boundary."

    lowered = [argument.lower() for argument in argv[1:]]
    if executable == "git":
        if lowered and lowered[0] == "reset" and "--hard" in lowered[1:]:
            return "Destructive git reset --hard is denied."
        if lowered and lowered[0] == "clean" and any(
            "f" in argument.lstrip("-") for argument in lowered[1:] if argument.startswith("-")
        ):
            return "Destructive git clean with force is denied."
        if lowered and lowered[0] in {"checkout", "restore"} and any(
            argument in {".", "*", "--worktree"} for argument in lowered[1:]
        ):
            return "Broad destructive Git worktree restoration is denied."
        if lowered and lowered[0] == "push" and any(
            argument in {"--force", "-f", "--force-with-lease"} for argument in lowered[1:]
        ):
            return "Forced remote history rewrites are denied."

    if executable == "rm":
        flags = "".join(argument.lstrip("-") for argument in lowered if argument.startswith("-"))
        targets = [argument for argument in lowered if not argument.startswith("-")]
        if "r" in flags and "f" in flags and any(
            target in {".", "*", "./*"} for target in targets
        ):
            return "Broad recursive forced deletion is denied."
    if executable in {"del", "erase", "rmdir"}:
        if any(argument in {"/s", "/q"} for argument in lowered) and any(
            "*" in argument or argument == "." for argument in lowered
        ):
            return "Broad recursive deletion is denied."
    return None


def _workspace_boundary_violation(argv: list[str], *, sandboxed: bool) -> str | None:
    for raw in argv[1:]:
        sensitive_reference = _embedded_sensitive_reference(raw)
        if sensitive_reference:
            return f"Sensitive workspace path is denied: {sensitive_reference}"
        candidate = raw
        if candidate.startswith("-") and "=" not in candidate:
            continue
        if candidate.startswith("-") and "=" in candidate:
            candidate = candidate.split("=", 1)[1]
        if not candidate or URL_PATTERN.match(candidate):
            continue
        normalized = candidate.replace("\\", "/")
        if not sandboxed:
            if normalized.startswith("~") or WINDOWS_ABSOLUTE_PATH.match(candidate):
                return f"Workspace-external path arguments are denied: {candidate}"
            if normalized.startswith("/") and not re.fullmatch(r"/[A-Za-z]", normalized):
                return f"Absolute path arguments are denied: {candidate}"
        parts = [part for part in PurePosixPath(normalized).parts if part not in {"", "."}]
        if not sandboxed and ".." in parts:
            return f"Parent-directory traversal is denied: {candidate}"
        lowered_parts = [part.lower() for part in parts]
        if any(part in {".mini-code", "runs"} for part in lowered_parts[:-1]):
            return f"Sensitive workspace path is denied: {candidate}"
        if lowered_parts:
            name = lowered_parts[-1]
            if name in {".git", ".env", "id_rsa"} or name.endswith((".pem", ".key", ".p12", ".jks")):
                return f"Sensitive credential path is denied: {candidate}"
    return None


def _embedded_sensitive_reference(value: str) -> str | None:
    normalized = value.lower().replace("\\", "/")
    explicit_names = sorted(
        {
            *SENSITIVE_FILE_NAMES,
            *(name for name in SENSITIVE_DIRECTORY_NAMES if name != "runs"),
        },
        key=len,
        reverse=True,
    )
    for name in explicit_names:
        if re.search(
            rf"(?<![A-Za-z0-9_.-]){re.escape(name)}(?=$|[^A-Za-z0-9_.-])",
            normalized,
        ):
            return name
    for suffix in SENSITIVE_SUFFIXES:
        match = re.search(
            rf"[A-Za-z0-9_.-]+{re.escape(suffix)}(?=$|[^A-Za-z0-9_.-])",
            normalized,
        )
        if match is not None:
            return match.group(0)
    return None


def _session_scope(argv: list[str]) -> tuple[str, list[str]]:
    if len(argv) >= 2 and not argv[1].startswith("-"):
        return "prefix", [argv[1]]
    return "exact", argv[1:]


def _executable_name(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1].lower()


def _allow(
    rule: str,
    argv: list[str],
    *,
    effects: list[str] | None = None,
) -> CommandPolicyResult:
    return CommandPolicyResult(
        action=CommandPolicyAction.ALLOW,
        risk_level=RiskLevel.LOW,
        category=CommandCategory.UNKNOWN,
        rule=rule,
        argv=argv,
        effects=effects or [],
    )


def _require_approval(
    rule: str,
    argv: list[str],
    *,
    reason: str,
    risk_level: RiskLevel = RiskLevel.MEDIUM,
) -> CommandPolicyResult:
    return CommandPolicyResult(
        action=CommandPolicyAction.REQUIRE_APPROVAL,
        risk_level=risk_level,
        category=CommandCategory.UNKNOWN,
        rule=rule,
        argv=argv,
        effects=["executes a process outside the automatic sandbox/rule allowance"],
        reason=reason,
    )


def _deny(
    reason: str,
    argv: list[str] | None = None,
    *,
    rule: str | None = None,
) -> CommandPolicyResult:
    return CommandPolicyResult(
        action=CommandPolicyAction.DENY,
        risk_level=RiskLevel.HIGH,
        category=CommandCategory.DANGEROUS,
        rule=rule,
        argv=argv or [],
        reason=reason,
    )
