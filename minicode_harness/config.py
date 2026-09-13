"""Minimal user-level defaults for normal MiniCode runs."""

from __future__ import annotations

from pathlib import Path
import tomllib

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from minicode_harness.policy import ApprovalPolicy, PermissionMode
from minicode_harness.tools import SandboxMode


class UserConfig(BaseModel):
    """Stable user defaults loaded from ~/.minicode/config.toml."""

    model_config = ConfigDict(extra="forbid")

    provider: str | None = None
    model: str | None = None
    permission_mode: PermissionMode | None = None
    approval_policy: ApprovalPolicy | None = None
    sandbox: SandboxMode | None = None

    @field_validator("provider", "model")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


def default_user_config_path() -> Path:
    return Path.home() / ".minicode" / "config.toml"


def load_user_config(path: Path | str | None = None) -> UserConfig:
    config_path = (
        Path(path).expanduser()
        if path is not None
        else default_user_config_path()
    )
    if not config_path.is_file():
        return UserConfig()
    try:
        with config_path.open("rb") as handle:
            payload = tomllib.load(handle)
        return UserConfig.model_validate(payload)
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ValueError(
            f"Invalid MiniCode user config at {config_path}: {exc}"
        ) from exc
