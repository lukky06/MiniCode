"""Structured user decisions requested by an interactive Plan run."""

from __future__ import annotations

from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, Field, StrictStr, field_validator, model_validator


class UserInputOption(BaseModel):
    """One bounded choice shown to the user."""

    label: StrictStr = Field(min_length=1, max_length=80)
    description: StrictStr | None = Field(default=None, max_length=240)

    model_config = {"extra": "forbid"}

    @field_validator("label")
    @classmethod
    def normalize_label(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Option label must not be empty.")
        return normalized

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        return normalized or None


class UserInputRequest(BaseModel):
    """One structured user decision request."""

    id: str = Field(default_factory=lambda: f"input_{uuid4().hex[:12]}")
    question: StrictStr = Field(min_length=1, max_length=500)
    options: list[UserInputOption] = Field(min_length=2, max_length=4)

    model_config = {"extra": "forbid"}

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Question must not be empty.")
        return normalized

    @model_validator(mode="after")
    def validate_unique_labels(self) -> "UserInputRequest":
        labels = [option.label.casefold() for option in self.options]
        if len(labels) != len(set(labels)):
            raise ValueError("User input option labels must be unique.")
        return self


class UserInputResponse(BaseModel):
    """Selected option index returned by the interaction channel."""

    selected_index: int = Field(ge=0)


class UserInputClient(Protocol):
    """Decision provider for one structured user question."""

    def choose(self, request: UserInputRequest) -> UserInputResponse:
        """Return one selected option."""


class StaticUserInputClient:
    """Deterministic user-input client for tests and adapters."""

    def __init__(self, *, selected_index: int = 0) -> None:
        self.selected_index = selected_index
        self.requests: list[UserInputRequest] = []

    def choose(self, request: UserInputRequest) -> UserInputResponse:
        self.requests.append(request)
        if self.selected_index >= len(request.options):
            raise ValueError("Static user input selection is outside the option range.")
        return UserInputResponse(selected_index=self.selected_index)
