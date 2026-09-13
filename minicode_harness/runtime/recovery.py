"""Bounded recovery policy for model calls.

The policy retries only transient provider failures. Context overflow and output
truncation are returned to ``AgentLoop`` because recovering those conditions
requires canonical-message changes owned by the loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Callable, TypeVar

from minicode_harness.models.errors import ModelCallFailure, classify_model_exception


T = TypeVar("T")


@dataclass(frozen=True)
class ModelRecoveryConfig:
    max_transient_retries: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0
    jitter_ratio: float = 0.25
    max_reactive_compactions: int = 1
    max_output_recoveries: int = 1


class PromptTooLongFailure(RuntimeError):
    def __init__(self, failure: ModelCallFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


class ModelCallFailed(RuntimeError):
    def __init__(self, failure: ModelCallFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


class ModelRecoveryPolicy:
    """Execute one provider call with bounded transient retries."""

    def __init__(
        self,
        config: ModelRecoveryConfig | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        random_fraction: Callable[[], float] = random.random,
    ) -> None:
        self.config = config or ModelRecoveryConfig()
        self._sleep = sleep
        self._random_fraction = random_fraction

    def invoke(
        self,
        operation: Callable[[], T],
        *,
        on_retry: Callable[[int, ModelCallFailure, float], None] | None = None,
    ) -> T:
        attempt = 0
        while True:
            try:
                return operation()
            except Exception as exc:
                failure = classify_model_exception(exc)
                if failure.kind == "prompt_too_long":
                    raise PromptTooLongFailure(failure) from exc
                if not failure.retryable or attempt >= self.config.max_transient_retries:
                    raise ModelCallFailed(failure) from exc
                attempt += 1
                delay = self._retry_delay(attempt, failure)
                if on_retry is not None:
                    on_retry(attempt, failure, delay)
                self._sleep(delay)

    def _retry_delay(self, attempt: int, failure: ModelCallFailure) -> float:
        if failure.retry_after_seconds is not None:
            return max(0.0, failure.retry_after_seconds)
        base = min(
            self.config.base_delay_seconds * (2 ** max(0, attempt - 1)),
            self.config.max_delay_seconds,
        )
        jitter = base * self.config.jitter_ratio * max(0.0, min(1.0, self._random_fraction()))
        return base + jitter
