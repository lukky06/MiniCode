"""Prepare one exact model request from persistent conversation messages."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any, Callable

from pydantic import BaseModel, Field

from minicode_harness.models.request import ModelRequest

from .compaction_state import SessionCompactionState
from .history_compaction import (
    COMPACTED_EXECUTION_HEADING,
    CURRENT_TOOL_FRONTIER_HEADING,
    MAX_EXECUTION_RECORD_CHARS,
    compact_current_tool_frontier,
    compact_execution_history_with_details,
)
from .message_groups import (
    MessageGroup,
    find_latest_user_group_index,
    flatten_groups,
    group_messages,
    is_final_assistant_message,
    select_recent_semantic_turns,
)
from .policy import CompactionPolicy
from .session_projection import (
    canonical_groups,
    canonical_prefix_digest,
    execution_state_for_boundary,
    project_canonical_messages,
    semantic_state_for_boundary,
    valid_execution_boundary_index,
    valid_semantic_boundary_index,
)
from .semantic_compaction import (
    SEMANTIC_HISTORY_HEADING,
    SemanticHistoryCompactionError,
    SemanticHistoryCompactor,
    estimate_semantic_compaction_request_tokens,
    render_semantic_history,
)
from .task_projection import (
    CURRENT_TASK_HEADING,
    insert_current_task_record,
    strip_task_protocol,
)
from .token import TOKEN_ESTIMATOR_VERSION, estimate_tokens
from .types import (
    ContextCompressionEvent,
    ContextObservation,
    TokenBudget,
)


SEMANTIC_COMPACTION_MIN_SAFETY_TOKENS = 512
SEMANTIC_COMPACTION_SAFETY_RATIO = 0.02
SEMANTIC_COMPACTION_MARKER = "[semantic message compacted]"
_SUMMARY_CHAR_LIMITS = (
    MAX_EXECUTION_RECORD_CHARS,
    6_000,
    3_000,
    1_500,
    750,
    320,
    0,
)
_CURRENT_TOOL_FRONTIER_CHAR_LIMITS = (3_000, 2_000, 1_200, 600, 320)
TaskProjection = tuple[str, list[list[str]]]
TaskProjectionProvider = Callable[[], TaskProjection]


@dataclass(frozen=True)
class HistoryCompactionPlan:
    """Complete message groups selected for one deterministic compaction pass."""

    historical_groups: list[MessageGroup]
    semantic_tail_groups: list[MessageGroup]
    active_user_group: list[MessageGroup]
    retained_current_groups: list[MessageGroup]
    removed_current_groups: list[MessageGroup]
    removed_tokens: int
    protected_current_group_count: int = 0
    protected_current_tokens: int = 0
    compressible_current_group_count: int = 0
    effective_token_limit: int = 0
    summary_char_limit: int = MAX_EXECUTION_RECORD_CHARS
    semantic_turns_retained: int = 0
    semantic_message_char_limit: int | None = None


class PromptBudgetExceeded(RuntimeError):
    """The minimum legal provider request cannot fit the hard prompt budget."""

    def __init__(
        self,
        *,
        token_estimate: int,
        hard_token_limit: int,
        source_tokens: dict[str, int],
    ) -> None:
        self.token_estimate = token_estimate
        self.hard_token_limit = hard_token_limit
        self.source_tokens = dict(source_tokens)
        sources = ", ".join(
            f"{name}={tokens}" for name, tokens in self.source_tokens.items()
        )
        super().__init__(
            "Minimum prompt exceeds the local hard token limit: "
            f"estimated={token_estimate}, hard_limit={hard_token_limit} "
            f"({sources})."
        )


class PreparedModelRequest(BaseModel):
    """Prepared request plus trace-friendly budget metadata."""

    request: ModelRequest
    token_estimate: int
    compression_events: list[ContextCompressionEvent] = Field(default_factory=list)
    context_window: int = 0
    prompt_budget: int = 0
    reserved_output: int = 0
    request_tokens_before_compaction: int = 0
    request_tokens_after_argument_compaction: int = 0
    budget_usage_ratio: float = 0.0
    history_groups_compacted: int = 0
    source_tokens: dict[str, int] = Field(default_factory=dict)
    compaction_update: SessionCompactionState = Field(
        default_factory=SessionCompactionState
    )
    token_estimator_version: str = TOKEN_ESTIMATOR_VERSION

    @property
    def message_count(self) -> int:
        return len(self.request.messages)


class ContextPreparer:
    """Apply protocol-safe budgeting before each model call.

    Soft and provider-overflow compaction remain deterministic. After Soft,
    proactive semantic compaction may summarize only old semantic history while
    the current User and recent protocol tail remain exact messages. Hard is an
    emergency deterministic fallback, not the first semantic trigger.
    """

    def __init__(
        self,
        budget: TokenBudget | None = None,
        *,
        semantic_compactor: SemanticHistoryCompactor | None = None,
        policy: CompactionPolicy | None = None,
    ) -> None:
        self.budget = budget or TokenBudget()
        self.semantic_compactor = semantic_compactor
        self.policy = policy or CompactionPolicy()

    def prepare(
        self,
        *,
        system_messages: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        compaction_state: SessionCompactionState | None = None,
        tool_effects: dict[str, dict[str, bool]] | None = None,
        task_projection_provider: TaskProjectionProvider | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PreparedModelRequest:
        """Project one request without mutating the canonical Session history."""

        system = _join_system_messages(system_messages)
        canonical_messages = deepcopy(messages)
        state = (
            compaction_state.model_copy(deep=True)
            if compaction_state is not None
            else SessionCompactionState()
        )
        compression_events: list[ContextCompressionEvent] = []
        task_projection: tuple[str, list[list[str]]] | None = None
        task_projection_loaded = False

        def projected(current: SessionCompactionState) -> list[dict[str, Any]]:
            nonlocal task_projection, task_projection_loaded
            base = project_canonical_messages(canonical_messages, current)
            if current.execution is None and current.semantic is None:
                return base
            if not task_projection_loaded:
                task_projection = (
                    _copy_task_projection(task_projection_provider())
                    if task_projection_provider is not None
                    else None
                )
                task_projection_loaded = True
            if task_projection is None:
                return base
            return _apply_task_projection(
                _task_free_compaction_source(base, task_projection),
                task_projection,
            )

        prepared_messages = projected(state)
        before_tokens = _request_tokens(system, prepared_messages, tools)
        after_argument_compaction = before_tokens
        token_estimate = before_tokens
        history_groups_compacted = 0

        if token_estimate > self.budget.soft_token_limit:
            state, event, removed = _advance_execution_state(
                canonical_messages,
                state,
                system=system,
                tools=tools,
                projection_builder=projected,
                target_tokens=min(
                    self.budget.soft_token_limit,
                    max(1, int(self.budget.prompt_budget * 0.58)),
                ),
                phase="soft",
                protected_current_token_limit=_current_frontier_token_limit(
                    self.policy,
                    token_limit=self.budget.prompt_budget,
                    phase="soft",
                ),
            )
            if event is not None:
                compression_events.append(event)
                history_groups_compacted += removed
                prepared_messages = projected(state)
                token_estimate = _request_tokens(system, prepared_messages, tools)

        if (
            token_estimate > self.budget.soft_token_limit
            and self.semantic_compactor is not None
        ):
            state, event, removed = self._advance_semantic_state(
                canonical_messages,
                state,
                system=system,
                tools=tools,
                projection_builder=projected,
            )
            if event is not None:
                compression_events.append(event)
                if bool(event.details.get("success")):
                    history_groups_compacted += removed
                    prepared_messages = projected(state)
                    token_estimate = _request_tokens(system, prepared_messages, tools)

        if token_estimate > self.budget.hard_token_limit:
            (
                state,
                prepared_messages,
                token_estimate,
                hard_events,
                removed,
            ) = _fit_under_hard_limit(
                canonical_messages,
                state,
                prepared_messages,
                system=system,
                tools=tools,
                budget=self.budget,
                policy=self.policy,
                tool_effects=tool_effects,
                projection_builder=projected,
            )
            compression_events.extend(hard_events)
            history_groups_compacted += removed

        if token_estimate > self.budget.hard_token_limit:
            raise PromptBudgetExceeded(
                token_estimate=token_estimate,
                hard_token_limit=self.budget.hard_token_limit,
                source_tokens=_request_source_tokens(system, prepared_messages, tools),
            )

        request_metadata = dict(metadata or {})
        request_metadata["estimated_prompt_tokens"] = token_estimate
        request_metadata["token_estimator_version"] = TOKEN_ESTIMATOR_VERSION
        request = ModelRequest(
            system=system,
            messages=prepared_messages,
            tools=deepcopy(tools),
            metadata=request_metadata,
        )
        return PreparedModelRequest(
            request=request,
            token_estimate=token_estimate,
            compression_events=compression_events,
            context_window=self.budget.context_budget,
            prompt_budget=self.budget.prompt_budget,
            reserved_output=self.budget.reserved_output,
            request_tokens_before_compaction=before_tokens,
            request_tokens_after_argument_compaction=after_argument_compaction,
            budget_usage_ratio=_usage_ratio(token_estimate, self.budget.prompt_budget),
            history_groups_compacted=history_groups_compacted,
            source_tokens=_context_source_tokens(system, prepared_messages, tools),
            compaction_update=state,
            token_estimator_version=TOKEN_ESTIMATOR_VERSION,
        )

    def _advance_semantic_state(
        self,
        canonical_messages: list[dict[str, Any]],
        state: SessionCompactionState,
        *,
        system: str,
        tools: list[dict[str, Any]],
        projection_builder: Callable[[SessionCompactionState], list[dict[str, Any]]],
        focus: str = "",
    ) -> tuple[SessionCompactionState, ContextCompressionEvent | None, int]:
        """Incrementally summarize completed semantic Turns and persist the result."""

        groups = canonical_groups(canonical_messages)
        raw_groups = [record.group for record in groups]
        first_kept_index = find_latest_user_group_index(raw_groups)
        if first_kept_index is None or first_kept_index <= 0:
            return state, None, 0

        prior_index = valid_semantic_boundary_index(groups, state.semantic)
        if prior_index == first_kept_index:
            return state, None, 0
        attempt_group_id = groups[first_kept_index].group_id
        attempt_digest = canonical_prefix_digest(
            groups,
            end_exclusive=first_kept_index + 1,
        )
        if (
            state.semantic_attempt_group_id == attempt_group_id
            and state.semantic_attempt_source_digest == attempt_digest
        ):
            return state, None, 0
        attempted_state = state.model_copy(deep=True)
        attempted_state.semantic_attempt_group_id = attempt_group_id
        attempted_state.semantic_attempt_source_digest = attempt_digest
        source_groups: list[MessageGroup] = []
        if prior_index is not None and state.semantic is not None:
            source_groups.append(
                [{"role": "assistant", "content": state.semantic.summary}]
            )
            source_groups.extend(
                deepcopy(record.group)
                for record in groups[prior_index:first_kept_index]
            )
        else:
            source_groups.extend(
                deepcopy(record.group)
                for record in groups[:first_kept_index]
            )
        if not source_groups or not _has_semantic_source(source_groups):
            return state, None, 0

        semantic_groups = [*source_groups, deepcopy(groups[first_kept_index].group)]
        compressible_indexes = set(range(len(source_groups)))
        context_indexes = {len(semantic_groups) - 1}
        current_frontier = raw_groups[first_kept_index + 1 :]
        frontier_context, _ = compact_current_tool_frontier(
            current_frontier,
            max_chars=max(_CURRENT_TOOL_FRONTIER_CHAR_LIMITS),
        )
        if frontier_context.strip():
            semantic_groups.append(
                [{"role": "assistant", "content": frontier_context}]
            )
            context_indexes.add(len(semantic_groups) - 1)
        input_tokens = estimate_semantic_compaction_request_tokens(
            semantic_groups,
            compressible_group_indexes=compressible_indexes,
            context_group_indexes=context_indexes,
            max_output_tokens=self.policy.semantic_summary_max_tokens,
            focus=focus,
        )
        safety_tokens = max(
            SEMANTIC_COMPACTION_MIN_SAFETY_TOKENS,
            int(self.budget.context_budget * SEMANTIC_COMPACTION_SAFETY_RATIO),
        )
        required_tokens = (
            input_tokens
            + self.policy.semantic_summary_max_tokens
            + safety_tokens
        )
        before_tokens = _request_tokens(
            system,
            projection_builder(state),
            tools,
        )
        if required_tokens > self.budget.context_budget:
            return (
                attempted_state,
                ContextCompressionEvent(
                    reason="semantic_history",
                    before_tokens=before_tokens,
                    after_tokens=before_tokens,
                    details={
                        "phase": "semantic",
                        "attempted": False,
                        "skipped": True,
                        "success": False,
                        "changed": False,
                        "failure_reason": "semantic_compaction_request_exceeds_context_window",
                        "request_input_tokens": input_tokens,
                        "request_required_tokens": required_tokens,
                    },
                ),
                0,
            )

        try:
            summary = self.semantic_compactor.compact(
                semantic_groups,
                compressible_group_indexes=compressible_indexes,
                context_group_indexes=context_indexes,
                max_output_tokens=self.policy.semantic_summary_max_tokens,
                focus=focus,
            )
            if not summary.items:
                raise SemanticHistoryCompactionError(
                    "semantic compaction returned no semantic items"
                )
            rendered = render_semantic_history(summary)
            execution_summary, _ = compact_execution_history_with_details(
                raw_groups[:first_kept_index],
            )
            if execution_summary.strip():
                rendered = f"{rendered}\n\n{execution_summary}"
        except SemanticHistoryCompactionError as exc:
            return (
                attempted_state,
                ContextCompressionEvent(
                    reason="semantic_history",
                    before_tokens=before_tokens,
                    after_tokens=before_tokens,
                    details={
                        "phase": "semantic",
                        "attempted": True,
                        "success": False,
                        "changed": False,
                        "failure_reason": str(exc),
                        "request_input_tokens": input_tokens,
                        "request_required_tokens": required_tokens,
                    },
                ),
                0,
            )

        semantic_state = semantic_state_for_boundary(
            groups,
            first_kept_index=first_kept_index,
            summary=rendered,
        )
        if semantic_state is None:
            return state, None, 0
        updated = attempted_state.model_copy(deep=True)
        updated.semantic = semantic_state
        after_tokens = _request_tokens(
            system,
            projection_builder(updated),
            tools,
        )
        if after_tokens >= before_tokens:
            return (
                attempted_state,
                ContextCompressionEvent(
                    reason="semantic_history",
                    before_tokens=before_tokens,
                    after_tokens=before_tokens,
                    details={
                        "phase": "semantic",
                        "attempted": True,
                        "success": False,
                        "changed": False,
                        "failure_reason": "semantic_compaction_did_not_reduce_prompt",
                    },
                ),
                0,
            )
        removed = first_kept_index - (prior_index or 0)
        return (
            updated,
            ContextCompressionEvent(
                reason="semantic_history",
                before_tokens=before_tokens,
                after_tokens=after_tokens,
                details={
                    "phase": "semantic",
                    "attempted": True,
                    "success": True,
                    "changed": True,
                    "removed_groups": max(0, removed),
                    "first_kept_group_id": semantic_state.first_kept_group_id,
                    "request_input_tokens": input_tokens,
                    "request_required_tokens": required_tokens,
                },
            ),
            max(0, removed),
        )

    def manual_compact(
        self,
        messages: list[dict[str, Any]],
        *,
        compaction_state: SessionCompactionState | None = None,
        focus: str = "",
        tool_effects: dict[str, dict[str, bool]] | None = None,
    ) -> tuple[SessionCompactionState, ContextCompressionEvent | None]:
        """Persist one user-requested semantic summary without replacing history."""

        del tool_effects
        if self.semantic_compactor is None:
            raise ValueError("manual compaction requires a semantic compactor")
        state = (
            compaction_state.model_copy(deep=True)
            if compaction_state is not None
            else SessionCompactionState()
        )
        updated, event, _ = self._advance_semantic_state(
            deepcopy(messages),
            state,
            system="",
            tools=[],
            projection_builder=lambda current: project_canonical_messages(
                messages,
                current,
            ),
            focus=focus,
        )
        return updated, event

    def reactive_compact(
        self,
        messages: list[dict[str, Any]],
        *,
        compaction_state: SessionCompactionState | None = None,
        tool_effects: dict[str, dict[str, bool]] | None = None,
        task_projection_provider: TaskProjectionProvider | None = None,
    ) -> tuple[SessionCompactionState, ContextCompressionEvent | None]:
        """Advance deterministic state after provider overflow without an LLM."""

        del tool_effects
        state = (
            compaction_state.model_copy(deep=True)
            if compaction_state is not None
            else SessionCompactionState()
        )
        task_projection: tuple[str, list[list[str]]] | None = None
        task_projection_loaded = False

        def projected(current: SessionCompactionState) -> list[dict[str, Any]]:
            nonlocal task_projection, task_projection_loaded
            base = project_canonical_messages(messages, current)
            if current.execution is None and current.semantic is None:
                return base
            if not task_projection_loaded:
                task_projection = (
                    _copy_task_projection(task_projection_provider())
                    if task_projection_provider is not None
                    else None
                )
                task_projection_loaded = True
            if task_projection is None:
                return base
            return _apply_task_projection(
                _task_free_compaction_source(base, task_projection),
                task_projection,
            )

        before = _request_tokens("", projected(state), [])
        target = min(
            self.budget.soft_token_limit,
            max(1, int(before * 0.75)),
        )
        updated, event, _ = _advance_execution_state(
            messages,
            state,
            system="",
            tools=[],
            projection_builder=projected,
            target_tokens=target,
            phase="reactive",
        )
        if event is None:
            updated, event, _ = _advance_execution_state(
                messages,
                state,
                system="",
                tools=[],
                projection_builder=projected,
                target_tokens=target,
                phase="reactive",
                protect_latest_tool_group=False,
            )
        if event is None:
            return state, None
        return (
            updated,
            ContextCompressionEvent(
                reason="reactive_prompt_too_long",
                before_tokens=event.before_tokens,
                after_tokens=event.after_tokens,
                details=event.details,
            ),
        )


def render_tool_result_message(observation: ContextObservation) -> str:
    """Render a tool result without duplicating tool-call or payload fields."""

    content = observation.content.strip()
    metadata = observation.metadata
    status = str(metadata.get("status") or "ok")
    if status == "ok":
        return content

    try:
        structured = json.loads(content)
    except json.JSONDecodeError:
        structured = None
    if isinstance(structured, dict):
        payload = dict(structured)
        payload["status"] = status
        for key in (
            "error_type",
            "retryable",
            "retry_hint",
            "side_effect",
            "command_status",
            "duration_ms",
            "runtime_task_id",
        ):
            value = metadata.get(key)
            if value not in (None, "", []):
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)

    payload: dict[str, Any] = {
        "status": status,
        "message": content,
    }
    for key in (
        "error_type",
        "retryable",
        "retry_hint",
        "side_effect",
        "reason",
        "path",
        "query",
        "files",
        "requested_range",
        "covered_by",
        "total_lines",
        "source_tool_call_id",
        "match_count",
        "command",
        "argv",
        "returncode",
        "previous_returncode",
        "workspace_generation",
        "timed_out",
        "repeat_count",
        "approval_decision",
        "risk_level",
    ):
        value = metadata.get(key)
        if value not in (None, "", []):
            payload[key] = value
    if observation.artifact_path:
        payload["artifact_path"] = observation.artifact_path
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _fit_under_hard_limit(
    canonical_messages: list[dict[str, Any]],
    state: SessionCompactionState,
    prepared_messages: list[dict[str, Any]],
    *,
    system: str,
    tools: list[dict[str, Any]],
    budget: TokenBudget,
    policy: CompactionPolicy,
    tool_effects: dict[str, dict[str, bool]] | None,
    projection_builder: Callable[[SessionCompactionState], list[dict[str, Any]]],
) -> tuple[
    SessionCompactionState,
    list[dict[str, Any]],
    int,
    list[ContextCompressionEvent],
    int,
]:
    """Fit one projected request under the hard limit with bounded deterministic fallbacks."""

    current_messages = prepared_messages
    token_estimate = _request_tokens(system, current_messages, tools)
    events: list[ContextCompressionEvent] = []
    removed_total = 0

    for protect_latest, protected_current_token_limit in (
        (
            True,
            _current_frontier_token_limit(
                policy,
                token_limit=budget.prompt_budget,
                phase="hard",
            ),
        ),
        (False, 0),
    ):
        if token_estimate <= budget.hard_token_limit:
            break

        state, event, removed = _advance_execution_state(
            canonical_messages,
            state,
            system=system,
            tools=tools,
            projection_builder=projection_builder,
            target_tokens=budget.hard_token_limit,
            phase="hard",
            protect_latest_tool_group=protect_latest,
            protected_current_token_limit=protected_current_token_limit,
        )
        if event is not None:
            events.append(event)
            removed_total += removed
            current_messages = projection_builder(state)
            token_estimate = _request_tokens(system, current_messages, tools)

        if token_estimate <= budget.hard_token_limit:
            break

        fallback = _emergency_hard_projection(
            current_messages,
            system=system,
            tools=tools,
            budget=budget,
            policy=policy,
            tool_effects=tool_effects,
        )
        if fallback is not None:
            current_messages, fallback_event, removed = fallback
            token_estimate = _request_tokens(system, current_messages, tools)
            events.append(fallback_event)
            removed_total += removed

    return state, current_messages, token_estimate, events, removed_total


def _emergency_hard_projection(
    messages: list[dict[str, Any]],
    *,
    system: str,
    tools: list[dict[str, Any]],
    budget: TokenBudget,
    policy: CompactionPolicy,
    tool_effects: dict[str, dict[str, bool]] | None,
) -> tuple[
    list[dict[str, Any]],
    ContextCompressionEvent,
    int,
] | None:
    """Deterministically shrink an irreducible hard-overflow projection."""

    before_tokens = _request_tokens(system, messages, tools)
    plan = _select_compaction_groups(
        messages,
        system=system,
        tools=tools,
        token_limit=budget.hard_token_limit,
        allow_empty_summary=True,
        protected_current_token_limit=_current_frontier_token_limit(
            policy,
            token_limit=budget.hard_token_limit,
            phase="hard",
        ),
        policy=policy,
        tool_effects=tool_effects,
    )
    if plan is None:
        return None
    rebuilt, details = _apply_history_plan(
        plan,
        phase="hard",
        tool_effects=tool_effects,
    )
    visible = _model_visible_messages(rebuilt)
    after_tokens = _request_tokens(system, visible, tools)
    if after_tokens >= before_tokens:
        return None
    details["trigger"] = "hard_limit"
    event = ContextCompressionEvent(
        reason="execution_history",
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        details=details,
    )
    return visible, event, int(details.get("removed_groups") or 0)


def _advance_execution_state(
    canonical_messages: list[dict[str, Any]],
    state: SessionCompactionState,
    *,
    system: str,
    tools: list[dict[str, Any]],
    projection_builder: Callable[[SessionCompactionState], list[dict[str, Any]]],
    target_tokens: int,
    phase: str,
    protect_latest_tool_group: bool = True,
    protected_current_token_limit: int = 0,
) -> tuple[SessionCompactionState, ContextCompressionEvent | None, int]:
    """Advance one stable Tool Group boundary until the target is reached."""

    groups = canonical_groups(canonical_messages)
    if not groups:
        return state, None, 0
    current_boundary = valid_execution_boundary_index(groups, state.execution)
    raw_groups = [record.group for record in groups]
    active_user_index = find_latest_user_group_index(raw_groups)
    current_start = (
        active_user_index + 1
        if active_user_index is not None
        else 0
    )
    current_records = groups[current_start:]
    protected_indexes: set[int] = set()
    protected_groups: list[MessageGroup] = []
    if protect_latest_tool_group and current_records:
        _, protected_groups = _split_current_groups_for_compaction(
            [record.group for record in current_records],
            protected_token_limit=protected_current_token_limit,
        )
        protected_start = len(current_records) - len(protected_groups)
        protected_indexes = {
            record.index
            for record in current_records[protected_start:]
            if _is_tool_group(record.group)
        }
    candidate_indexes = [
        record.index
        for record in groups
        if _is_tool_group(record.group)
        and record.index > (current_boundary if current_boundary is not None else -1)
        and record.index not in protected_indexes
    ]
    if not candidate_indexes:
        return state, None, 0

    before_tokens = _request_tokens(system, projection_builder(state), tools)
    selected_state: SessionCompactionState | None = None
    selected_index: int | None = None
    selected_tokens = before_tokens
    for boundary_index in candidate_indexes:
        execution_state = execution_state_for_boundary(
            groups,
            boundary_index=boundary_index,
        )
        if execution_state is None:
            continue
        candidate = state.model_copy(deep=True)
        candidate.execution = execution_state
        candidate_tokens = _request_tokens(
            system,
            projection_builder(candidate),
            tools,
        )
        if candidate_tokens < selected_tokens:
            selected_state = candidate
            selected_index = boundary_index
            selected_tokens = candidate_tokens
        if candidate_tokens <= max(1, target_tokens):
            selected_state = candidate
            selected_index = boundary_index
            selected_tokens = candidate_tokens
            break

    if selected_state is None or selected_index is None:
        return state, None, 0
    newly_compacted = sum(
        1
        for index in candidate_indexes
        if index <= selected_index
    )
    execution_state = selected_state.execution
    assert execution_state is not None
    return (
        selected_state,
        ContextCompressionEvent(
            reason="execution_history",
            before_tokens=before_tokens,
            after_tokens=selected_tokens,
            details={
                "phase": phase,
                "strategy": "stable_tool_group_boundary",
                "changed": True,
                "removed_groups": newly_compacted,
                "boundary_group_id": execution_state.boundary_group_id,
                "target_tokens": max(1, target_tokens),
                "active_user_anchored": True,
                "protected_latest_tool_group": bool(protected_indexes),
                "protected_current_groups": len(protected_groups),
                "protected_current_tokens": _groups_tokens(protected_groups),
            },
        ),
        newly_compacted,
    )


def _is_tool_group(group: MessageGroup) -> bool:
    return bool(
        group
        and group[0].get("role") == "assistant"
        and group[0].get("tool_calls")
    )


def _join_system_messages(messages: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        str(message.get("content") or "").strip()
        for message in messages
        if message.get("role") == "system"
        and str(message.get("content") or "").strip()
    )


def _request_tokens(
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> int:
    payload = json.dumps(
        _model_visible_messages(messages),
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    tool_payload = json.dumps(
        tools,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return estimate_tokens(system) + estimate_tokens(payload) + estimate_tokens(tool_payload)


def _copy_task_projection(projection: TaskProjection) -> TaskProjection:
    user_task, open_rows = projection
    return str(user_task), [list(map(str, row)) for row in deepcopy(open_rows)]


def _task_free_compaction_source(
    messages: list[dict[str, Any]],
    task_projection: TaskProjection | None,
) -> list[dict[str, Any]]:
    if task_projection is None:
        return deepcopy(messages)
    return strip_task_protocol(messages)


def _apply_task_projection(
    messages: list[dict[str, Any]],
    task_projection: TaskProjection | None,
) -> list[dict[str, Any]]:
    if task_projection is None:
        return deepcopy(messages)
    user_task, open_rows = task_projection
    return insert_current_task_record(
        messages,
        user_task=user_task,
        open_rows=open_rows,
    )


def _has_semantic_source(groups: list[MessageGroup]) -> bool:
    for group in groups:
        for message in group:
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            if message.get("role") == "user":
                return True
            if is_final_assistant_message(message) or content.startswith(
                SEMANTIC_HISTORY_HEADING
            ):
                return True
    return False


def _select_compaction_groups(
    messages: list[dict[str, Any]],
    *,
    system: str,
    tools: list[dict[str, Any]],
    token_limit: int,
    allow_empty_summary: bool = True,
    protected_current_token_limit: int = 0,
    policy: CompactionPolicy,
    tool_effects: dict[str, dict[str, bool]] | None = None,
    task_projection: TaskProjection | None = None,
) -> HistoryCompactionPlan | None:
    """Select recent semantic Turns and current protocol groups under a limit."""

    groups = group_messages(messages)
    active_index = find_latest_user_group_index(groups)
    if active_index is None:
        return None

    active_group = groups[active_index : active_index + 1]
    current_groups = groups[active_index + 1 :]
    compressible_current_groups, protected_current_groups = (
        _split_current_groups_for_compaction(
            current_groups,
            protected_token_limit=protected_current_token_limit,
        )
    )
    semantic_turns = select_recent_semantic_turns(
        groups,
        active_user_index=active_index,
        max_turns=policy.recent_semantic_turns,
    )
    minimum_turns = 1 if semantic_turns else 0

    for turn_count in range(len(semantic_turns), minimum_turns - 1, -1):
        selected_turns = semantic_turns[-turn_count:] if turn_count else []
        semantic_indexes = {index for turn in selected_turns for index in turn}
        historical_groups = [
            group
            for index, group in enumerate(groups[:active_index])
            if index not in semantic_indexes
        ]
        raw_semantic_groups = [
            groups[index]
            for turn in selected_turns
            for index in turn
        ]
        for semantic_char_limit in (None, 6_000, 3_000, 1_500, 750, 320):
            semantic_groups = _compact_semantic_groups(
                raw_semantic_groups,
                max_chars=semantic_char_limit,
            )
            effective_token_limit = token_limit
            if not allow_empty_summary:
                effective_token_limit = max(
                    token_limit,
                    _request_tokens(
                        system,
                        _apply_task_projection(
                            flatten_groups(
                                [
                                    *semantic_groups,
                                    *active_group,
                                    *protected_current_groups,
                                ]
                            ),
                            task_projection,
                        ),
                        tools,
                    ),
                )

            candidate = HistoryCompactionPlan(
                historical_groups=historical_groups,
                semantic_tail_groups=semantic_groups,
                active_user_group=active_group,
                retained_current_groups=protected_current_groups,
                removed_current_groups=compressible_current_groups,
                removed_tokens=_groups_tokens(
                    [*historical_groups, *compressible_current_groups]
                ),
                protected_current_group_count=len(protected_current_groups),
                protected_current_tokens=_groups_tokens(protected_current_groups),
                compressible_current_group_count=len(
                    compressible_current_groups
                ),
                effective_token_limit=effective_token_limit,
                semantic_turns_retained=turn_count,
                semantic_message_char_limit=semantic_char_limit,
            )
            selected = _fit_summary_to_budget(
                candidate,
                system=system,
                tools=tools,
                token_limit=effective_token_limit,
                allow_empty_summary=allow_empty_summary,
                tool_effects=tool_effects,
                task_projection=task_projection,
            )

            if selected is not None:
                if (
                    not selected.historical_groups
                    and not selected.removed_current_groups
                    and selected.retained_current_groups == current_groups
                    and semantic_char_limit is None
                ):
                    return None
                return selected
    return None


def _current_frontier_token_limit(
    policy: CompactionPolicy,
    *,
    token_limit: int,
    phase: str,
) -> int:
    """Resolve one bounded current-frontier allowance for the active window."""

    configured = max(0, policy.current_frontier_token_limit)
    divisor = 4
    if phase in {"hard", "reactive"}:
        configured //= 2
        divisor = 2
    budget_cap = max(64, max(0, token_limit) // divisor)
    return min(configured, budget_cap)


def _split_current_groups_for_compaction(
    current_groups: list[MessageGroup],
    *,
    protected_token_limit: int,
) -> tuple[list[MessageGroup], list[MessageGroup]]:
    """Protect a newest suffix under Tokens, always keeping unconsumed evidence."""

    if not current_groups:
        return [], []
    returned_indexes = [
        index
        for index, group in enumerate(current_groups)
        if _is_returned_tool_protocol_group(group)
    ]
    if not returned_indexes:
        return [], list(current_groups)

    protected_start = returned_indexes[-1]
    token_limit = max(0, protected_token_limit)
    for candidate_start in range(protected_start - 1, -1, -1):
        candidate = current_groups[candidate_start:]
        if _groups_tokens(candidate) > token_limit:
            break
        protected_start = candidate_start
    return (
        list(current_groups[:protected_start]),
        list(current_groups[protected_start:]),
    )


def _is_returned_tool_protocol_group(group: MessageGroup) -> bool:
    if len(group) < 2:
        return False
    owner = group[0]
    if owner.get("role") != "assistant" or not owner.get("tool_calls"):
        return False
    expected_ids = {
        str(call.get("id"))
        for call in owner.get("tool_calls") or []
        if call.get("id") is not None
    }
    actual_ids = {
        str(message.get("tool_call_id"))
        for message in group[1:]
        if message.get("role") == "tool"
    }
    return (
        bool(expected_ids)
        and len(group) == len(expected_ids) + 1
        and all(message.get("role") == "tool" for message in group[1:])
        and actual_ids == expected_ids
    )


def _fit_summary_to_budget(
    plan: HistoryCompactionPlan,
    *,
    system: str,
    tools: list[dict[str, Any]],
    token_limit: int,
    allow_empty_summary: bool,
    tool_effects: dict[str, dict[str, bool]] | None,
    task_projection: TaskProjection | None = None,
) -> HistoryCompactionPlan | None:
    removed_groups = [*plan.historical_groups, *plan.removed_current_groups]
    full_summary, _ = compact_execution_history_with_details(
        removed_groups,
        tool_effects=tool_effects,
    )
    char_limits = _SUMMARY_CHAR_LIMITS[:-1] if full_summary else _SUMMARY_CHAR_LIMITS
    for char_limit in char_limits:
        candidate = HistoryCompactionPlan(
            historical_groups=plan.historical_groups,
            semantic_tail_groups=plan.semantic_tail_groups,
            active_user_group=plan.active_user_group,
            retained_current_groups=plan.retained_current_groups,
            removed_current_groups=plan.removed_current_groups,
            removed_tokens=plan.removed_tokens,
            protected_current_group_count=plan.protected_current_group_count,
            protected_current_tokens=plan.protected_current_tokens,
            compressible_current_group_count=plan.compressible_current_group_count,
            effective_token_limit=plan.effective_token_limit,
            summary_char_limit=char_limit,
            semantic_turns_retained=plan.semantic_turns_retained,
            semantic_message_char_limit=plan.semantic_message_char_limit,
        )
        rebuilt, _ = _apply_history_plan(
            candidate,
            phase="selection",
            tool_effects=tool_effects,
            task_projection=task_projection,
        )
        has_summary = any(
            str(message.get("content") or "").startswith(
                COMPACTED_EXECUTION_HEADING
            )
            for message in rebuilt
        )
        if full_summary and not has_summary:
            continue
        if _request_tokens(system, rebuilt, tools) <= token_limit:
            return candidate
    return None


def _rebuild_compacted_history(
    plan: HistoryCompactionPlan,
    historical_summary: str,
    current_summary: str,
) -> list[dict[str, Any]]:
    rebuilt: list[dict[str, Any]] = []
    if historical_summary.strip():
        rebuilt.append({"role": "assistant", "content": historical_summary})
    rebuilt.extend(flatten_groups(plan.semantic_tail_groups))
    rebuilt.extend(flatten_groups(plan.active_user_group))
    if current_summary.strip():
        rebuilt.append({"role": "assistant", "content": current_summary})
    rebuilt.extend(flatten_groups(plan.retained_current_groups))
    return rebuilt


def _compact_semantic_groups(
    groups: list[MessageGroup],
    *,
    max_chars: int | None,
) -> list[MessageGroup]:
    if max_chars is None:
        return deepcopy(groups)
    compacted: list[MessageGroup] = []
    for group in groups:
        copied = deepcopy(group)
        for message in copied:
            content = message.get("content")
            if isinstance(content, str) and len(content) > max_chars:
                message["content"] = _head_tail_compact(content, max_chars=max_chars)
        compacted.append(copied)
    return compacted


def _head_tail_compact(content: str, *, max_chars: int) -> str:
    marker = f"\n{SEMANTIC_COMPACTION_MARKER}\n"
    if max_chars <= len(marker) + 2:
        return marker.strip()
    remaining = max_chars - len(marker)
    head_chars = (remaining + 1) // 2
    tail_chars = remaining // 2
    return content[:head_chars] + marker + content[-tail_chars:]


def _groups_tokens(groups: list[MessageGroup]) -> int:
    if not groups:
        return 0
    return estimate_tokens(
        json.dumps(
            _model_visible_messages(flatten_groups(groups)),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    )


def _apply_history_plan(
    plan: HistoryCompactionPlan,
    *,
    phase: str,
    tool_effects: dict[str, dict[str, bool]] | None = None,
    task_projection: TaskProjection | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    current_summary, current_omitted = compact_execution_history_with_details(
        plan.removed_current_groups,
        max_chars=plan.summary_char_limit,
        tool_effects=tool_effects,
    )
    remaining_chars = max(0, plan.summary_char_limit - len(current_summary))
    historical_summary, historical_omitted = compact_execution_history_with_details(
        plan.historical_groups,
        max_chars=remaining_chars,
        tool_effects=tool_effects,
    )
    rebuilt = _rebuild_compacted_history(
        plan,
        historical_summary,
        current_summary,
    )
    rebuilt = _apply_task_projection(rebuilt, task_projection)
    execution_tokens = estimate_tokens(historical_summary) + estimate_tokens(
        current_summary
    )
    return rebuilt, {
        "phase": phase,
        "strategy": "deterministic_execution",
        "removed_groups": len(plan.historical_groups)
        + len(plan.removed_current_groups),
        "removed_tokens": plan.removed_tokens,
        "retained_groups": len(plan.retained_current_groups),
        "protected_current_groups": plan.protected_current_group_count,
        "protected_current_tokens": plan.protected_current_tokens,
        "compressible_current_groups": plan.compressible_current_group_count,
        "effective_token_limit": plan.effective_token_limit,
        "semantic_turns_retained": plan.semantic_turns_retained,
        "semantic_message_char_limit": plan.semantic_message_char_limit,
        "execution_record_tokens": execution_tokens,
        "omitted_execution_entries": historical_omitted + current_omitted,
        "active_user_anchored": True,
    }


def _request_source_tokens(
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, int]:
    return {
        "system": estimate_tokens(system),
        "messages": estimate_tokens(
            json.dumps(
                _model_visible_messages(messages),
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        ),
        "tools": estimate_tokens(
            json.dumps(tools, ensure_ascii=False, separators=(",", ":"), default=str)
        ),
    }


def _context_source_tokens(
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, int]:
    """Return a lightweight, trace-facing breakdown of the prepared request."""

    visible = _model_visible_messages(messages)
    latest_user_index = next(
        (
            index
            for index in range(len(visible) - 1, -1, -1)
            if visible[index].get("role") == "user"
        ),
        len(visible),
    )
    sources = {
        "system": estimate_tokens(system),
        "tools": estimate_tokens(
            json.dumps(
                tools,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        ),
        "historical_messages": 0,
        "current_turn": 0,
        "semantic_history": 0,
        "execution_record": 0,
        "current_tool_frontier": 0,
        "task_projection": 0,
        "runtime_notifications": 0,
    }
    for index, message in enumerate(visible):
        content = str(message.get("content") or "").strip()
        if content.startswith(SEMANTIC_HISTORY_HEADING):
            category = "semantic_history"
        elif content.startswith(COMPACTED_EXECUTION_HEADING):
            category = "execution_record"
        elif content.startswith(CURRENT_TOOL_FRONTIER_HEADING):
            category = "current_tool_frontier"
        elif content.startswith(CURRENT_TASK_HEADING):
            category = "task_projection"
        elif content.startswith("[MiniCode runtime notification]"):
            category = "runtime_notifications"
        elif index < latest_user_index:
            category = "historical_messages"
        else:
            category = "current_turn"
        sources[category] += estimate_tokens(
            json.dumps(
                message,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        )
    return sources


def _model_visible_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return deepcopy(messages)


def _usage_ratio(tokens: int, prompt_budget: int) -> float:
    if prompt_budget <= 0:
        return 0.0
    return round(tokens / prompt_budget, 6)
