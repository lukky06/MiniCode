# Skill: Test Generation

## When to use
Use this skill when the task asks to add new automated tests, expand branch coverage, reproduce a defect, or create a regression test in any language or framework.

## Test evidence record
Before selecting a strategy, record from inspected code:

- Target behavior: public function, method, route, command, component, or module.
- Observable contract: return value, emitted output, exception, state change, persisted data, or collaborator call.
- Branch inputs: the smallest inputs required to reach the selected branch.
- Dependency access: constructor arguments, imports, globals, dependency injection, factories, fixtures, or framework context.
- Side effects: files, network, database, cache, queue, timers, environment, process state, or UI rendering.
- Lifecycle resources: setup, teardown, temporary resources, event loops, threads, processes, and cleanup requirements.
- Existing test style: test location, naming, fixtures, mocks, assertions, and runner conventions.
- Focused verification: the narrowest command accepted by the repository toolchain.

Do not write a test until the selected behavior can be reached through an existing public interface with a compileable or runnable setup.

## Strategy selection
Choose the smallest strategy that tests the intended behavior:

- Pure unit test: deterministic logic with no external boundary.
- Mocked unit test: logic whose collaborators can be replaced using facilities already present in the repository.
- Component test: a small group of modules whose wiring is part of the behavior.
- Web boundary test: routing, binding, validation, authentication, serialization, and response mapping.
- Integration test: only when framework wiring, persistence, transactions, IPC, filesystem behavior, or real infrastructure is the behavior under test.
- End-to-end test: only when the task explicitly requires a full user-visible workflow and the repository already supports it.

Do not introduce a heavier framework merely because it makes setup easier.

## Branch selection
1. Identify the public entry point and enumerate relevant branches from source evidence.
2. Select branches that can be observed without unsupported private access or hidden runtime state.
3. Cover normal behavior, meaningful boundaries, and failure behavior requested by the task.
4. Avoid duplicate assertions that exercise the same path with equivalent inputs.
5. Exclude context-dependent branches unless the repository already provides a supported harness for that context.

## Mocking convention
- Mock at external or architectural boundaries, not inside the unit's own logic.
- Use the repository's installed mocking API and match exact call signatures.
- Do not invent setters, constructors, annotations, fixtures, or helper APIs.
- Do not replace immutable constants or unsupported runtime internals.
- Restore patched modules, globals, environment, timers, and process state after each test.
- Prefer fakes over deep mock chains when a small deterministic implementation is clearer.

## Delegation-contract coverage
When the target service mainly normalizes, validates, orders, paginates, or forwards inputs, use a mock, fake, spy, or recording stub to assert the downstream call directly.

- Verify normalized values, call count, ordering keys, offsets, limits, and omitted filters.
- Include `null`/`None`, empty, and whitespace-only inputs when the public contract distinguishes them.
- Do not use only a concrete in-memory dependency when it collapses distinct arguments such as `None` and `""` into the same observable result.
- Pair collaborator assertions with one returned-result assertion so the test covers both the owned boundary and user-visible behavior.

## Mockito/JUnit checks when applicable
- Initialize the system under test after Mockito has initialized `@Mock` fields, normally in `@BeforeEach` or inside each test.
- Stub `void` methods with `doThrow`, `doNothing`, or `doAnswer`; `when(voidCall)` is invalid.
- Use `verifyNoInteractions` only for true zero-call paths. If a lookup is expected but a write is forbidden, verify the lookup and use `never()` for the write, or use `verifyNoMoreInteractions` after all allowed calls.
- Prefer focused edits over replacing an entire existing test file when only setup, imports, or several cases need to change.

## Pre-write runnable review
Confirm all of the following before writing:

- Target symbols and imports exist.
- Test file placement and naming match runner discovery.
- Setup can instantiate or invoke the public entry point.
- Every helper-created domain object reuses a constructor, factory, and import pattern already proven runnable in the repository. If no such pattern is visible, inspect the exact definitions before writing; never infer field names or module re-exports from semantic aliases.
- Collaborators can be supplied using supported repository mechanisms.
- Assertions match actual types, errors, asynchronous behavior, and side effects.
- Cleanup prevents state leakage.
- The selected verification command is allowed and focused.

## Mutation-effective test review
When the task asks for mutation-effective, boundary-sensitive, or strong regression tests:

1. For each required behavior, name at least one plausible faulty implementation.
2. Choose an input for which the correct and faulty implementations produce different observable results.
3. Inspect constructors and value objects for normalization, clamping, truncation, or rounding before selecting boundary inputs.
4. Reject degenerate inputs that produce the same result whether the target guard or calculation exists or not.
5. Prefer a small set of discriminating tests over many branch-equivalent examples.

For numeric logic, calculate the expected value through both the intended and faulty paths before writing the assertion. For time boundaries, use the injected clock and values immediately before, at, and after the boundary only when those values distinguish behavior.

## Verification and repair
Run the focused test after writing. Classify failures in this order:

1. test discovery or import failure;
2. compile, parse, or type-check failure;
3. fixture, lifecycle, or environment failure;
4. mock setup or call-signature failure;
5. assertion failure revealing implementation behavior.

Repair the test unit first for categories 1–4. Read the first actionable diagnostic, group related import, lifecycle, and mock-syntax corrections into one coherent edit, then rerun the same focused command. Change production code only when the test accurately expresses the intended contract and the observed implementation violates it.

- Do not claim completion until the changed test compiles or imports and the focused runner executes it successfully.
- If a command is rejected, do not repeat the same argv or switch to a shell wrapper; use an allowed equivalent when the result provides one, otherwise report the concrete blocker.
- Once the focused test passes and the diff remains within task scope, stop exploring and return the final answer.
- When the remaining tool-call budget is low, spend it only on the final compile, test, or diff check needed to finish.
