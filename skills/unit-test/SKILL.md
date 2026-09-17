---
name: unit-test
description: Use this skill when creating, repairing, or reviewing focused automated tests for a function, class, module, component, or small behavior boundary.
---

# Skill: Unit Test

## When to use
Use this skill when creating, repairing, or reviewing focused automated tests for a function, class, module, component, or small behavior boundary.

## Evidence checklist
Before writing or changing a test, inspect:

1. The public behavior and branch being tested.
2. Inputs, outputs, state changes, exceptions, and side effects visible to callers.
3. Constructor parameters, injected dependencies, module imports, globals, and lifecycle requirements.
4. Existing nearby tests and repository naming conventions.
5. The test framework, assertion library, mock facilities, and supported verification command.
6. Build or test configuration only when discovery, dependencies, plugins, or runtime setup require it.

## Contract authority and reconciliation
A self-authored test is a hypothesis about the intended behavior, not a new source of truth. Reconcile it against repository evidence before changing production code.

Use this priority when evidence conflicts:

1. Explicit user and task requirements.
2. Existing shared validators, domain rules, and public API contracts.
3. Established focused tests and representative call sites.
4. Assumptions introduced by the newly written test.

- Do not change production behavior solely to make a newly written test pass when that test conflicts with a shared validator or established public contract.
- After a self-authored test fails, re-read the relevant validator, public boundary, and task wording, then decide whether the test or implementation violates the intended contract before editing either one.
- When task wording is ambiguous, preserve an existing shared contract rather than silently inventing clamping, defaulting, coercion, or another incompatible behavior.
- If the task explicitly changes an existing contract, update production code and tests together and keep the contract change within the allowed scope.

## Boundary validation
When an existing validator owns an input boundary:

- Pass the original input through that validator unless the established contract explicitly requires normalization before validation.
- Test the exact valid boundary and the first invalid value on each relevant side.
- Assert that invalid input is rejected before downstream collaborators are called.
- Distinguish rejection, clamping, defaulting, and normalization; do not treat them as interchangeable implementations of a limit.

## Test design
- When tests are optional rather than required, implement the production behavior and run the existing focused verification first. Add a test only when it closes a meaningful verification gap.
- Test observable behavior rather than private implementation details.
- Keep each test focused on one behavior or one meaningful branch.
- Reuse repository fixtures and helpers when they make the setup clearer.
- Isolate external systems with the repository's existing mocking or fake strategy.
- Do not start a full application, container, browser, database, or network service unless that integration is the behavior under test.
- Avoid sleeps, wall-clock dependence, random inputs without a fixed seed, and order-dependent shared state.
- Restore patched globals, environment variables, clocks, singleton state, and temporary resources after each test.

## Delegation-contract tests
When the unit primarily normalizes, validates, orders, paginates, or delegates arguments to a collaborator, assert the collaborator contract directly with a mock, fake, spy, or recording stub.

- Verify normalized argument values, call count, ordering keys, offsets, limits, and omitted filters.
- Cover `null`/`None`, empty, and whitespace-only inputs when they have distinct contract meanings.
- Do not rely only on a concrete in-memory collaborator when it treats distinct arguments such as `None` and `""` as equivalent and can hide a delegation defect.
- Keep one behavior-level test for the returned result, but add a direct interaction assertion for the boundary owned by the unit.

## Mockito/JUnit checks when applicable
- Construct the system under test only after `@Mock` fields have been initialized, normally in `@BeforeEach` or inside the test.
- Stub `void` methods with `doThrow`, `doNothing`, or `doAnswer`; do not use `when(voidCall)`.
- Use `verifyNoInteractions` only when no collaborator call is expected. After an expected lookup, use a specific `never()` assertion for the forbidden side effect or `verifyNoMoreInteractions` after verifying allowed calls.
- Prefer focused edits to an existing test file over a whole-file rewrite when only imports, setup, or a few cases need changing.

## Compileability and discovery review
Before writing, verify that:

- imports and symbols exist;
- constructor or factory calls match the implementation;
- mock APIs match the repository's installed framework version;
- assertions use the actual return type and error behavior;
- test names and locations are discoverable by the detected runner;
- no private method, nonexistent setter, unsupported static patch, uninstalled assertion library, or invented value-object constructor is required.

## Verification
Run the narrowest allowed command that executes the changed test. If the test does not compile, parse, import, or discover, repair the test unit before changing production behavior. Read the first actionable diagnostic and repair related imports, lifecycle setup, and mock syntax in one coherent edit before rerunning.

- Do not claim the test is complete until the changed test compiles or imports and the focused runner executes it successfully.
- If a command is rejected, do not repeat the same argv or switch to a shell wrapper; use an allowed equivalent when the result provides one, otherwise report the concrete blocker.
- After the focused test passes and the diff remains within task scope, stop exploring and return the final answer.
- When the remaining tool-call budget is low, use it only for the single compile, test, or diff check needed to finish.
