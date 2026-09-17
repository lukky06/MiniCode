---
name: code-debug
description: Use this skill for failing tests, compiler or interpreter errors, runtime exceptions, incorrect behavior, build failures, and regression diagnosis in any supported language or toolchain.
---

# Skill: Code Debug

## When to use
Use this skill for failing tests, compiler or interpreter errors, runtime exceptions, incorrect behavior, build failures, and regression diagnosis in any supported language or toolchain.

## Evidence checklist
Prefer evidence from tool results in this order:

1. The latest failing verification output, including command, exit code, first actionable error, stack trace, compiler diagnostic, or assertion mismatch.
2. The named file, symbol, test, endpoint, or behavior from the task or diagnostic.
3. The smallest relevant source region around the failing symbol.
4. The smallest relevant test region around setup, inputs, assertions, and expected failures.
5. Build or dependency configuration only when the failure concerns parsing, dependency resolution, compilation, linking, plugins, or test discovery.
6. Nearby implementation style only when choosing the minimal patch shape.

## Failure classification
Classify the latest visible failure before changing code:

- `build_config_failure`: invalid project manifest, build script, or plugin configuration.
- `dependency_failure`: unresolved package, version conflict, missing module, or linker dependency.
- `compile_or_parse_failure`: compiler, type checker, parser, or syntax error.
- `test_runtime_failure`: the test starts but crashes or times out.
- `assertion_failure`: the test runs and observed behavior differs from the expectation.
- `behavioral_failure`: reproducible incorrect output without a test assertion.
- `environment_failure`: unavailable service, port, network, database, container, credential, or local runtime.

When verification fails before source execution, inspect the build or dependency artifact named by the diagnostic before reading unrelated code.

## Hypothesis-to-action gate

### Root-cause hypotheses
- Maintain no more than three active root-cause candidates at once.
- Each candidate must be falsifiable by one focused diagnostic, one local source read, or one minimal reversible edit.
- Prefer one diagnostic that can eliminate multiple candidates.

### Sufficient evidence
Stop expanding the investigation when all of the following are true:

- A specific code location can fully explain the observed failure.
- The location has a direct causal relationship to the failure mechanism.
- The required change is small and reversible.
- A focused test or diagnostic can verify the changed behavior.

### Action rule
- Once the evidence is sufficient, make the minimal change immediately.
- A minimal reversible patch may itself be used as a diagnostic experiment.
- If verification fails, revert or adjust the patch before investigating the next candidate.
- Do not enumerate every parent class, Mixin, Metaclass, caller, or complete call graph merely to obtain absolute certainty before editing.

### Environment failures
- Classify import, dependency, service, permission, and runtime setup failures as possible `environment_failure` before attributing them to production code.
- An environment failure does not by itself prove that production code is defective.
- Do not respond to an environment-command failure by expanding source reads without a bound.
- If reproduction is blocked, either make the smallest repair supported by static evidence or report the single concrete environment blocker.

## Fix convention
- Make the smallest behavior-preserving change supported by evidence.
- Fix production code when tests describe intended behavior.
- Fix tests only when the task asks to create or update tests, or when the expectation is contradicted by source or specification evidence.
- Do not weaken assertions, suppress errors, remove coverage, or bypass validation merely to make a command pass.
- Do not rewrite unrelated files or public interfaces without evidence.
- When adding a regression test for normalization or delegation logic, assert the collaborator call arguments directly instead of relying only on an in-memory implementation that may treat distinct values as equivalent.

## Test-repair discipline
When a focused verification fails because of the newly written test rather than production behavior:

- Repair the first actionable compile, import, lifecycle, or mock diagnostic before expanding source exploration.
- Group related import and setup corrections into one coherent edit before rerunning.
- With Mockito, construct the system under test after `@Mock` initialization, stub `void` methods with `doThrow`/`doAnswer`, and distinguish true zero-interaction paths from paths that permit reads but forbid writes.
- Prefer targeted edits to the existing test file over repeated whole-file rewrites.

## Verification convention
- Prefer the narrowest command supported by the detected repository toolchain.
- Re-run the same focused command after a repair when it remains applicable.
- If focused verification is unavailable, use the repository profile's preferred project-level command.
- Treat the latest command output as the primary next evidence after a failed verification.
- If a command is rejected, do not repeat the same argv or switch to a shell wrapper; use an allowed equivalent when the result provides one, otherwise report the concrete blocker.

## Completion convention
- When the minimal repair is applied, the focused verification passes, and the diff remains within task scope, stop investigating and return the final answer.
- Do not read additional callers, alternatives, or legacy implementations merely to increase confidence after those completion conditions are met.
- When the remaining tool-call budget is low, spend it only on the single action required to verify or report the current result.
